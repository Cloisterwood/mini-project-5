"""Freeze the three-run decision using validation only, before S7."""

import math
from datetime import datetime, timezone
from pathlib import Path

from dating_compat.data import DataError, digest, file_hash, parse_json, write_immutable
from dating_compat.train import evaluation_read_guard
from dating_compat.validation import validate_run

RUN_IDS = ("run-001", "run-002", "run-003")


def rank_candidates(reports: dict) -> tuple[list[str], str]:
    if set(reports) != set(RUN_IDS):
        raise DataError("Selection requires exactly three completed runs")
    for run_id, report in reports.items():
        for key in ("cohens_d", "fpr"):
            value = report["metrics"][key]
            if value is None or isinstance(value, bool) or not math.isfinite(value):
                raise DataError(f"Undefined selection metric for {run_id}: {key}")
    primary = [r for r in RUN_IDS if all(x["met"] is True for x in reports[r]["targets"]["final_checklist"].values())]
    fpr_eligible = [r for r in RUN_IDS if reports[r]["metrics"]["fpr"] <= .35]
    eligible = primary or fpr_eligible or list(RUN_IDS)
    mode = "primary_targets_met" if primary else "fallback_fpr_constraint" if fpr_eligible else "diagnostic_unmet_constraints"
    return sorted(eligible, key=lambda r: (-reports[r]["metrics"]["cohens_d"], reports[r]["metrics"]["fpr"], r)), mode


def freeze_selection() -> dict:
    with evaluation_read_guard():
        root = Path("reports/runs")
        # Fail before inference if any required experiment is absent/incomplete.
        for run_id in RUN_IDS:
            path = root / run_id / "training_result.json"
            if not path.exists() or parse_json(path.read_text())["status"] != "completed":
                raise DataError("Selection requires exactly three completed runs")
            if not (root / run_id / "validation_manifest.json").exists():
                raise DataError("All three runs require sealed validation evidence")
        reports = {run_id: validate_run(run_id) for run_id in RUN_IDS}
        manifests = {r: parse_json((root / r / "run_manifest.json").read_text()) for r in RUN_IDS}
        first = manifests[RUN_IDS[0]]
        for manifest in manifests.values():
            for key in ("fit_ids", "validation_ids", "preflight_manifest_sha256", "software", "device"):
                if manifest[key] != first[key]:
                    raise DataError("Run cohort/environment differs before selection")
            for key in ("model_id", "revision", "loss", "seed", "precision", "max_seq_length"):
                if manifest["config"][key] != first["config"][key]:
                    raise DataError("Run model/preprocessing differs before selection")
        ranking, mode = rank_candidates(reports)
        chosen = ranking[0]
        result = parse_json((root / chosen / "training_result.json").read_text())
        bound_names = ("run_manifest.json", "training_result.json", "validation_manifest.json", "validation_metrics.json", "selected_checkpoint_manifest.json")
        binding = {r: {n: file_hash(root / r / n) for n in bound_names} for r in RUN_IDS}
        decision = {"version": "s6-selection-v1", "selection_split": "validation", "selected_run": chosen,
            "selection_mode": mode, "eligible_ranked_runs": ranking,
            "rule": "All four primary targets first, then highest validation Cohen d, lower FPR tie-break; otherwise FPR<=.35 fallback, otherwise highest d diagnostic; exact ties by run ID",
            "binding": binding, "validation_ids_sha256": digest(first["validation_ids"]),
            "preflight_manifest_sha256": first["preflight_manifest_sha256"],
            "selected_path": result["selected_path"], "selected_trainer_checkpoint": result["best_model_checkpoint"],
            "selected_weights_sha256": result["selected_weights_sha256"],
            "selected_config": manifests[chosen]["config"],
            "candidates": {r: {"metrics": reports[r]["metrics"], "coverage_adjusted_purity": reports[r]["clustering"]["coverage_adjusted_purity"], "targets": reports[r]["targets"]} for r in RUN_IDS},
            "evaluation_used_for_selection": False,
            "limitations": ["One seed per configuration; differences do not establish significance", "Validation used for checkpoint/run selection; final holdout comparison deferred to S7", "No retraining on all data; selected checkpoint remains final"]}
        path = Path("reports/selected_run_manifest.json")
        if path.exists():
            saved = parse_json(path.read_text())
            if any(saved.get(k) != value for k, value in decision.items()) or digest({k: v for k, v in saved.items() if k != "manifest_sha256"}) != saved.get("manifest_sha256"):
                raise DataError("Frozen selection differs from current evidence")
            return saved
        decision["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        decision["selection_code_sha256"] = file_hash(Path(__file__))
        decision["manifest_sha256"] = digest(decision)
        write_immutable(path, decision)
        return decision

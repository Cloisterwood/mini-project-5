"""Separate preflight certifies fit/validation inputs; training never opens evaluation data."""

import json
from pathlib import Path

from dating_compat.data import DataError, digest, file_hash, parse_json, write_immutable

PACKAGE = Path("data/training-ready")
CERTIFICATE = Path("reports/training_preflight_manifest.json")


def prepare_training(data_dir: Path = Path("data"), package: Path = PACKAGE,
                     certificate: Path = CERTIFICATE) -> dict:
    # This is the separate governance command. It may validate held-out sources,
    # but exports only fit/validation rows to the worker's input package.
    from dating_compat.cohort import load_eligible_bundle
    from dating_compat.quality import require_quality_gate
    from dating_compat.evaluate import baseline, split_records, write_bytes_immutable
    quality = require_quality_gate(data_dir)
    bundle = load_eligible_bundle(data_dir)
    if not Path("reports/baseline_manifest.json").exists():
        raise DataError("Complete Stage 4 before training preflight")
    baseline(data_dir)  # Existing seal verified; never runs inference here.
    files, ids = {}, {}
    for split in ("fit", "validation"):
        rows = split_records(bundle, split)
        path = package / f"{split}.jsonl"
        content = "".join(json.dumps({"record_id": r.id, "text_1": r.pair.text_1,
                                     "text_2": r.pair.text_2, "label": float(r.pair.label)},
                                    ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
        write_bytes_immutable(path, content.encode("utf-8"))
        files[str(path)] = file_hash(path)
        ids[split] = [r.id for r in rows]
    proof_paths = ["reports/quality_report.json", "configs/quality-rubric.yaml",
        "reports/quality_rubric_manifest.json", "reports/human_review_sample.json",
        "docs/human_review_responses.csv", "reports/quarantine-v1/cohort_manifest.json",
        "reports/quarantine-v1/split_manifest.json", "configs/quarantine-v1.json",
        "reports/baseline_manifest.json", "reports/base_model_manifest.json",
        "configs/baseline.json", "reports/baseline_config_manifest.json",
        str(data_dir / "dating_pairs.jsonl"), str(data_dir / "dating_pairs_metadata.json")]
    report = {"version": "s5-preflight-v1", "quality_passed": quality["gate"]["passed"],
              "quality_report_sha256": quality["report_sha256"], "source_hashes": bundle["hashes"],
              "proof_files": {p: file_hash(Path(p)) for p in proof_paths}, "package_files": files,
              "ids": ids, "evaluation_access": "Governance preflight only; no evaluation text or labels exported"}
    report["manifest_sha256"] = digest(report)
    write_immutable(certificate, report)
    return report


def load_training_inputs(package: Path = PACKAGE, certificate: Path = CERTIFICATE) -> tuple[dict, dict]:
    """Check receipt and current allowed inputs without opening any evaluation file."""
    saved = parse_json(certificate.read_text(encoding="utf-8"))
    if saved.get("version") != "s5-preflight-v1" or saved.get("quality_passed") is not True:
        raise DataError("Training requires a passing preflight certificate")
    if digest({k: v for k, v in saved.items() if k != "manifest_sha256"}) != saved.get("manifest_sha256"):
        raise DataError("Preflight certificate checksum mismatch")
    # Fixed allowlist prevents a modified certificate from directing the worker
    # to evaluation sources. Original training paths can use a different data dir.
    allowed_names = {"dating_pairs.jsonl", "dating_pairs_metadata.json"}
    allowed_reports = {"reports/quality_report.json", "configs/quality-rubric.yaml",
        "reports/quality_rubric_manifest.json", "reports/human_review_sample.json",
        "docs/human_review_responses.csv", "reports/quarantine-v1/cohort_manifest.json",
        "reports/quarantine-v1/split_manifest.json", "configs/quarantine-v1.json",
        "reports/baseline_manifest.json", "reports/base_model_manifest.json",
        "configs/baseline.json", "reports/baseline_config_manifest.json"}
    if not allowed_reports <= set(saved["proof_files"]):
        raise DataError("Preflight proof files missing")
    for name, expected in saved["proof_files"].items():
        path = Path(name)
        if name not in allowed_reports and path.name not in allowed_names:
            raise DataError("Training proof path is outside allowed fit/validation governance inputs")
        if path.is_symlink() or file_hash(path) != expected:
            raise DataError(f"Training preflight stale or altered: {name}; rerun governance preparation")
    split_manifest = parse_json(Path("reports/quarantine-v1/split_manifest.json").read_text(encoding="utf-8"))
    cohort = parse_json(Path("reports/quarantine-v1/cohort_manifest.json").read_text(encoding="utf-8"))
    result = {}
    for split in ("fit", "validation"):
        path = package / f"{split}.jsonl"
        if path.is_symlink() or saved["package_files"].get(str(path)) != file_hash(path):
            raise DataError("Prepared training package checksum mismatch")
        rows = [parse_json(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if any(set(row) != {"record_id", "text_1", "text_2", "label"} or
               type(row["label"]) is not float or row["label"] not in (0., 1.) or
               any(not isinstance(row[k], str) or not row[k].strip() for k in ("record_id", "text_1", "text_2"))
               for row in rows):
            raise DataError("Training rows require two raw text columns and float binary labels only")
        ids = [row["record_id"] for row in rows]
        if ids != saved["ids"][split] or ids != split_manifest[f"{split}_ids"] or len(ids) != len(set(ids)):
            raise DataError("Training package IDs/order differ from frozen split")
        if set(ids) & (set(cohort["evaluation_ids"]) | set(cohort["excluded_training_ids"])):
            raise DataError("Evaluation/quarantine IDs cannot enter training inputs")
        result[split] = rows
    if set(saved["ids"]["fit"]) & set(saved["ids"]["validation"]):
        raise DataError("Fit/validation overlap")
    return result, saved

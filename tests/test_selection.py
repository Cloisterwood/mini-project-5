import pytest

from dating_compat.data import DataError
from dating_compat.selection import freeze_selection, rank_candidates


def report(d, fpr, met):
    return {"metrics": {"cohens_d": d, "fpr": fpr}, "targets": {"final_checklist": {"margin": {"met": met}, "cohens_d": {"met": met}, "fpr": {"met": met}, "cluster_purity": {"met": met}}}}


def test_primary_constraints_before_larger_d():
    runs = {"run-001": report(1, .2, True), "run-002": report(4, .2, False), "run-003": report(2, .1, True)}
    assert rank_candidates(runs) == (["run-003", "run-001"], "primary_targets_met")


def test_fpr_fallback_and_tie_break():
    runs = {"run-001": report(2, .3, False), "run-002": report(5, .4, False), "run-003": report(2, .2, False)}
    assert rank_candidates(runs) == (["run-003", "run-001"], "fallback_fpr_constraint")


def test_diagnostic_and_undefined_metrics():
    runs = {"run-001": report(1, .4, False), "run-002": report(3, .5, False), "run-003": report(2, .6, False)}
    assert rank_candidates(runs) == (["run-002", "run-003", "run-001"], "diagnostic_unmet_constraints")
    runs["run-003"]["metrics"]["cohens_d"] = None
    with pytest.raises(DataError, match="Undefined"):
        rank_candidates(runs)


def test_incomplete_run_budget_fails_before_validation(tmp_path, monkeypatch):
    from dating_compat import selection
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(selection, "validate_run", lambda *args: pytest.fail("Must not validate before all runs complete"))
    with pytest.raises(DataError, match="three completed"):
        freeze_selection()
    with pytest.raises(DataError, match="three completed"):
        rank_candidates({"run-001": report(1, .2, True)})


def test_freeze_repeat_and_changed_evidence(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from dating_compat import selection
    monkeypatch.chdir(tmp_path)
    reports = {}
    for index, run_id in enumerate(selection.RUN_IDS, 1):
        reports[run_id] = {**report(index, .2, True), "clustering": {"coverage_adjusted_purity": .8}}
        folder = Path("reports/runs") / run_id
        folder.mkdir(parents=True)
        config = {"model_id": "base", "revision": "sha", "loss": "CosineSimilarityLoss", "seed": 42, "precision": "float32", "max_seq_length": 256}
        manifest = {"config": config, "fit_ids": ["fit"], "validation_ids": ["val"], "preflight_manifest_sha256": "preflight", "software": {}, "device": "cpu"}
        (folder / "run_manifest.json").write_text(json.dumps(manifest))
        (folder / "training_result.json").write_text(json.dumps({"status": "completed", "selected_path": f"checkpoints/{run_id}/selected", "best_model_checkpoint": f"checkpoints/{run_id}/checkpoint-10", "selected_weights_sha256": f"weights-{run_id}"}))
        for name in ("validation_manifest.json", "validation_metrics.json", "selected_checkpoint_manifest.json"):
            (folder / name).write_text("{}")
    monkeypatch.setattr(selection, "validate_run", lambda run_id: reports[run_id])
    first = freeze_selection()
    assert first["selected_run"] == "run-003" and first["evaluation_used_for_selection"] is False
    assert freeze_selection() == first
    Path("reports/runs/run-003/validation_metrics.json").write_text('{"changed":true}')
    with pytest.raises(DataError, match="Frozen selection differs"):
        freeze_selection()

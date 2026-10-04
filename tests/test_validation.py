import csv
import json
from pathlib import Path

import pytest

from dating_compat.data import DataError, digest, file_hash
from dating_compat.metrics import pair_metrics
from dating_compat.validation import aligned_baseline, signed_deltas, validate_run


def test_deltas_preserve_sign_and_nulls():
    before = {key: .3 for key in ("margin", "cohens_d", "fpr", "accuracy", "precision", "recall", "f1", "auc")}
    after = {**before, "cohens_d": .9, "fpr": .1, "precision": None}
    result = signed_deltas(before, after)
    assert result["cohens_d"]["delta"] == pytest.approx(.6)
    assert result["fpr"]["delta"] == pytest.approx(-.2)
    assert result["precision"]["delta"] is None


def test_baseline_must_have_same_ids_and_sealed_predictions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("reports/validation_baseline_metrics.json")
    csv_path = Path("reports/baseline/baseline_validation_predictions.csv")
    csv_path.parent.mkdir(parents=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "label", "cosine"])
        writer.writeheader(); writer.writerows([{"record_id": "a", "label": 0, "cosine": .1}, {"record_id": "b", "label": 1, "cosine": .8}])
    path.write_text(json.dumps({"split": "validation", "ordered_record_ids_sha256": digest(["a", "b"]), "metrics": pair_metrics([0, 1], [.1, .8])}))
    seal = {"artifacts": {str(p): file_hash(p) for p in (path, csv_path)}}
    report, scores = aligned_baseline(path, ["a", "b"], seal)
    assert report["metrics"]["auc"] == 1 and list(scores) == [.1, .8]
    with pytest.raises(DataError, match="IDs/order"):
        aligned_baseline(path, ["b", "a"], seal)
    csv_path.write_text("tampered")
    with pytest.raises(DataError, match="sealed"):
        aligned_baseline(path, ["a", "b"], seal)


def test_validation_requires_gate_before_loading_model(monkeypatch):
    from dating_compat import validation
    def blocked():
        raise DataError("Stale preflight")
    monkeypatch.setattr(validation, "load_training_inputs", blocked)
    monkeypatch.setattr(validation, "_model", lambda *args: pytest.fail("Must not load model"))
    with pytest.raises(DataError, match="Stale"):
        validate_run("run-001")
    with pytest.raises(DataError, match="Run ID"):
        validate_run("../../unsafe")

import json
from pathlib import Path

import pytest

from dating_compat.data import DataError, digest, file_hash
from dating_compat.train import evaluation_read_guard, load_config, model_dataset, sample_rows, train_run, train_smoke
from dating_compat.training_data import load_training_inputs


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    proof_paths = ["reports/quality_report.json", "configs/quality-rubric.yaml",
        "reports/quality_rubric_manifest.json", "reports/human_review_sample.json",
        "docs/human_review_responses.csv", "reports/quarantine-v1/cohort_manifest.json",
        "reports/quarantine-v1/split_manifest.json", "configs/quarantine-v1.json",
        "reports/baseline_manifest.json", "reports/base_model_manifest.json",
        "configs/baseline.json", "reports/baseline_config_manifest.json"]
    for name in proof_paths:
        path = Path(name); path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{}")
    split = {"fit_ids": ["fit-1", "fit-2"], "validation_ids": ["val-1"]}
    Path("reports/quarantine-v1/split_manifest.json").write_text(json.dumps(split))
    Path("reports/quarantine-v1/cohort_manifest.json").write_text(json.dumps({"evaluation_ids": ["eval-1"], "excluded_training_ids": ["quarantine-1"]}))
    files = {}
    for partition in ("fit", "validation"):
        path = Path("data/training-ready") / f"{partition}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({"record_id": i, "text_1": "Raw first", "text_2": "Raw second", "label": float(n % 2)}) + "\n"
                                for n, i in enumerate(split[f"{partition}_ids"])))
        files[str(path)] = file_hash(path)
    receipt = {"version": "s5-preflight-v1", "quality_passed": True,
               "proof_files": {p: file_hash(Path(p)) for p in proof_paths}, "package_files": files,
               "ids": {k: split[f"{k}_ids"] for k in ("fit", "validation")}}
    receipt["manifest_sha256"] = digest(receipt)
    Path("reports/training_preflight_manifest.json").write_text(json.dumps(receipt))
    return receipt


def test_inputs_and_worker_read_no_evaluation(prepared):
    with evaluation_read_guard():
        rows, receipt = load_training_inputs()
        assert len(rows["fit"]) == 2 and len(rows["validation"]) == 1
        assert receipt["manifest_sha256"] == prepared["manifest_sha256"]
        for path in ("data/eval_pairs.jsonl", "data/eval_pairs_metadata.json", "reports/baseline_metrics.json"):
            with pytest.raises(DataError, match="inaccessible"):
                Path(path).open()


def test_stale_human_quality_evidence_blocks(prepared):
    Path("docs/human_review_responses.csv").write_text("changed")
    with pytest.raises(DataError, match="stale"):
        load_training_inputs()


def test_changed_training_package_blocks(prepared):
    Path("data/training-ready/fit.jsonl").write_text("changed")
    with pytest.raises(DataError, match="checksum"):
        load_training_inputs()


def test_forged_pass_flag_blocks(prepared):
    prepared["quality_passed"] = False
    Path("reports/training_preflight_manifest.json").write_text(json.dumps(prepared))
    with pytest.raises(DataError, match="passing"):
        load_training_inputs()


def test_worker_rejects_certificate_evaluation_path(prepared):
    prepared["proof_files"]["data/eval_pairs.jsonl"] = "fake"
    prepared["manifest_sha256"] = digest({k: v for k, v in prepared.items() if k != "manifest_sha256"})
    Path("reports/training_preflight_manifest.json").write_text(json.dumps(prepared))
    with pytest.raises(DataError, match="outside allowed"):
        load_training_inputs()


def test_smoke_sample_reproducible_and_model_columns_only():
    rows = [{"record_id": str(i), "text_1": f"first {i}", "text_2": "second", "label": float(i % 2)} for i in range(100)]
    selected = sample_rows(rows, 32, 42)
    assert selected == sample_rows(rows, 32, 42) and len({r["record_id"] for r in selected}) == 32
    dataset = model_dataset(selected)
    assert dataset.column_names == ["text_1", "text_2", "label"]
    assert dataset.features["label"].dtype == "float32"
    with pytest.raises(DataError):
        sample_rows(rows, 101, 42)


def test_smoke_and_run_require_preflight_before_model(monkeypatch):
    from dating_compat import train
    def blocked():
        raise DataError("Missing preflight")
    monkeypatch.setattr(train, "load_training_inputs", blocked)
    monkeypatch.setattr(train, "_model", lambda *args: pytest.fail("Must not load model"))
    with pytest.raises(DataError, match="preflight"):
        train_smoke()
    with pytest.raises(DataError, match="preflight"):
        train_run("run-001", Path("unused"), "cpu")


def test_invalid_run_id_rejected():
    with pytest.raises(DataError, match="Run ID"):
        train_run("../../overwrite", Path("unused"), "cpu")


@pytest.mark.parametrize("change", [{"label": 1}, {"label": True}, {"category": "metadata must not enter Trainer"}])
def test_resealed_invalid_model_columns_rejected(prepared, change):
    path = Path("data/training-ready/fit.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(change)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prepared["package_files"][str(path)] = file_hash(path)
    prepared["manifest_sha256"] = digest({k: v for k, v in prepared.items() if k != "manifest_sha256"})
    Path("reports/training_preflight_manifest.json").write_text(json.dumps(prepared))
    with pytest.raises(DataError, match="float binary labels only"):
        load_training_inputs()


def test_missing_checkpoint_state_rejected(tmp_path):
    from dating_compat.train import checkpoint_receipt
    with pytest.raises(DataError, match="lacks"):
        checkpoint_receipt(tmp_path, 8)


def test_config_matches_pinned_contract():
    config = load_config()
    assert config["epochs"] == 4 and config["batch_size"] == 16
    assert config["loss"] == "CosineSimilarityLoss" and config["precision"] == "float32"
    assert config["learning_rate"] == 2e-5 and config["warmup_ratio"] == .1

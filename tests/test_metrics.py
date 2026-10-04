import json
import math

import numpy as np
import pytest

from dating_compat.cluster import cluster_purity, pair_features, pair_scores, project_and_cluster
from dating_compat.data import DataError, Pair, Record, digest
from dating_compat.evaluate import checked_predictions, embedding_cache, frozen_config, split_records, verify_seal
from dating_compat.metrics import breakdown, pair_metrics, target_results


def records():
    return [Record(str(i), "fixture.jsonl", i + 1,
                   Pair(text_1=f"First {i}", text_2=f"Second {i}", label=i % 2,
                        category="values" if i < 2 else "lifestyle", subcategory="test", pair_type="test"))
            for i in range(4)]


def test_hand_calculated_classification_and_continuous_auc():
    m = pair_metrics([0, 0, 1, 1], [.5, .1, .6, .4])
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["fpr"] == .5 and m["fpr_numerator"] == 1 and m["fpr_denominator"] == 2
    assert m["accuracy"] == m["precision"] == m["recall"] == m["f1"] == .5
    assert m["auc"] == .75  # Binary predictions would give .5.
    assert m["margin"] == pytest.approx(.2)


def test_hand_calculated_weighted_sample_variance_and_signed_d():
    y, scores = [1, 1, 1, 0, 0], [.5, .7, .9, .1, .3]
    m = pair_metrics(y, scores)
    assert m["pooled_sample_std"] == pytest.approx(math.sqrt(1 / 30))
    assert m["margin"] == pytest.approx(.5)
    assert m["cohens_d"] == pytest.approx(math.sqrt(7.5))
    flipped = pair_metrics([1 - label for label in y], scores)
    assert flipped["cohens_d"] == pytest.approx(-math.sqrt(7.5))


def test_threshold_and_auc_ties():
    m = pair_metrics([0, 1], [.5, .5])
    assert m["counts"] == {"tn": 0, "fp": 1, "fn": 0, "tp": 1}
    assert m["auc"] == .5
    assert m["cohens_d"] is None


def test_undefined_cases_are_json_null_with_reasons():
    for labels, scores in (([0, 0], [.1, .2]), ([1, 1], [.1, .2]), ([0, 0, 1, 1], [.1] * 4)):
        m = pair_metrics(labels, scores)
        json.dumps(m, allow_nan=False)
        assert m["cohens_d"] is None and m["undefined"]["cohens_d"]
    m = pair_metrics([0, 0], [.1, .2])
    assert m["margin"] is m["recall"] is m["precision"] is m["f1"] is m["auc"] is None
    m = pair_metrics([1, 1], [.1, .2])
    assert m["fpr"] is None and m["fpr_denominator"] == 0
    assert m["recall"] == m["f1"] == 0  # Defined denominators retain real zero.


@pytest.mark.parametrize("labels,scores", [([], []), ([0], []), ([True], [.2]), (["1"], [.2]),
                                          ([2], [.2]), ([0], [float("nan")]), ([0], [1.1])])
def test_invalid_metrics_rejected(labels, scores):
    with pytest.raises(DataError):
        pair_metrics(labels, scores)


def test_symmetric_cosine_and_features_and_scale_invariance():
    a, b = np.array([[1, 0], [1, 1]]), np.array([[0, 2], [2, 0]])
    assert pair_scores(a, b) == pytest.approx([0, 1 / math.sqrt(2)])
    assert np.array_equal(pair_features(a, b), pair_features(b, a))
    assert np.array_equal(pair_scores(a, b), pair_scores(b, a))
    assert np.allclose(pair_features(a, b), pair_features(a * 3, b * 3))
    with pytest.raises(DataError, match="Zero-length"):
        pair_scores([[0, 0]], [[1, 0]])
    with pytest.raises(DataError, match="Nonfinite"):
        pair_features([[float("nan"), 1]], [[1, 0]])


def test_hand_calculated_noise_coverage_and_purity():
    m = cluster_purity([1, 0, 1, 0, 1], [0, 0, 1, 1, -1])
    assert m["coverage_adjusted_purity"] == .4
    assert m["clustered_only_purity"] == .5
    assert m["noise_fraction"] == .2
    assert m["cluster_count"] == 2 and m["largest_cluster_size"] == 2
    assert m["observed_label_majority_baseline"] == .6
    noise = cluster_purity([0, 1], [-1, -1])
    assert noise["coverage_adjusted_purity"] == 0
    assert noise["clustered_only_purity"] is None and noise["clustered_only_undefined_reason"]
    assert cluster_purity([1, 1], [0, 0])["coverage_adjusted_purity"] == 1


def test_exact_category_breakdown_and_prediction_order():
    rows = records()
    scores = np.array([.1, .9, .6, .8])
    predictions = checked_predictions(rows, scores, np.zeros((4, 2)), np.array([-1, 0, 0, 0]))
    assert [p["record_id"] for p in predictions] == [r.id for r in rows]
    assert [p["prediction"] for p in predictions] == [0, 1, 1, 1]
    assert breakdown(rows, scores)["category"]["lifestyle"]["fpr"] == 1
    with pytest.raises(DataError, match="aligned"):
        checked_predictions(rows, scores[:3], np.zeros((4, 2)), np.zeros(4, dtype=int))
    with pytest.raises(DataError, match="unique"):
        checked_predictions([rows[0]] * 4, scores, np.zeros((4, 2)), np.zeros(4, dtype=int))


def test_frozen_split_ids_and_unknown_split():
    rows = records()
    bundle = {"records": {"train": rows, "eval": rows[:1]},
              "split_manifest": {"fit_ids": ["2", "0"], "validation_ids": ["3", "1"]}}
    assert [r.id for r in split_records(bundle, "fit")] == ["2", "0"]
    assert [r.id for r in split_records(bundle, "validation")] == ["3", "1"]
    assert split_records(bundle, "eval") == rows[:1]
    with pytest.raises(DataError, match="Unknown"):
        split_records(bundle, "training")


def test_config_frozen_before_model_and_cannot_change(tmp_path):
    path, manifest = tmp_path / "config.json", tmp_path / "freeze.json"
    config = {"revision": "a" * 40, "preprocessing": {"threshold": .5}}
    path.write_text(json.dumps(config))
    frozen_config(path, manifest)
    assert json.loads(manifest.read_text())["frozen_before_model_outcomes"]
    config["seed"] = 12
    path.write_text(json.dumps(config))
    with pytest.raises(DataError):
        frozen_config(path, manifest)


def test_seal_rejects_changed_inputs_or_artifacts(tmp_path):
    from dating_compat.data import file_hash
    artifact, path = tmp_path / "predictions.csv", tmp_path / "seal.json"
    artifact.write_text("original\n")
    inputs = {"dataset": "fixed"}
    seal = {"inputs": inputs, "artifacts": {str(artifact): file_hash(artifact)}}
    seal["manifest_sha256"] = digest(seal)
    path.write_text(json.dumps(seal))
    verify_seal(path, inputs)
    with pytest.raises(DataError, match="provenance"):
        verify_seal(path, {"dataset": "changed"})
    artifact.write_text("modified\n")
    with pytest.raises(DataError, match="changed"):
        verify_seal(path, inputs)


def test_cached_embeddings_are_text_only_and_bound_to_preprocessing(tmp_path):
    class Model:
        max_seq_length = 256
        calls = 0
        def tokenizer(self, texts, truncation):
            assert truncation is False
            return {"input_ids": [[1, 2, 3]] * len(texts)}
        def encode(self, texts, **kwargs):
            self.calls += 1
            assert all(isinstance(text, str) for text in texts)
            assert kwargs["normalize_embeddings"] and kwargs["prompt"] == ""
            return np.tile(np.array([[.6, .8]], dtype=np.float32), (len(texts), 1))
    model = Model()
    config = {"preprocessing": {"max_seq_length": 256}, "device": "cpu", "batch_size": 32}
    first, second, receipt = embedding_cache(model, records(), config, {"manifest_sha256": "pinned"}, tmp_path, {})
    first2, _, receipt2 = embedding_cache(model, records(), config, {"manifest_sha256": "pinned"}, tmp_path, {})
    assert model.calls == 1 and np.array_equal(first, second) and np.array_equal(first, first2)
    assert receipt == receipt2 and receipt["token_lengths"]["profile_occurrences_truncated"] == 0
    config["preprocessing"]["max_seq_length"] = 128
    _, _, revised = embedding_cache(model, records(), config, {"manifest_sha256": "pinned"}, tmp_path, {})
    assert revised["key"] != receipt["key"] and model.calls == 2
    from pathlib import Path
    Path(receipt["array_path"]).write_bytes(b"corruption")
    config["preprocessing"]["max_seq_length"] = 256
    with pytest.raises(DataError, match="checksum"):
        embedding_cache(model, records(), config, {"manifest_sha256": "pinned"}, tmp_path, {})


def test_target_tiers_keep_undefined_as_null():
    result = target_results({"cohens_d": None, "fpr": .2}, {"coverage_adjusted_purity": .6},
                            {"required": {"cohens_d": [">=", .5], "fpr": ["<=", .35], "cluster_purity": [">=", .65]}})
    assert result["required"]["cohens_d"]["met"] is None
    assert result["required"]["fpr"]["met"] is True
    assert result["required"]["cluster_purity"]["met"] is False


def test_umap_refuses_to_silently_adjust_small_samples():
    with pytest.raises(DataError, match="Too few"):
        project_and_cluster([[1., 0.]] * 3, [[0., 1.]] * 3, {"umap": {"n_neighbors": 15}})


def test_baseline_checks_quality_before_loading_or_downloading(monkeypatch):
    from dating_compat import evaluate, quality
    def blocked(*args):
        raise DataError("Human review gate blocked")
    monkeypatch.setattr(quality, "require_quality_gate", blocked)
    monkeypatch.setattr(evaluate, "model_snapshot", lambda config: pytest.fail("Model loading must not happen"))
    with pytest.raises(DataError, match="Human review gate blocked"):
        evaluate.baseline()

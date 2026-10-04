import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dating_compat.data import (DataError, Pair, cross_audit, digest, load_bundle,
                               load_records, metadata_observed, normalize, pair_audit, write_immutable)
from dating_compat.split import assert_data_ready, make_split, profile_groups, validate_manifest


def row(index=0, label=0, **changes):
    return {"text_1": f"girl: Unique profile {index}", "text_2": f"boy: Partner {index}",
            "label": label, "category": "values", "subcategory": "career",
            "pair_type": "compatible", **changes}


def records(tmp_path, rows, name="dating_pairs.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return load_records(path)


@pytest.mark.parametrize("label", [True, False, "1", "0", 1.0, -1, 2, None])
def test_label_is_strict_binary_integer(label):
    with pytest.raises(ValidationError):
        Pair.model_validate(row(label=label))


@pytest.mark.parametrize("field", ["text_1", "text_2", "category", "subcategory", "pair_type"])
def test_required_text_is_nonblank(field):
    with pytest.raises(ValidationError):
        Pair.model_validate(row(**{field: " \t "}))


def test_raw_input_is_preserved_and_pair_type_is_not_label():
    pair = Pair.model_validate(row(label=0, text_1="  girl: Mixed CASE  "))
    assert pair.text_1 == "  girl: Mixed CASE  "
    assert pair.label == 0 and pair.pair_type == "compatible"


@pytest.mark.parametrize("bad", ["{", "", '{"label":0,"label":1}', '{"label":NaN}', '[]'])
def test_jsonl_error_has_filename_and_line(tmp_path, bad):
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(row()) + "\n" + bad + "\n")
    with pytest.raises(DataError, match="broken.jsonl:2:"):
        load_records(path)


def test_missing_field_or_extra_field_rejected():
    missing = row()
    del missing["subcategory"]
    with pytest.raises(ValidationError):
        Pair.model_validate(missing)
    with pytest.raises(ValidationError):
        Pair.model_validate(row(extra="hidden model input"))


def test_stable_ids_identify_duplicate_occurrences(tmp_path):
    first = records(tmp_path, [row(), row()])
    second = load_records(tmp_path / "dating_pairs.jsonl")
    assert [r.id for r in first] == [r.id for r in second]
    assert first[0].id != first[1].id


def test_normalized_reversed_contradictions_and_cross_overlap(tmp_path):
    a = row(text_1="Girl: Ａ  hobby", text_2="boy: Other hobby", label=1)
    b = row(text_1=" BOY: other hobby ", text_2="girl: a hobby", label=0)
    rows = records(tmp_path, [a, b])
    audit = pair_audit(rows)
    assert audit["normalized_unordered_duplicate_extras"] == 1
    assert len(audit["contradictory_groups"]) == 1
    evaluation = records(tmp_path, [b], "eval_pairs.jsonl")
    overlap = cross_audit(rows, evaluation)
    assert overlap["overlapping_pair_count"] == 1
    assert overlap["overlapping_pairs"][0]["labels"] == [0, 1]
    assert normalize(a["text_1"]) == normalize(b["text_2"])


def test_missing_metadata_blocks_bundle(tmp_path):
    records(tmp_path, [row()])
    with pytest.raises(DataError, match="dating_pairs_metadata.json"):
        load_bundle(tmp_path)


def test_metadata_discrepancies_are_reported_without_dropping_rows(tmp_path):
    for stem in ("dating_pairs", "eval_pairs"):
        rows = records(tmp_path, [row(1, 0), row(2, 1)], stem + ".jsonl")
        metadata = metadata_observed(rows)
        metadata["total_pairs"] = 3
        (tmp_path / (stem + "_metadata.json")).write_text(json.dumps(metadata))
    bundle = load_bundle(tmp_path)
    assert len(bundle["records"]["train"]) == 2
    assert bundle["metadata_comparison"]["train"]["differences"]["total_pairs"]["observed"] == 2
    assert "train: metadata mismatch" in bundle["blockers"]


def test_profile_connected_split_is_seeded_and_pair_safe(tmp_path):
    rows = [row(i, i % 2) for i in range(60)]
    rows += [row(60, 0, text_1=rows[0]["text_2"], text_2=rows[0]["text_1"]),
             row(61, 1, text_1=rows[1]["text_2"])]
    source = records(tmp_path, rows)
    a = make_split(source, {"source": "hash"}, [])
    b = make_split(source, {"source": "hash"}, [])
    assert a == b
    validate_manifest(a, source, {"source": "hash"})
    assert a["fit_validation_audit"]["shared_normalized_profiles"] == 0
    assert a["fit_validation_audit"]["overlapping_pair_count"] == 0
    for first, second in ((0, 60), (1, 61)):
        assert (source[first].id in a["fit_ids"]) == (source[second].id in a["fit_ids"])


def test_split_does_not_silently_abandon_grouping(tmp_path):
    source = records(tmp_path, [row(i, i % 2, text_1="girl: one shared profile") for i in range(40)])
    assert len(set(profile_groups(source))) == 1
    with pytest.raises(DataError, match="no ungrouped fallback"):
        make_split(source, {}, [])


def test_manifest_blocks_training_on_leakage_and_changed_sources(tmp_path):
    source = records(tmp_path, [row(i, i % 2) for i in range(40)])
    manifest = make_split(source, {"source": "hash"}, ["Train/eval leakage"])
    validate_manifest(manifest, source, {"source": "hash"}, require_unblocked=False)
    with pytest.raises(DataError, match="gate blocked"):
        validate_manifest(manifest, source, {"source": "hash"})
    with pytest.raises(DataError, match="source hashes"):
        validate_manifest(manifest, source, {"source": "changed"})
    manifest["seed"] = 7
    with pytest.raises(DataError, match="checksum mismatch"):
        validate_manifest(manifest, source, {"source": "hash"})


def test_even_rehashed_manifest_cannot_move_shared_profiles(tmp_path):
    rows = [row(i, i % 2) for i in range(40)] + [row(0, 0)]
    source = records(tmp_path, rows)
    manifest = make_split(source, {}, [])
    # Force the repeated profile into the other partition, then recalculate digest.
    current = "fit_ids" if source[0].id in manifest["fit_ids"] else "validation_ids"
    other = "validation_ids" if current == "fit_ids" else "fit_ids"
    manifest[current].remove(source[-1].id)
    manifest[other].append(source[-1].id)
    manifest["manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    with pytest.raises(DataError, match="leaks normalized"):
        validate_manifest(manifest, source, {})


def test_frozen_artifact_refuses_overwrite(tmp_path):
    path = tmp_path / "split.json"
    write_immutable(path, {"seed": 42})
    original = path.read_bytes()
    write_immutable(path, {"seed": 42})
    with pytest.raises(DataError, match="immutable artifact differs"):
        write_immutable(path, {"seed": 7})
    assert path.read_bytes() == original


def test_rehashed_ready_claim_cannot_erase_cross_file_leakage(tmp_path):
    source = records(tmp_path, [row(i, i % 2) for i in range(40)])
    evaluation = records(tmp_path, [row(0, 0)], "eval_pairs.jsonl")
    manifest = make_split(source, {}, [])
    bundle = {"records": {"train": source, "eval": evaluation}, "hashes": {}, "blockers": []}
    with pytest.raises(DataError, match="train/eval overlapping"):
        assert_data_ready(bundle, manifest)


def test_bad_acquisition_provenance_is_rejected(tmp_path):
    from dating_compat.data import verify_acquisition
    (tmp_path / "acquisition_manifest.json").write_text('{"revision":"different"}')
    with pytest.raises(DataError, match="provenance"):
        verify_acquisition(tmp_path, tmp_path)


def test_raw_reversed_duplicate_count(tmp_path):
    a = row()
    b = row(text_1=a["text_2"], text_2=a["text_1"])
    audit = pair_audit(records(tmp_path, [a, b]))
    assert audit["exact_ordered_duplicate_extras"] == 0
    assert audit["reversed_duplicate_extras"] == 1


def test_saved_manifest_survives_json_roundtrip(tmp_path):
    rows = [row(i, i % 2) for i in range(80)]
    rows += [row(i + 100, i % 2, text_1=rows[0]["text_1"]) for i in range(10)]
    rows.append(row(200, 0, text_1=rows[2]["text_1"]))
    source = records(tmp_path, rows)
    manifest = make_split(source, {}, [])
    path = tmp_path / "manifest.json"
    write_immutable(path, manifest)
    loaded = json.loads(path.read_text())
    validate_manifest(loaded, source, {})
    assert loaded == manifest


def test_model_columns_exclude_metadata_and_preserve_raw_text(tmp_path):
    from dating_compat.data import model_columns
    source = records(tmp_path, [row(text_1=" Girl: Original CASE ", label=0)])
    columns = model_columns(source)
    assert set(columns) == {"text_1", "text_2", "label"}
    assert columns["text_1"] == [" Girl: Original CASE "]
    assert columns["label"] == [0.0]

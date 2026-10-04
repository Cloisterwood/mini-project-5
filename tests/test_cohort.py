import json

import pytest

from dating_compat.cohort import QuarantinePolicy, derive_cohort
from dating_compat.data import DataError, digest, load_records, write_immutable
from dating_compat.split import assert_data_ready, make_split


def fixture(tmp_path):
    rows = [{"text_1": f"girl: Profile {i}", "text_2": f"boy: Partner {i}",
             "label": i % 2, "category": "values", "subcategory": "career", "pair_type": "compatible"}
            for i in range(60)]
    train = tmp_path / "dating_pairs.jsonl"
    evaluation = tmp_path / "eval_pairs.jsonl"
    train.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    reversed_pair = {**rows[0], "text_1": rows[0]["text_2"], "text_2": rows[0]["text_1"]}
    evaluation.write_text(json.dumps(reversed_pair) + "\n")
    records = load_records(train)
    original = {"records": {"train": records, "eval": load_records(evaluation)},
                "hashes": {"train": "original", "eval": "original"}, "blockers": []}
    policy = QuarantinePolicy(protocol_version="s2-quarantine-v1", approved_by="user", approved_on="2026-10-04",
        approval="Use only the eligible rows", reason="quarantine source overlap", source_hashes=original["hashes"],
        excluded_training_ids=[records[0].id])
    return original, policy


def split_for(bundle):
    split = make_split(bundle["records"]["train"], bundle["hashes"], [])
    split["cohort_manifest_sha256"] = bundle["cohort_manifest"]["manifest_sha256"]
    split["manifest_sha256"] = digest({k: v for k, v in split.items() if k != "manifest_sha256"})
    return split


def test_quarantine_preserves_originals_eval_and_raw_text(tmp_path):
    original, policy = fixture(tmp_path)
    before = (tmp_path / "dating_pairs.jsonl").read_bytes()
    derived, cohort = derive_cohort(original, policy)
    assert len(original["records"]["train"]) == 60
    assert len(derived["records"]["train"]) == 59
    assert derived["records"]["eval"] == original["records"]["eval"]
    assert derived["records"]["train"][0].pair.text_1 == original["records"]["train"][1].pair.text_1
    assert cohort["post_quarantine_cross_audit"]["overlapping_pair_count"] == 0
    assert (tmp_path / "dating_pairs.jsonl").read_bytes() == before
    split = split_for(derived)
    assert_data_ready(derived, split)
    assert policy.excluded_training_ids[0] not in split["fit_ids"] + split["validation_ids"]
    path = tmp_path / "cohort.json"
    write_immutable(path, cohort)
    assert json.loads(path.read_text()) == cohort
    assert derive_cohort(original, policy)[1] == cohort


@pytest.mark.parametrize("change", ["extra", "missing", "unknown", "source"])
def test_policy_cannot_silently_change_exclusions_or_sources(tmp_path, change):
    original, policy = fixture(tmp_path)
    payload = policy.model_dump()
    if change == "extra":
        payload["excluded_training_ids"].append(original["records"]["train"][1].id)
    elif change == "missing":
        payload["excluded_training_ids"] = [original["records"]["train"][2].id]
    elif change == "unknown":
        payload["excluded_training_ids"] = ["unknown"]
    else:
        payload["source_hashes"] = {"train": "modified"}
    with pytest.raises(DataError):
        derive_cohort(original, QuarantinePolicy.model_validate(payload))


def test_quarantine_cannot_bypass_metadata_blockers(tmp_path):
    original, policy = fixture(tmp_path)
    original["blockers"] = ["metadata mismatch"]
    with pytest.raises(DataError, match="structural/metadata gate blocked"):
        derive_cohort(original, policy)


def test_tampering_derived_eligibility_or_eval_is_rejected(tmp_path):
    original, policy = fixture(tmp_path)
    derived, _ = derive_cohort(original, policy)
    split = split_for(derived)
    derived["records"]["train"].append(original["records"]["train"][0])
    with pytest.raises(DataError, match="differs from approved"):
        assert_data_ready(derived, split)
    derived, _ = derive_cohort(original, policy)
    derived["records"]["eval"] = []
    with pytest.raises(DataError, match="Evaluation cohort changed"):
        assert_data_ready(derived, split)


def test_rehashed_cohort_claim_is_recomputed(tmp_path):
    original, policy = fixture(tmp_path)
    derived, cohort = derive_cohort(original, policy)
    split = split_for(derived)
    cohort["excluded_training_ids"] = []
    cohort["manifest_sha256"] = digest({k: v for k, v in cohort.items() if k != "manifest_sha256"})
    with pytest.raises(DataError, match="differs from approved"):
        assert_data_ready(derived, split)

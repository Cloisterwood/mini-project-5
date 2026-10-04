"""Verify the approved quarantine against immutable original sources."""

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from dating_compat.data import (DataError, cross_audit, digest, load_bundle,
                               pair_audit, parse_json)


class QuarantinePolicy(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    protocol_version: Literal["s2-quarantine-v1"]
    approved_by: Literal["user"]
    approved_on: str
    approval: str
    reason: str
    source_hashes: dict[str, str]
    excluded_training_ids: list[str]

    @field_validator("excluded_training_ids")
    @classmethod
    def unique_ids(cls, ids: list[str]) -> list[str]:
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Excluded IDs must be nonempty and unique")
        return ids


def load_policy(path: Path) -> QuarantinePolicy:
    try:
        return QuarantinePolicy.model_validate(parse_json(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise DataError(f"{path}: invalid quarantine policy: {exc}") from exc


def derive_cohort(original: dict, policy: QuarantinePolicy) -> tuple[dict, dict]:
    if original["hashes"] != policy.source_hashes:
        raise DataError("Quarantine policy source hashes differ from original data")
    if original["blockers"]:
        raise DataError(f"Original structural/metadata gate blocked: {original['blockers']}")
    for split, records in original["records"].items():
        if pair_audit(records)["contradictory_groups"]:
            raise DataError(f"{split}: unresolved contradictory source pairs")
    train, evaluation = original["records"]["train"], original["records"]["eval"]
    evaluation_keys = {record.key for record in evaluation}
    actual_exclusions = [record for record in train if record.key in evaluation_keys]
    approved = set(policy.excluded_training_ids)
    if approved != {record.id for record in actual_exclusions}:
        raise DataError("Approved exclusions do not exactly match source train/eval overlapping rows")
    eligible = [record for record in train if record.id not in approved]
    if not eligible:
        raise DataError("Quarantine leaves no eligible training rows")
    after = cross_audit(eligible, evaluation)
    if after["overlapping_pair_count"]:
        raise DataError("Quarantine failed to remove cross-file pair leakage")
    cohort = {
        "protocol_version": policy.protocol_version, "status": "ready_for_quality_review",
        "policy_sha256": digest(policy.model_dump()), "source_hashes": original["hashes"],
        "approval": {"by": policy.approved_by, "on": policy.approved_on, "text": policy.approval},
        "reason": policy.reason,
        "original_training_count": len(train), "eligible_training_count": len(eligible),
        "evaluation_count": len(evaluation), "evaluation_ids": [record.id for record in evaluation],
        "eligible_training_ids": [record.id for record in eligible],
        "excluded_training_ids": [record.id for record in actual_exclusions],
        "excluded_training_lines": [record.line for record in actual_exclusions],
        "eligible_label_counts": dict(sorted(Counter(str(record.pair.label) for record in eligible).items())),
        "post_quarantine_cross_audit": after,
        "limitations": "Shared individual profiles and surface templates remain; pair quarantine does not prove independent people or realistic compatibility",
    }
    cohort["manifest_sha256"] = digest(cohort)
    # Original metadata describes original data; keep that comparison intact.
    derived = {**original, "records": {"train": eligible, "eval": evaluation},
               "cohort_manifest": cohort, "original_bundle": original,
               "quarantine_policy": policy.model_dump()}
    return derived, cohort


def load_eligible_bundle(directory: Path = Path("data"),
                         policy_path: Path = Path("configs/quarantine-v1.json"),
                         reports: Path = Path("reports/quarantine-v1")) -> dict:
    """Quality/training entry point: rederive eligibility from original bytes.

    Requires frozen cohort and split manifests; recomputes leakage and exact ID
    coverage. This structural gate does not replace the S3 human quality gate.
    """
    from dating_compat.data import SOURCE_HASHES, verify_acquisition
    from dating_compat.split import assert_data_ready

    original = load_bundle(directory)
    if any(value != SOURCE_HASHES[name] for name, value in original["hashes"].items()):
        raise DataError("Original source differs from pinned acquisition")
    verify_acquisition(directory, reports.parent)
    derived, expected = derive_cohort(original, load_policy(policy_path))
    try:
        saved = parse_json((reports / "cohort_manifest.json").read_text(encoding="utf-8"))
        split = parse_json((reports / "split_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataError(f"Missing/invalid derived manifest: {exc}") from exc
    if saved != expected:
        raise DataError("Saved cohort differs from recomputed approved source derivation")
    assert_data_ready(derived, split)
    derived["split_manifest"] = split
    return derived

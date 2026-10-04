"""Descriptive data reports and explicit structural/leakage gate."""

import re
from collections import Counter
from pathlib import Path

import numpy as np

from dating_compat.data import (DataError, SOURCE_HASHES, cross_audit, digest, load_bundle,
                               normalize, pair_audit, verify_acquisition, write_immutable)
from dating_compat.split import assert_data_ready, make_split, validate_manifest


def statistics(values: list[int]) -> dict:
    return {"count": len(values), "min": min(values), "max": max(values), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def describe(records) -> dict:
    texts = [text for record in records for text in (record.pair.text_1, record.pair.text_2)]
    tokens = [re.findall(r"\b[\w']+\b", normalize(text)) for text in texts]
    vocabulary = Counter(token for profile in tokens for token in profile)
    prefixes = Counter()
    for record in records:
        pair = []
        for text in (record.pair.text_1, record.pair.text_2):
            match = re.match(r"^(boy|girl)\s*:", normalize(text))
            pair.append(match.group(1) if match else "other")
        prefixes["/".join(pair)] += 1
    counts = Counter(normalize(text) for text in texts)
    return {"rows": len(records), "distributions": {
        field: dict(sorted(Counter(str(getattr(record.pair, field)) for record in records).items()))
        for field in ("label", "category", "subcategory", "pair_type")},
        "gender_prefix_order": dict(sorted(prefixes.items())),
        "lengths": {"characters": statistics([len(text) for text in texts]),
                    "word_tokens": statistics([len(profile) for profile in tokens])},
        "vocabulary": {"definition": "NFKC/casefold Unicode word tokens including apostrophes; not model tokenizer",
                       "unique_tokens": len(vocabulary), "total_tokens": sum(vocabulary.values()),
                       "most_common_30": vocabulary.most_common(30)},
        "profiles": {"occurrences": len(texts), "unique_normalized": len(counts),
                     "repeated_occurrences": sum(count - 1 for count in counts.values()),
                     "unique_ratio": len(counts) / len(texts)},
        "pair_audit": pair_audit(records)}


def explore(directory: Path = Path("data"), reports: Path = Path("reports"),
            policy_path: Path | None = None) -> dict:
    # Acquire is separate: explore is always offline and requires all original files.
    try:
        bundle = load_bundle(directory)
    except DataError as exc:
        report = {"protocol_version": "s2-v1", "status": "blocked", "training_blockers": [str(exc)]}
        write_immutable(reports / "exploration.json", report)
        return report
    hashes = bundle["hashes"]
    blockers = list(bundle["blockers"])
    try:
        verify_acquisition(directory, reports.parent if policy_path else reports)
    except DataError as exc:
        blockers.append(str(exc))
    # This project is frozen to these observed original source hashes.
    for name, actual in hashes.items():
        if actual != SOURCE_HASHES[name]:
            blockers.append(f"{name}: hash differs from pinned original acquisition")
    original_overlap = cross_audit(bundle["records"]["train"], bundle["records"]["eval"])
    cohort = None
    if policy_path:
        from dating_compat.cohort import derive_cohort, load_policy
        bundle, cohort = derive_cohort(bundle, load_policy(policy_path))
    descriptions = {split: describe(records) for split, records in bundle["records"].items()}
    for split, description in descriptions.items():
        if description["pair_audit"]["contradictory_groups"]:
            blockers.append(f"{split}: contradictory normalized pairs")
    overlap = cross_audit(bundle["records"]["train"], bundle["records"]["eval"])
    if overlap["overlapping_pair_count"]:
        blockers.append(f"Train/eval leakage: {overlap['overlapping_pair_count']} normalized unordered pairs overlap")
    report = {"protocol_version": "s2-v1", "source_hashes": hashes,
        "status": "blocked" if blockers else "ready_for_quality_review", "training_blockers": blockers,
        "metadata_comparison": bundle["metadata_comparison"], "splits": descriptions,
        "train_eval_audit": overlap, "model_input_fields": ["text_1", "text_2"],
        "target_field": "label", "pair_type_used_as_target": False,
        "normalization": "NFKC + casefold + collapsed whitespace; audit/grouping only, raw model text preserved"}
    if cohort:
        report["protocol_version"] = "s2-quarantine-v1"
        report["cohort_manifest_sha256"] = cohort["manifest_sha256"]
        report["original_train_eval_audit"] = original_overlap
        report["metadata_comparison_scope"] = "Original immutable files; eligible training count is in cohort_manifest.json"
    try:
        manifest = make_split(bundle["records"]["train"], hashes, blockers)
        if cohort:
            manifest["protocol_version"] = "s2-quarantine-v1"
            manifest["cohort_manifest_sha256"] = cohort["manifest_sha256"]
            manifest["manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        validate_manifest(manifest, bundle["records"]["train"], hashes, require_unblocked=False)
        if not blockers:
            assert_data_ready(bundle, manifest)
        report["split_manifest_sha256"] = manifest["manifest_sha256"]
    except DataError as exc:
        report["training_blockers"].append(f"Split: {exc}")
        report["status"] = "blocked"
        manifest = None
    write_immutable(reports / "exploration.json", report)
    if manifest is not None:
        if cohort:
            write_immutable(reports / "cohort_manifest.json", cohort)
        write_immutable(reports / "split_manifest.json", manifest)
    return report

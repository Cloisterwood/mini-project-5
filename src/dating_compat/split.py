"""Training-only, profile-component grouped validation and frozen manifests."""

from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from dating_compat.data import DataError, Record, cross_audit, digest, normalize, pair_audit


def profile_groups(records: list[Record]) -> list[str]:
    parent = {}

    def root(text):
        parent.setdefault(text, text)
        while parent[text] != text:
            parent[text] = parent[parent[text]]
            text = parent[text]
        return text

    for record in records:
        a, b = root(normalize(record.pair.text_1)), root(normalize(record.pair.text_2))
        parent[max(a, b)] = min(a, b)
    return [digest(root(normalize(record.pair.text_1))) for record in records]


def make_split(records: list[Record], source_hashes: dict, blockers: list[str], seed: int = 42) -> dict:
    if len({record.id for record in records}) != len(records):
        raise DataError("Record IDs must be unique")
    groups = profile_groups(records)
    sizes = Counter(groups)
    if len(sizes) < 5:
        raise DataError("Fewer than five profile components: cannot form a grouped 80/20 split; no ungrouped fallback")
    # Retain grouping throughout. Coarsen strata only when distinct-group support
    # makes finer stratification impossible. Report every attempted scheme.
    schemes = [
        ("label/category/pair_type", lambda r: (r.pair.label, r.pair.category, r.pair.pair_type)),
        ("label/pair_type", lambda r: (r.pair.label, r.pair.pair_type)),
        ("label", lambda r: (r.pair.label,)),
    ]
    attempts = []
    for name, key in schemes:
        strata = [str(key(record)) for record in records]
        supports = {s: len({g for s2, g in zip(strata, groups) if s2 == s}) for s in set(strata)}
        rare = {s: count for s, count in sorted(supports.items()) if count < 5}
        attempts.append({"scheme": name, "distinct_group_support_below_5": rare})
        if not rare:
            break
    else:
        raise DataError(f"Label support cannot sustain five grouped folds: {attempts}; no grouping fallback")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = []
    fields = ("label", "category", "pair_type")
    population = {field: Counter(str(getattr(r.pair, field)) for r in records) for field in fields}
    for fold, (fit_indices, val_indices) in enumerate(splitter.split(np.zeros(len(records)), strata, groups)):
        fit = [records[i] for i in fit_indices]
        validation = [records[i] for i in val_indices]
        if {r.pair.label for r in fit} != {0, 1} or {r.pair.label for r in validation} != {0, 1}:
            continue
        deviations = {}
        for field in fields:
            counts = Counter(str(getattr(r.pair, field)) for r in validation)
            deviations[field] = sum(abs(counts[value] / len(validation) - total / len(records))
                                    for value, total in population[field].items())
        objective = 4 * abs(len(validation) / len(records) - .2) + sum(deviations.values())
        candidates.append((objective, fold, fit, validation, deviations))
    if not candidates:
        raise DataError("No profile-grouped fold contains both labels in fit and validation")
    _, fold, fit, validation, deviations = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    audit = cross_audit(fit, validation)
    if audit["overlapping_pair_count"] or audit["shared_normalized_profiles"]:
        raise DataError("Internal error: grouped split leaks normalized pairs/profiles")
    manifest = {
        "protocol_version": "s2-profile-components-v1", "seed": seed, "target_validation_fraction": .2,
        "source_hashes": source_hashes, "status": "blocked" if blockers else "ready_for_quality_review",
        "training_blockers": blockers, "fit_ids": [r.id for r in fit], "validation_ids": [r.id for r in validation],
        "counts": {"fit": len(fit), "validation": len(validation), "total": len(records)},
        "grouping": {"method": "Connected components of NFKC/casefold/whitespace-normalized training profiles",
            "component_count": len(sizes), "largest_component_pairs": max(sizes.values()),
            "component_size_histogram": {str(size): count for size, count in sorted(Counter(sizes.values()).items())},
            "exact_and_reversed_pairs_kept_together": True, "individual_profiles_kept_together": True,
            "template_identity_grouped": False,
            "limitation": "Surface template proxies may cross partitions; no claim of new unseen writing templates"},
        "stratification": {"attempts": attempts, "selected_scheme": name, "selected_fold": fold,
            "method": "Seeded StratifiedGroupKFold; choose fold minimizing train-only size and label/category/pair_type deviations",
            "distribution_l1_deviations": deviations},
        "fit_validation_audit": audit,
        "distributions": {partition: {field: dict(sorted(Counter(str(getattr(r.pair, field)) for r in subset).items()))
                           for field in fields} for partition, subset in (("fit", fit), ("validation", validation))},
    }
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def validate_manifest(manifest: dict, records: list[Record], source_hashes: dict, *, require_unblocked: bool = True) -> None:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if digest(payload) != manifest.get("manifest_sha256"):
        raise DataError("Split manifest checksum mismatch")
    if manifest["source_hashes"] != source_hashes:
        raise DataError("Split manifest source hashes differ from current data")
    fit, val = manifest["fit_ids"], manifest["validation_ids"]
    if len(set(fit)) != len(fit) or len(set(val)) != len(val):
        raise DataError("Repeated IDs in split manifest")
    if set(fit) & set(val) or set(fit) | set(val) != {r.id for r in records}:
        raise DataError("Split IDs overlap, are unknown, or do not cover the training file")
    fit_ids, val_ids = set(fit), set(val)
    audit = cross_audit([r for r in records if r.id in fit_ids], [r for r in records if r.id in val_ids])
    if audit["overlapping_pair_count"] or audit["shared_normalized_profiles"]:
        raise DataError("Split manifest leaks normalized pairs/profiles")
    if require_unblocked and (manifest["training_blockers"] or manifest["status"] != "ready_for_quality_review"):
        raise DataError(f"Training data gate blocked: {manifest['training_blockers']}")


def assert_data_ready(bundle: dict, manifest: dict) -> None:
    """Recompute data blockers; editing/re-hashing a report cannot erase leakage.

    This is the structural S2 gate. The separate S3 human quality gate is still
    required before training. Call with a fresh load_bundle, not cached claims.
    """
    if bundle["blockers"]:
        raise DataError(f"Training data gate blocked: {bundle['blockers']}")
    for split, records in bundle["records"].items():
        if pair_audit(records)["contradictory_groups"]:
            raise DataError(f"Training data gate blocked: {split} contradictory pairs")
    overlap = cross_audit(bundle["records"]["train"], bundle["records"]["eval"])
    if overlap["overlapping_pair_count"]:
        raise DataError(f"Training data gate blocked: {overlap['overlapping_pair_count']} train/eval overlapping pairs")
    validate_manifest(manifest, bundle["records"]["train"], bundle["hashes"])

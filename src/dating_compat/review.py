"""Seeded human review selection and strict, unfilled response sheets."""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from dating_compat.data import DataError, digest, file_hash, write_immutable

RATING_FIELDS = ("label_agreement", "naturalness", "coherence", "preference_realism", "hierarchy_consistency", "clarity")
IDENTITY_FIELDS = ("sample_number", "record_id", "text_1", "text_2", "label", "category", "subcategory", "pair_type")
COLUMNS = (*IDENTITY_FIELDS, *RATING_FIELDS, "notes", "reviewer", "human_reviewed")
RATINGS = {"acceptable", "ambiguous", "unacceptable"}


def select_review(bundle: dict, rubric: dict, rubric_hash: str) -> dict:
    fit = set(bundle["split_manifest"]["fit_ids"])
    rows = [record for record in bundle["records"]["train"] if record.id in fit]
    by_type = defaultdict(list)
    for record in sorted(rows, key=lambda r: r.id):
        by_type[record.pair.pair_type].append(record)
    rng = random.Random(rubric["human"]["seed"])
    selected, seen = [], set()
    minimum = rubric["human"]["minimum_per_pair_type"]
    for pair_type in sorted(by_type):
        candidates = by_type[pair_type].copy()
        rng.shuffle(candidates)
        count = 0
        for record in candidates:
            if record.key not in seen:
                selected.append(record)
                seen.add(record.key)
                count += 1
                if count == minimum:
                    break
        if count < minimum:
            raise DataError(f"Human sampling lacks {minimum} unique fit pairs for {pair_type}")
    if len(selected) > rubric["human"]["sample_size"]:
        raise DataError("Human per-type quotas exceed frozen sample size")
    remaining = [record for record in rows if record.key not in seen]
    rng.shuffle(remaining)
    for record in remaining:
        if len(selected) == rubric["human"]["sample_size"]:
            break
        if record.key not in seen:
            selected.append(record)
            seen.add(record.key)
    if len(selected) != rubric["human"]["sample_size"]:
        raise DataError("Not enough unique eligible fit pairs for frozen human sample")
    rng.shuffle(selected)
    manifest = {"version": "s3-review-v1", "seed": rubric["human"]["seed"],
        "sample_size": len(selected), "rubric_sha256": rubric_hash,
        "cohort_manifest_sha256": bundle["cohort_manifest"]["manifest_sha256"],
        "split_manifest_sha256": bundle["split_manifest"]["manifest_sha256"],
        "selected_ids": [record.id for record in selected],
        "pair_type_counts": dict(sorted(Counter(r.pair.pair_type for r in selected).items())),
        "label_counts": dict(sorted(Counter(str(r.pair.label) for r in selected).items())),
        "selection": "Seeded random selection without replacement within each fit pair_type (two unique pairs/type), then random remaining fit pairs; shuffled presentation; excludes evaluation/validation/quarantine IDs",
        "limitation": "Stratified diagnostic sample; small rare pair types overrepresented. Unweighted review fractions are not population estimates."}
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def expected_rows(bundle: dict, sample: dict) -> list[dict]:
    lookup = {record.id: record for record in bundle["records"]["train"]}
    return [{"sample_number": str(i), "record_id": identifier,
             **{key: str(value) for key, value in lookup[identifier].pair.model_dump().items()}}
            for i, identifier in enumerate(sample["selected_ids"], 1)]


def prepare_review(bundle: dict, rubric: dict, rubric_hash: str, reports: Path, response_path: Path) -> dict:
    sample = select_review(bundle, rubric, rubric_hash)
    write_immutable(reports / "human_review_sample.json", sample)
    if not response_path.exists():
        response_path.parent.mkdir(parents=True, exist_ok=True)
        with response_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in expected_rows(bundle, sample):
                writer.writerow(row)  # Human ratings/notes/identity stay blank.
    return sample


def read_review(bundle: dict, sample: dict, path: Path) -> dict:
    empty = {"complete": False, "rows": [], "rated_rows": 0, "errors": [], "sha256": None}
    if not path.exists():
        return {**empty, "errors": ["Human review response file is missing"]}
    result = {**empty, "sha256": file_hash(path)}
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(COLUMNS):
                raise DataError("Human CSV header differs from the frozen response schema")
            rows = list(reader)
        expected = expected_rows(bundle, sample)
        if len(rows) != len(expected):
            raise DataError("Human response row count differs from the selected sample")
        for row, reference in zip(rows, expected):
            if None in row or any(row.get(key) != reference[key] for key in IDENTITY_FIELDS):
                raise DataError("Human response sample IDs/order/text/metadata changed")
            for field in RATING_FIELDS:
                if row[field] and row[field] not in RATINGS:
                    raise DataError(f"Sample {row['sample_number']}: invalid {field} rating {row[field]!r}")
            if any(row[field] in {"ambiguous", "unacceptable"} for field in RATING_FIELDS) and not row["notes"].strip():
                raise DataError(f"Sample {row['sample_number']}: notes required for ambiguous/unacceptable ratings")
            if row["human_reviewed"] not in {"", "yes"}:
                raise DataError(f"Sample {row['sample_number']}: human_reviewed must be yes or blank")
        result["rows"] = rows
        result["rated_rows"] = sum(all(row[field] in RATINGS for field in RATING_FIELDS)
                                   and bool(row["reviewer"].strip()) and row["human_reviewed"] == "yes" for row in rows)
        result["complete"] = result["rated_rows"] == len(expected)
        if not result["complete"]:
            result["errors"].append("Human review pending: all six ratings, reviewer, and human_reviewed=yes required on every row")
    except (OSError, ValueError, csv.Error) as exc:
        result["errors"].append(str(exc))
    return result

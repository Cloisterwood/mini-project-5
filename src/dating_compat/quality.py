"""Frozen quality scoring and source-bound gate; no LLM human substitutes."""

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr
from scipy.stats.contingency import association

from dating_compat.cohort import load_eligible_bundle
from dating_compat.data import DataError, digest, file_hash, normalize, pair_audit, parse_json, template_key
from dating_compat.review import RATING_FIELDS, prepare_review, read_review, select_review


class QualityGateError(DataError):
    pass


def load_rubric(path: Path = Path("configs/quality-rubric.yaml"),
                freeze_path: Path = Path("reports/quality_rubric_manifest.json")) -> tuple[dict, str]:
    try:
        rubric = yaml.safe_load(path.read_text(encoding="utf-8"))
        frozen = parse_json(freeze_path.read_text(encoding="utf-8"))
        actual = file_hash(path)
        if actual != frozen["rubric_sha256"] or rubric["version"] != frozen["version"]:
            raise DataError("Frozen quality rubric hash/version differs; no threshold changes allowed")
        if len(rubric["dimensions"]) != 5 or any(len(parts) != 4 for parts in rubric["dimensions"].values()):
            raise DataError("Rubric must have five dimensions with four components each")
        return rubric, actual
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise DataError(f"Cannot verify frozen rubric: {exc}") from exc


def score_value(value: float | None, rule: dict) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        raise DataError("Quality metric must be finite or explicitly null")
    method = rule["method"]
    if method in {"acceptable_fraction", "human_fraction"}:
        score = 5 * value
    elif method == "lower_better":
        score = 5 * (rule["bad"] - value) / (rule["bad"] - rule["good"])
    elif method == "label_balance":
        score = 5 if .4 <= value <= .6 else 5 * (value / .4 if value < .4 else (1 - value) / .4)
    else:
        raise DataError(f"Unknown scoring method {method}")
    return float(np.clip(score, 0, 5))


def prefix_order(record) -> str:
    prefixes = []
    for text in (record.pair.text_1, record.pair.text_2):
        match = re.match(r"^(boy|girl)\s*:", normalize(text))
        prefixes.append(match.group(1) if match else "other")
    return "/".join(prefixes)


def phi(a: int, b: int, c: int, d: int) -> float | None:
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return (a * d - b * c) / denominator if denominator else None


def automated_metrics(bundle: dict, rubric: dict) -> dict:
    records = bundle["records"]["train"]
    n = len(records)
    labels = np.array([r.pair.label for r in records])
    texts = [text for r in records for text in (r.pair.text_1, r.pair.text_2)]
    audits = pair_audit(records)
    metrics = {}

    def add(name, value, numerator, denominator, evidence, reason=None):
        metrics[name] = {"raw_statistic": value, "numerator": numerator, "denominator": denominator,
                         "evidence": evidence, "undefined_reason": reason}

    add("schema_completeness", 1., n, n, "Every eligible record passes strict source schema; invalid sources are rejected")
    coherent = bundle["cohort_manifest"]["eligible_label_counts"] == dict(sorted(Counter(str(r.pair.label) for r in records).items()))
    add("label_metadata_consistency", float(coherent), n if coherent else 0, n, "Original metadata validated before derivation; eligible label counts checked against cohort, never pair_type")
    duplicate_extras = audits["normalized_unordered_duplicate_extras"]
    add("unordered_duplicates", duplicate_extras / n, duplicate_extras, n, audits["duplicate_groups"])
    contradictions = {identifier for group in audits["contradictory_groups"] for identifier in group["ids"]}
    add("contradictory_pairs", 1 - len(contradictions) / n, n - len(contradictions), n, audits["contradictory_groups"])
    positive = int(labels.sum())
    add("label_balance", positive / n, positive, n, {"positive": positive, "negative": n - positive})
    for name, field, required in (("category_coverage", "category", rubric["required_categories"]),
                                  ("pair_type_coverage", "pair_type", rubric["required_pair_types"])):
        counts = Counter(getattr(r.pair, field) for r in records)
        present = sum(value in counts for value in required)
        add(name, present / len(required), present, len(required), {"required": required, "counts": dict(sorted(counts.items()))})
    unique = len({normalize(text) for text in texts})
    add("unique_profiles", unique / len(texts), unique, len(texts), "Unique normalized profile ratio; occurrences retained")

    orders = sorted({prefix_order(r) for r in records})
    order_counts = {order: [sum(prefix_order(r) == order and r.pair.label == label for r in records) for label in (0, 1)] for order in orders}
    if len(orders) > 1 and len(set(labels)) == 2:
        gender_value = float(association(np.array(list(order_counts.values())), method="cramer", correction=False))
        reason = None
    else:
        gender_value, reason = None, "One prefix order or one label class; Cramers V undefined"
    add("gender_prefix_association", gender_value, None, n, {"columns": [0, 1], "counts": order_counts}, reason)
    categories = defaultdict(list)
    for record in records:
        categories[record.pair.category].append(record.pair.label)
    overall_positive = positive / n
    breakdown = {name: {"support": len(values), "positive": sum(values), "positive_fraction": sum(values) / len(values),
                       "absolute_deviation": abs(sum(values) / len(values) - overall_positive)} for name, values in sorted(categories.items())}
    skew = sum(value["support"] * value["absolute_deviation"] for value in breakdown.values()) / n
    add("category_label_skew", skew, None, n, breakdown)
    lengths = [len(r.pair.text_1) + len(r.pair.text_2) for r in records]
    if len(set(lengths)) > 1 and len(set(labels)) == 2:
        signed_r = float(pearsonr(lengths, labels).statistic)
        correlation, reason = abs(signed_r), None
    else:
        signed_r, correlation, reason = None, None, "Constant total length or label; correlation undefined"
    add("length_label_association", correlation, None, n, {"signed_pearson_r": signed_r, "positive": positive, "negative": n - positive, "length_unit": "sum of raw character lengths"}, reason)

    h = rubric["heuristics"]
    stopwords = set(h["vocabulary_stopwords"])
    pair_tokens = [set(re.findall(r"\b[\w']+\b", normalize(r.pair.text_1 + " " + r.pair.text_2))) - stopwords for r in records]
    tokens = defaultdict(lambda: [0, 0])
    for tokens_in_pair, label in zip(pair_tokens, labels):
        for token in tokens_in_pair:
            tokens[token][int(label)] += 1
    minimum = math.ceil(n * h["vocabulary_minimum_document_fraction"])
    associations = []
    for token, (negative_count, positive_count) in sorted(tokens.items()):
        support = negative_count + positive_count
        if support < minimum:
            continue
        value = phi(positive_count, negative_count, positive - positive_count, n - positive - negative_count)
        if value is not None:
            associations.append({"token": token, "support": support, "positive": positive_count, "negative": negative_count, "phi": value})
    shortcuts = [item for item in associations if abs(item["phi"]) >= h["vocabulary_phi_cutoff"]]
    shortcut_words = {item["token"] for item in shortcuts}
    covered = sum(bool(words & shortcut_words) for words in pair_tokens)
    add("vocabulary_shortcuts", covered / n, covered, n,
        {"minimum_pair_support": minimum, "phi_cutoff": h["vocabulary_phi_cutoff"], "shortcut_token_count": len(shortcuts),
         "shortcut_tokens": sorted(shortcuts, key=lambda item: (-abs(item["phi"]), item["token"])),
         "top_20_associations": sorted(associations, key=lambda item: (-abs(item["phi"]), item["token"]))[:20]})
    openings = Counter(template_key(text) for text in texts)
    largest = max(openings.values())
    add("template_repetition", largest / len(texts), largest, len(texts), {"definition": h["template_definition"], "top_10": openings.most_common(10)})
    invalid = []
    for record in records:
        for field in ("text_1", "text_2"):
            text = getattr(record.pair, field)
            reasons = []
            if not text.strip() or sum(c.isalpha() for c in text) < h["text_minimum_alpha_characters"]:
                reasons.append("blank or too few alphabetic characters")
            if "\ufffd" in text or any(unicodedata.category(c) == "Cc" and c not in "\t\n\r" for c in text):
                reasons.append("replacement/control character")
            if re.search(r"(.)\1{" + str(h["text_maximum_character_run"] - 1) + r",}", text):
                reasons.append("long repeated-character run")
            if any(marker in text for marker in h["truncation_markers"]):
                reasons.append("explicit truncation marker")
            if reasons:
                invalid.append({"record_id": record.id, "field": field, "reasons": reasons})
    add("text_integrity", 1 - len(invalid) / len(texts), len(texts) - len(invalid), len(texts), invalid)
    return metrics


def make_quality_report(bundle: dict, rubric: dict, rubric_hash: str, sample: dict, review: dict) -> dict:
    metrics = automated_metrics(bundle, rubric)
    dimensions = {}
    for dimension, parts in rubric["dimensions"].items():
        components = {}
        for name, rule in parts.items():
            if rule["method"] == "human_fraction":
                if review["complete"]:
                    counts = Counter(row[rule["field"]] for row in review["rows"])
                    acceptable = counts["acceptable"]
                    item = {"raw_statistic": acceptable / sample["sample_size"], "numerator": acceptable,
                            "denominator": sample["sample_size"], "evidence": {"field": rule["field"], "counts": dict(counts)}, "undefined_reason": None}
                else:
                    item = {"raw_statistic": None, "numerator": None, "denominator": sample["sample_size"],
                            "evidence": {"field": rule["field"], "rated_rows": review["rated_rows"]}, "undefined_reason": "Completed human review required"}
            else:
                item = metrics[name]
            components[name] = {**item, "score_out_of_5": score_value(item["raw_statistic"], rule),
                                "raw_range": [0, 1], "score_range": [0, 5], "scoring_rule": rule}
        scores = [item["score_out_of_5"] for item in components.values()]
        known = sum(value for value in scores if value is not None)
        total = sum(scores) if all(value is not None for value in scores) else None
        earlier = rubric["earlier_dimension_targets_out_of_100"][dimension]
        dimensions[dimension] = {"components": components, "score_out_of_20": total,
            "score_out_of_100": total * 5 if total is not None else None,
            "known_subscore_sum": known, "maximum_with_missing": known + 5 * scores.count(None),
            "earlier_target_out_of_100": earlier,
            "earlier_target_met": total * 5 >= earlier if total is not None and earlier is not None else None}
    totals = [item["score_out_of_20"] for item in dimensions.values()]
    overall = sum(totals) if all(value is not None for value in totals) else None
    failed = [name for name, item in dimensions.items() if item["maximum_with_missing"] < rubric["gate"]["each_dimension_minimum"]]
    if failed or (overall is not None and overall < rubric["gate"]["overall_minimum"]):
        status = "failed"
    elif not review["complete"] or any(value is None for value in totals):
        status = "pending_human_review" if not review["complete"] else "undefined_metric"
    else:
        status = "passed"
    report = {"version": "s3-quality-v1", "scope": "approved eligible training cohort; human sample from fit only",
        "source_hashes": bundle["hashes"], "cohort_manifest_sha256": bundle["cohort_manifest"]["manifest_sha256"],
        "split_manifest_sha256": bundle["split_manifest"]["manifest_sha256"], "rubric_sha256": rubric_hash,
        "sample_manifest_sha256": sample["manifest_sha256"], "human_response_sha256": review["sha256"],
        "eligible_rows": len(bundle["records"]["train"]), "dimensions": dimensions,
        "overall_score_out_of_100": overall,
        "gate": {"status": status, "passed": status == "passed", "thresholds": rubric["gate"],
                 "failed_dimensions": failed, "human_review_complete": review["complete"], "pending_or_invalid_review": review["errors"]},
        "human_review": {"sample_size": sample["sample_size"], "rated_rows": review["rated_rows"], "responses": review["rows"], "sampling_limitation": sample["limitation"]},
        "limitations": rubric["limitations"]}
    report["report_sha256"] = digest(report)
    return report


def quality_report(data_dir: Path = Path("data"), report_path: Path = Path("reports/quality_report.json"),
                   response_path: Path = Path("docs/human_review_responses.csv")) -> dict:
    bundle = load_eligible_bundle(data_dir)
    rubric, rubric_hash = load_rubric()
    sample = prepare_review(bundle, rubric, rubric_hash, Path("reports"), response_path)
    review = read_review(bundle, sample, response_path)
    report = make_quality_report(bundle, rubric, rubric_hash, sample, review)
    # This current-status report changes as human input arrives; frozen sample,
    # rubric, source/cohort/split artifacts remain immutable.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def require_quality_gate(data_dir: Path = Path("data"), report_path: Path = Path("reports/quality_report.json"),
                         response_path: Path = Path("docs/human_review_responses.csv"),
                         sample_path: Path = Path("reports/human_review_sample.json")) -> dict:
    """Training entry guard: reverify structural gate and recompute human quality."""
    bundle = load_eligible_bundle(data_dir)
    rubric, rubric_hash = load_rubric()
    try:
        sample = parse_json(sample_path.read_text(encoding="utf-8"))
        saved = parse_json(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QualityGateError(f"Missing/invalid quality evidence: {exc}") from exc
    expected_sample = select_review(bundle, rubric, rubric_hash)
    if sample != expected_sample:
        raise QualityGateError("Human sample differs from frozen eligible fit selection")
    report = make_quality_report(bundle, rubric, rubric_hash, sample, read_review(bundle, sample, response_path))
    # Compare the JSON representation: diagnostic tuples serialize as arrays.
    if digest(saved) != digest(report):
        raise QualityGateError("Quality report stale or modified; recompute with current source/rubric/review")
    if not report["gate"]["passed"]:
        raise QualityGateError(f"Quality gate {report['gate']['status']}: {report['gate']['failed_dimensions']}; {report['gate']['pending_or_invalid_review']}")
    return report

"""Signed separation and fixed-threshold classification, with explicit undefined cases."""

import numpy as np
from sklearn.metrics import roc_auc_score

from dating_compat.data import DataError


def validate(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or y.shape != s.shape or len(y) == 0:
        raise DataError("Labels/scores must be nonempty aligned one-dimensional arrays")
    if y.dtype.kind not in "iu" or not np.isin(y, [0, 1]).all():
        raise DataError("Labels must be binary integers")
    if not np.isfinite(s).all() or (np.abs(s) > 1 + 1e-6).any():
        raise DataError("Cosine scores must be finite and within [-1, 1]")
    return y, s


def pair_metrics(labels, scores, threshold: float = .5) -> dict:
    y, s = validate(labels, scores)
    if not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise DataError("Invalid cosine threshold")
    predicted = s >= threshold
    positive, negative = s[y == 1], s[y == 0]
    n1, n0 = len(positive), len(negative)
    tp, fp = int(((y == 1) & predicted).sum()), int(((y == 0) & predicted).sum())
    fn, tn = n1 - tp, n0 - fp
    undefined = {}

    def divide(name, numerator, denominator):
        if denominator == 0:
            undefined[name] = "Zero denominator"
            return None
        return float(numerator / denominator)

    mean1, mean0 = (float(positive.mean()) if n1 else None), (float(negative.mean()) if n0 else None)
    margin = mean1 - mean0 if n1 and n0 else None
    if margin is None:
        undefined["margin"] = "Both label classes required"
    d, pooled = None, None
    if n1 < 2 or n0 < 2:
        undefined["cohens_d"] = "At least two observations per class required"
    else:
        pooled = float(np.sqrt(((n1 - 1) * positive.var(ddof=1) + (n0 - 1) * negative.var(ddof=1)) / (n1 + n0 - 2)))
        if pooled == 0:
            undefined["cohens_d"] = "Zero pooled sample variance"
        else:
            d = float(margin / pooled)
    auc = float(roc_auc_score(y, s)) if n1 and n0 else None
    if auc is None:
        undefined["auc"] = "Both label classes required"
    return {"n": len(y), "positive": n1, "negative": n0, "threshold": threshold,
        "means": {"compatible": mean1, "incompatible": mean0}, "margin": margin,
        "pooled_sample_std": pooled, "cohens_d": d,
        "fpr": divide("fpr", fp, n0), "fpr_numerator": fp, "fpr_denominator": n0,
        "accuracy": float((tp + tn) / len(y)), "precision": divide("precision", tp, tp + fp),
        "recall": divide("recall", tp, n1), "f1": divide("f1", 2 * tp, 2 * tp + fp + fn),
        "auc": auc, "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_matrix_order": "rows true 0/1; columns predicted 0/1",
        "counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}, "undefined": undefined}


def breakdown(records, scores, threshold: float = .5) -> dict:
    if len(records) != len(scores):
        raise DataError("Record/score counts differ")
    return {field: {value: pair_metrics([r.pair.label for r in records if getattr(r.pair, field) == value],
                                      [float(s) for r, s in zip(records, scores) if getattr(r.pair, field) == value], threshold)
                   for value in sorted({getattr(r.pair, field) for r in records})}
            for field in ("category", "pair_type")}


def target_results(metrics: dict, clustering: dict, tiers: dict) -> dict:
    values = {**metrics, "cluster_purity": clustering["coverage_adjusted_purity"]}
    return {tier: {name: {"value": values[name], "comparison": comparison, "target": target,
                         "met": None if values[name] is None else
                         values[name] >= target if comparison == ">=" else values[name] <= target}
                   for name, (comparison, target) in targets.items()} for tier, targets in tiers.items()}

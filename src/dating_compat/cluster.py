"""Symmetric label-free pair features; UMAP/HDBSCAN, with noise receiving zero credit."""

import numpy as np

from dating_compat.data import DataError


def normalized_pairs(first, second) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] == 0 or a.shape[1] == 0:
        raise DataError("Profile embeddings must be aligned nonempty matrices")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise DataError("Nonfinite embeddings")
    na, nb = np.linalg.norm(a, axis=1, keepdims=True), np.linalg.norm(b, axis=1, keepdims=True)
    if (na == 0).any() or (nb == 0).any():
        raise DataError("Zero-length profile embeddings")
    return a / na, b / nb


def pair_scores(first, second) -> np.ndarray:
    a, b = normalized_pairs(first, second)
    return np.clip(np.sum(a * b, axis=1), -1, 1)


def pair_features(first, second) -> np.ndarray:
    a, b = normalized_pairs(first, second)
    return np.concatenate([np.abs(a - b), a * b], axis=1)


def cluster_purity(labels, clusters) -> dict:
    y, c = np.asarray(labels), np.asarray(clusters)
    if y.ndim != 1 or y.shape != c.shape or not len(y):
        raise DataError("Labels/clusters must be nonempty aligned arrays")
    if y.dtype.kind not in "iu" or not np.isin(y, [0, 1]).all() or c.dtype.kind not in "iu" or (c < -1).any():
        raise DataError("Binary labels and integer cluster IDs >= -1 required")
    groups = []
    for identifier in sorted(set(c) - {-1}):
        counts = np.bincount(y[c == identifier], minlength=2)
        groups.append({"cluster_id": int(identifier), "size": int(counts.sum()),
                       "label_counts": {"0": int(counts[0]), "1": int(counts[1])},
                       "majority_count": int(counts.max())})
    clustered = int((c >= 0).sum())
    correct = sum(group["majority_count"] for group in groups)
    return {"coverage_adjusted_purity": correct / len(y),
        "majority_numerator": correct, "all_pair_denominator": len(y),
        "clustered_only_purity": correct / clustered if clustered else None,
        "clustered_only_undefined_reason": None if clustered else "All points are noise",
        "clustered_count": clustered, "noise_count": len(y) - clustered,
        "noise_fraction": (len(y) - clustered) / len(y), "cluster_count": len(groups),
        "largest_cluster_size": max((g["size"] for g in groups), default=0),
        "observed_label_majority_baseline": int(np.bincount(y, minlength=2).max()) / len(y),
        "balanced_label_majority_baseline": .5, "clusters": groups}


def project_and_cluster(first, second, config: dict) -> tuple[np.ndarray, np.ndarray]:
    # No labels accepted by this function: coloring/purity happen afterward.
    import hdbscan
    import umap

    features = pair_features(first, second)
    if len(features) <= config["umap"]["n_neighbors"]:
        raise DataError("Too few pairs for frozen UMAP settings; no silent parameter changes")
    coordinates = umap.UMAP(**config["umap"]).fit_transform(features)
    if coordinates.shape != (len(features), 2) or not np.isfinite(coordinates).all():
        raise DataError("Invalid UMAP output")
    assignments = hdbscan.HDBSCAN(**config["hdbscan"]).fit_predict(coordinates)
    return coordinates, assignments

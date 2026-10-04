"""Paired percentile bootstrap with explicit profile-dependence diagnostics."""

from collections import Counter
import numpy as np

from dating_compat.data import DataError, digest
from dating_compat.metrics import pair_metrics, validate
from dating_compat.split import profile_groups


def paired_bootstrap(records, baseline, finetuned, config):
    labels = np.array([r.pair.label for r in records], dtype=int)
    _, base = validate(labels, baseline)
    _, fine = validate(labels, finetuned)
    groups = profile_groups(records)
    sizes = Counter(groups)
    largest = max(sizes.values()) / len(records)
    grouped = len(sizes) >= config["component_minimum"] and largest <= config["largest_component_maximum_fraction"]
    units = [np.flatnonzero(np.array(groups) == g) for g in sorted(sizes)] if grouped else [np.array([i]) for i in range(len(records))]
    draws = config["bootstrap_resamples"]
    if draws < 1:
        raise DataError("Positive bootstrap resample count required")
    rng = np.random.default_rng(config["bootstrap_seed"])
    keys = config["bootstrap_metrics"]
    samples = {k: {name: [] for name in ("baseline", "finetuned", "delta")} for k in keys}
    for _ in range(draws):
        indices = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
        before = pair_metrics(labels[indices], base[indices])
        after = pair_metrics(labels[indices], fine[indices])
        for key in keys:
            for name, value in (("baseline", before[key]), ("finetuned", after[key]), ("delta", None if before[key] is None or after[key] is None else after[key] - before[key])):
                if value is not None and np.isfinite(value):
                    samples[key][name].append(value)
    alpha = (1 - config["confidence_level"]) / 2
    intervals = {k: {name: {"interval": [float(x) for x in np.quantile(values, [alpha, 1-alpha])] if values else None,
                            "valid_resamples": len(values), "undefined_resamples": draws-len(values)}
                     for name, values in names.items()} for k, names in samples.items()}
    return {"method": "paired_profile_component_percentile" if grouped else "paired_pair_percentile_conditional_approximate",
        "resamples": draws, "seed": config["bootstrap_seed"], "confidence_level": config["confidence_level"],
        "profile_components": {"count": len(sizes), "largest_pairs": max(sizes.values()), "largest_fraction": largest,
            "size_histogram": {str(n): c for n,c in sorted(Counter(sizes.values()).items())}, "ordered_groups_sha256": digest(groups)},
        "unit_count": len(units), "intervals": intervals,
        "assumptions": ["Same sampled indices for baseline and finetuned; delta=finetuned-baseline", "Components treated as exchangeable independent units when eligible; shared templates can still violate independence",
            "Pair fallback conditions on this synthetic sample and does not correct shared-profile dependence" if not grouped else "Component bootstrap preserves exact normalized profile dependence within components",
            "No training-seed or run-selection uncertainty included", "Purity not bootstrapped; transforms/clusters not refitted per resample"]}

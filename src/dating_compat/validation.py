"""Immutable validation-only comparison for a completed training experiment."""

import csv
import io
import time
from pathlib import Path

import numpy as np

from dating_compat.cluster import cluster_purity, pair_scores, project_and_cluster
from dating_compat.data import DataError, digest, file_hash, load_records, parse_json, write_immutable
from dating_compat.evaluate import checked_predictions, embedding_cache, write_bytes_immutable
from dating_compat.metrics import breakdown, pair_metrics, target_results
from dating_compat.train import _model, checkpoint_receipt, evaluation_read_guard, state_hash
from dating_compat.training_data import load_training_inputs

METRICS = ("margin", "cohens_d", "fpr", "accuracy", "precision", "recall", "f1", "auc")


def signed_deltas(before: dict, after: dict) -> dict:
    return {key: {"before": before[key], "after": after[key],
                  "delta": None if before[key] is None or after[key] is None else after[key] - before[key]}
            for key in METRICS}


def aligned_baseline(path: Path, expected_ids: list[str], seal: dict) -> tuple[dict, np.ndarray]:
    csv_path = Path("reports/baseline/baseline_validation_predictions.csv")
    for artifact in (path, csv_path):
        if seal["artifacts"].get(str(artifact)) != file_hash(artifact):
            raise DataError("Validation baseline differs from sealed artifact")
    report = parse_json(path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if report["split"] != "validation" or report["ordered_record_ids_sha256"] != digest(expected_ids) or [r["record_id"] for r in rows] != expected_ids:
        raise DataError("Baseline validation IDs/order do not match the frozen validation set")
    scores = np.array([float(r["cosine"]) for r in rows])
    if pair_metrics([int(r["label"]) for r in rows], scores) != report["metrics"]:
        raise DataError("Baseline validation metrics differ from saved predictions")
    return report, scores


def validation_plots(records, scores, base_scores, coordinates, clusters, history, run_id, output) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    plt.rcParams.update({"font.size": 11})
    labels = np.array([r.pair.label for r in records])
    colors = ("#0072B2", "#D55E00")
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    for ax, values, title in zip(axes, (base_scores, scores), ("Base MiniLM", run_id)):
        for label in (0, 1):
            ax.hist(values[labels == label], bins=np.linspace(-1, 1, 41), alpha=.6, color=colors[label],
                    label=f"{'Compatible' if label else 'Incompatible'} (n={(labels == label).sum()})")
        ax.axvline(.5, color="black", linestyle="--", label="Threshold .5")
        ax.set(title=f"{title} — validation (n={len(records)})", xlabel="Cosine similarity", ylabel="Pair count")
        ax.legend(); ax.grid(alpha=.2)
    figures.append(("similarity", fig))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in (0, 1):
        mask = labels == label
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=9, alpha=.6, color=colors[label], label="Compatible" if label else "Incompatible")
    noise = clusters == -1
    ax.scatter(coordinates[noise, 0], coordinates[noise, 1], facecolors="none", edgecolors="gray", s=14, label=f"Noise (n={noise.sum()})")
    ax.set(title=f"{run_id} label-free pair UMAP — validation (n={len(records)})", xlabel="UMAP 1 (arbitrary axis)", ylabel="UMAP 2 (arbitrary axis)")
    ax.legend(); ax.grid(alpha=.2)
    figures.append(("umap", fig))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for values, name, color in ((base_scores, "Baseline", colors[0]), (scores, run_id, colors[1])):
        fpr, tpr, _ = roc_curve(labels, values)
        ax.plot(fpr, tpr, color=color, label=f"{name} AUC={pair_metrics(labels, values)['auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(title="Validation ROC comparison", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(); ax.grid(alpha=.2)
    figures.append(("roc", fig))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    train_logs = [r for r in history if "loss" in r]
    val_logs = [r for r in history if "eval_loss" in r]
    ax.plot([r["epoch"] for r in train_logs], [r["loss"] for r in train_logs], color=colors[0], alpha=.7, label="Fit loss (10-step windows)")
    ax.plot([r["epoch"] for r in val_logs], [r["eval_loss"] for r in val_logs], color=colors[1], marker="o", label="Validation loss (full set)")
    ax.set(title=f"{run_id} CosineSimilarityLoss", xlabel="Epoch", ylabel="Mean squared cosine error")
    ax.legend(); ax.grid(alpha=.2)
    figures.append(("loss", fig))
    paths = []
    for name, fig in figures:
        fig.tight_layout(); buffer = io.BytesIO(); fig.savefig(buffer, format="png", dpi=150); plt.close(fig)
        path = output / f"validation_{name}.png"
        write_bytes_immutable(path, buffer.getvalue()); paths.append(path)
    return paths


def validate_run(run_id: str) -> dict:
    import re
    if not re.fullmatch(r"run-00[123]", run_id):
        raise DataError("Run ID must be one of the three planned experiments")
    with evaluation_read_guard():
        inputs, certificate = load_training_inputs()
        output = Path("reports/runs") / run_id
        manifest = parse_json((output / "run_manifest.json").read_text(encoding="utf-8"))
        result = parse_json((output / "training_result.json").read_text(encoding="utf-8"))
        ids = [r["record_id"] for r in inputs["validation"]]
        if result["status"] != "completed" or manifest["validation_ids"] != ids or manifest["preflight_manifest_sha256"] != certificate["manifest_sha256"]:
            raise DataError("Completed training provenance differs from certified validation inputs")
        lookup = {r.id: r for r in load_records(Path("data/dating_pairs.jsonl"))}
        records = [lookup[i] for i in ids]
        if any(row["text_1"] != r.pair.text_1 or row["text_2"] != r.pair.text_2 or row["label"] != r.pair.label
               for row, r in zip(inputs["validation"], records)):
            raise DataError("Validation raw text/labels differ from original training source")
        seal = parse_json(Path("reports/baseline_manifest.json").read_text(encoding="utf-8"))
        base, base_scores = aligned_baseline(Path("reports/validation_baseline_metrics.json"), ids, seal)
        selected = Path(result["selected_path"])
        model_files = {str(p.relative_to(selected)): file_hash(p) for p in sorted(selected.rglob("*")) if p.is_file()}
        binding = {"training_manifest_sha256": file_hash(output / "run_manifest.json"),
            "training_result_sha256": file_hash(output / "training_result.json"),
            "preflight_manifest_sha256": certificate["manifest_sha256"], "selected_files": model_files,
            "validation_ids_sha256": digest(ids), "baseline_validation_sha256": file_hash(Path("reports/validation_baseline_metrics.json"))}
        artifact_manifest = output / "validation_manifest.json"
        if artifact_manifest.exists():
            saved = parse_json(artifact_manifest.read_text(encoding="utf-8"))
            if saved["binding"] != binding or digest({k: v for k, v in saved.items() if k != "manifest_sha256"}) != saved["manifest_sha256"]:
                raise DataError("Saved validation provenance differs")
            for name, sha in saved["artifacts"].items():
                if file_hash(Path(name)) != sha:
                    raise DataError("Saved validation artifact changed")
            return parse_json((output / "validation_metrics.json").read_text(encoding="utf-8"))
        started = time.monotonic()
        model = _model(manifest["config"], manifest["device"], selected)
        if state_hash(model) != result["selected_weights_sha256"]:
            raise DataError("Selected model weights differ from completed Trainer result")
        chosen = Path(result["best_model_checkpoint"])
        receipt = checkpoint_receipt(chosen, int(chosen.name.split("-")[-1]))
        write_immutable(output / "selected_checkpoint_manifest.json", {"selected_path": str(selected), "selected_files": model_files,
            "selected_weights_sha256": result["selected_weights_sha256"], "trainer_checkpoint": receipt})
        config = base["config"]
        model_binding = {"manifest_sha256": digest({"files": model_files, "state_sha256": result["selected_weights_sha256"]})}
        first, second, cache = embedding_cache(model, records, config, model_binding,
                                              Path(".cache/embeddings"), base["execution"]["software"])
        scores = pair_scores(first, second)
        coordinates, clusters = project_and_cluster(first, second, config)
        predictions = checked_predictions(records, scores, coordinates, clusters)
        metrics = pair_metrics([r.pair.label for r in records], scores)
        clustering = cluster_purity([r.pair.label for r in records], clusters)
        targets = target_results(metrics, clustering, config["target_tiers"])
        report = {"version": "s6-validation-v1", "run_id": run_id, "split": "validation", "binding": binding,
            "selected_checkpoint": result["best_model_checkpoint"], "selected_weights_sha256": result["selected_weights_sha256"],
            "preprocessing": config["preprocessing"], "umap": config["umap"], "hdbscan": config["hdbscan"],
            "embedding_cache": cache, "metrics": metrics, "clustering": clustering, "breakdowns": breakdown(records, scores),
            "targets": targets, "classification_deltas": signed_deltas(base["metrics"], metrics),
            "purity_comparison": {"before": base["clustering"]["coverage_adjusted_purity"],
                "after": clustering["coverage_adjusted_purity"], "delta": clustering["coverage_adjusted_purity"] - base["clustering"]["coverage_adjusted_purity"]},
            "all_final_checklist_targets_met": all(item["met"] is True for item in targets["final_checklist"].values()),
            "selection_status": "Candidate retained pending completion of all three planned runs; no final selection",
            "limitations": ["Validation used for checkpoint selection; not final held-out performance", "Same profile-grouped validation IDs and threshold as baseline",
                "Separately fitted label-free UMAP maps have unaligned axes", "No significance or uncertainty claim before paired analysis in S7"]}
        report_path, csv_path = output / "validation_metrics.json", output / "validation_predictions.csv"
        write_immutable(report_path, report)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(predictions[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(predictions)
        write_bytes_immutable(csv_path, buffer.getvalue().encode("utf-8"))
        write_immutable(output / "loss_history.json", {"history": result["history"]})
        write_immutable(output / "config.json", manifest["config"])
        paths = [report_path, csv_path, output / "selected_checkpoint_manifest.json", output / "loss_history.json", output / "config.json",
                 *validation_plots(records, scores, base_scores, coordinates, clusters, result["history"], run_id, output)]
        final = {"binding": binding, "artifacts": {str(p): file_hash(p) for p in paths},
                 "validation_elapsed_seconds": time.monotonic() - started,
                 "validation_code_sha256": file_hash(Path(__file__))}
        final["manifest_sha256"] = digest(final)
        write_immutable(artifact_manifest, final)
        return report

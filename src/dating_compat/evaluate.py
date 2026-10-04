"""Pinned, gated baseline with verified embedding caches and sealed split artifacts."""

import csv
import io
import json
import platform
import random
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from dating_compat.cluster import cluster_purity, pair_scores, project_and_cluster
from dating_compat.data import DataError, digest, file_hash, parse_json, write_immutable
from dating_compat.metrics import breakdown, pair_metrics, target_results

CONFIG = Path("configs/baseline.json")


def frozen_config(path: Path = CONFIG, manifest: Path = Path("reports/baseline_config_manifest.json")) -> dict:
    config = parse_json(path.read_text(encoding="utf-8"))
    if config["preprocessing"]["threshold"] != .5 or len(config["revision"]) != 40:
        raise DataError("Baseline requires pinned model revision and threshold .5")
    write_immutable(manifest, {"version": "s4-config-freeze-v1", "config_sha256": file_hash(path),
                               "config": config, "frozen_before_model_outcomes": True})
    return config


def split_records(bundle: dict, split: str) -> list:
    if split == "eval":
        return bundle["records"]["eval"]
    if split not in {"fit", "validation"}:
        raise DataError("Unknown baseline split")
    identifiers = bundle["split_manifest"][f"{split}_ids"]
    lookup = {r.id: r for r in bundle["records"]["train"]}
    if len(set(identifiers)) != len(identifiers) or any(i not in lookup for i in identifiers):
        raise DataError("Split IDs missing or repeated")
    return [lookup[i] for i in identifiers]


def checked_predictions(records: list, scores, coordinates, clusters) -> list[dict]:
    n = len(records)
    if len({r.id for r in records}) != n or np.asarray(scores).shape != (n,) or np.asarray(coordinates).shape != (n, 2) or np.asarray(clusters).shape != (n,):
        raise DataError("Prediction IDs/arrays are not aligned and unique")
    pair_metrics([r.pair.label for r in records], scores)
    if not np.isfinite(coordinates).all():
        raise DataError("Nonfinite prediction coordinates")
    cluster_purity([r.pair.label for r in records], clusters)
    return [{"record_id": r.id, "source_line": r.line, "label": r.pair.label,
             "cosine": float(s), "prediction": int(s >= .5), "category": r.pair.category,
             "pair_type": r.pair.pair_type, "umap_x": float(xy[0]), "umap_y": float(xy[1]),
             "cluster_id": int(c)} for r, s, xy, c in zip(records, scores, coordinates, clusters)]


def write_bytes_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise DataError(f"Refusing to overwrite frozen artifact: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(content)


def verify_seal(path: Path, expected_inputs: dict) -> dict:
    seal = parse_json(path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in seal.items() if k != "manifest_sha256"}
    if digest(payload) != seal.get("manifest_sha256") or seal.get("inputs") != expected_inputs:
        raise DataError("Baseline seal/config/data/quality provenance differs")
    for name, expected in seal["artifacts"].items():
        if file_hash(Path(name)) != expected:
            raise DataError(f"Sealed baseline artifact changed: {name}")
    return seal


def model_snapshot(config: dict) -> tuple[Path, dict]:
    from huggingface_hub import snapshot_download
    directory = Path("models/base-minilm") / config["revision"]
    allowed = ["config.json", "config_sentence_transformers.json", "sentence_bert_config.json",
               "modules.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt", "1_Pooling/config.json",
               "2_Normalize/config.json", "README.md"]
    if not (directory / "model.safetensors").exists():
        snapshot_download(config["model_id"], revision=config["revision"], local_dir=directory,
                          allow_patterns=allowed, token=False, max_workers=4)
    files = {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and ".cache" not in p.parts}
    if "model.safetensors" not in files or "modules.json" not in files:
        raise DataError("Incomplete pinned model snapshot")
    manifest = {"model_id": config["model_id"], "revision": config["revision"], "files": files}
    manifest["manifest_sha256"] = digest(manifest)
    write_immutable(Path("reports/base_model_manifest.json"), manifest)
    return directory, manifest


def embedding_cache(model, records: list, config: dict, model_manifest: dict, cache_dir: Path,
                    software: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    texts = sorted({text for r in records for text in (r.pair.text_1, r.pair.text_2)})
    binding = {"ordered_record_ids": [r.id for r in records], "raw_text_sha256": digest(texts),
               "model_manifest_sha256": model_manifest["manifest_sha256"],
               "preprocessing": config["preprocessing"], "device": config["device"],
               "batch_size": config["batch_size"], "software": software}
    key = digest(binding)
    arrays, metadata = cache_dir / f"{key}.npz", cache_dir / f"{key}.json"
    if arrays.exists() or metadata.exists():
        if not arrays.exists() or not metadata.exists():
            raise DataError("Incomplete embedding cache; preserve and investigate before reuse")
        saved = parse_json(metadata.read_text(encoding="utf-8"))
        if saved["binding"] != binding or saved["array_sha256"] != file_hash(arrays):
            raise DataError("Embedding cache provenance/checksum mismatch")
        with np.load(arrays, allow_pickle=False) as data:
            embeddings = data["embeddings"]
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        lengths = [len(ids) for ids in model.tokenizer(texts, truncation=False)["input_ids"]]
        length_by_text = dict(zip(texts, lengths))
        embeddings = model.encode(texts, batch_size=config["batch_size"], prompt="",
                                  show_progress_bar=True, precision="float32", normalize_embeddings=True,
                                  convert_to_numpy=True)
        if embeddings.dtype != np.float32:
            raise DataError("Baseline embeddings must be float32")
        with arrays.open("xb") as stream:
            np.savez_compressed(stream, embeddings=embeddings)
        saved = {"binding": binding, "array_sha256": file_hash(arrays),
                 "unique_profiles": len(texts), "embedding_dimension": embeddings.shape[1],
                 "token_lengths": {"max": max(lengths), "unique_profiles_truncated": sum(n > model.max_seq_length for n in lengths),
                                  "profile_occurrences_truncated": sum(length_by_text[t] > model.max_seq_length
                                      for r in records for t in (r.pair.text_1, r.pair.text_2))}}
        write_immutable(metadata, saved)
    if embeddings.shape != (len(texts), saved["embedding_dimension"]) or not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5):
        raise DataError("Invalid cached embedding shape/norm/values")
    lookup = {text: i for i, text in enumerate(texts)}
    first = embeddings[[lookup[r.pair.text_1] for r in records]]
    second = embeddings[[lookup[r.pair.text_2] for r in records]]
    return first, second, {"key": key, "array_path": str(arrays), "metadata_path": str(metadata),
                           "array_sha256": saved["array_sha256"], "token_lengths": saved["token_lengths"],
                           "embedding_dimension": saved["embedding_dimension"], "unique_profiles": saved["unique_profiles"]}


def baseline_plots(records, scores, coordinates, clusters, split: str, directory: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    plt.rcParams.update({"font.size": 11})
    labels = np.array([r.pair.label for r in records])
    colors = ("#0072B2", "#D55E00")
    names = ("Incompatible", "Compatible")
    figures = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(-1, 1, 41)
    for label in (0, 1):
        ax.hist(scores[labels == label], bins=bins, alpha=.6, color=colors[label], label=f"{names[label]} (n={(labels == label).sum()})")
    ax.axvline(.5, color="black", linestyle="--", label="Threshold 0.5")
    ax.set(xlabel="Cosine similarity", ylabel="Pair count", title=f"Base MiniLM similarity — {split} (n={len(records)})")
    ax.legend(); ax.grid(alpha=.2)
    figures.append(("similarity", fig))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in (0, 1):
        mask = labels == label
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=8, alpha=.5, color=colors[label], label=names[label])
    noise = clusters == -1
    ax.scatter(coordinates[noise, 0], coordinates[noise, 1], s=12, facecolors="none", edgecolors="gray", linewidths=.4, label=f"Noise (n={noise.sum()})")
    ax.set(xlabel="UMAP 1 (arbitrary axis)", ylabel="UMAP 2 (arbitrary axis)", title=f"Label-free pair UMAP — {split} (n={len(records)})")
    ax.legend(); ax.grid(alpha=.2)
    figures.append(("umap", fig))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fpr, tpr, _ = roc_curve(labels, scores)
    ax.plot(fpr, tpr, color=colors[0], label=f"Baseline AUC={pair_metrics(labels, scores)['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"Base MiniLM ROC — {split}")
    ax.legend(); ax.grid(alpha=.2)
    figures.append(("roc", fig))
    paths = []
    for name, fig in figures:
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150)
        plt.close(fig)
        path = directory / f"baseline_{split}_{name}.png"
        write_bytes_immutable(path, buffer.getvalue())
        paths.append(path)
    return paths


def baseline(data_dir: Path = Path("data")) -> dict:
    from dating_compat.cohort import load_eligible_bundle
    from dating_compat.quality import require_quality_gate

    quality = require_quality_gate(data_dir)
    bundle = load_eligible_bundle(data_dir)
    config = frozen_config()
    inputs = {"config_sha256": file_hash(CONFIG), "source_hashes": bundle["hashes"],
              "cohort_manifest_sha256": bundle["cohort_manifest"]["manifest_sha256"],
              "split_manifest_sha256": bundle["split_manifest"]["manifest_sha256"],
              "quality_report_sha256": quality["report_sha256"]}
    seal_path = Path("reports/baseline_manifest.json")
    if seal_path.exists():
        seal = verify_seal(seal_path, inputs)
        print("BASELINE SEALED ARTIFACTS VERIFIED; no inference or evaluation repeated", flush=True)
        return seal
    directory, model_manifest = model_snapshot(config)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    model = SentenceTransformer(str(directory), device=config["device"], local_files_only=True, trust_remote_code=False).float()
    model.max_seq_length = config["preprocessing"]["max_seq_length"]
    software = {name: version(name) for name in ("torch", "sentence-transformers", "transformers", "numpy", "scikit-learn", "umap-learn", "hdbscan", "matplotlib")}
    execution = {"python": platform.python_version(), "platform": platform.platform(), "software": software,
                 "device": str(model.device), "max_seq_length": model.max_seq_length,
                 "code_commit_at_execution": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 "implementation_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in ("metrics.py", "cluster.py", "evaluate.py")}}
    artifacts = {"reports/baseline_config_manifest.json": file_hash(Path("reports/baseline_config_manifest.json")),
                 "reports/base_model_manifest.json": file_hash(Path("reports/base_model_manifest.json"))}
    durations = {}
    for split in ("fit", "validation", "eval"):
        started = time.monotonic()
        records = split_records(bundle, split)
        print(f"BASELINE {split}: {len(records)} pairs; encoding raw profiles", flush=True)
        first, second, cache = embedding_cache(model, records, config, model_manifest, Path(".cache/embeddings"), software)
        scores = pair_scores(first, second)
        print(f"BASELINE {split}: label-free UMAP/HDBSCAN", flush=True)
        coordinates, clusters = project_and_cluster(first, second, config)
        predictions = checked_predictions(records, scores, coordinates, clusters)
        metrics = pair_metrics([r.pair.label for r in records], scores)
        clustering = cluster_purity([r.pair.label for r in records], clusters)
        report = {"version": "s4-baseline-v1", "split": split, "inputs": inputs, "config": config,
                  "execution": execution, "model_manifest_sha256": model_manifest["manifest_sha256"],
                  "ordered_record_ids_sha256": digest([r.id for r in records]), "embedding_cache": cache,
                  "metrics": metrics, "clustering": clustering, "breakdowns": breakdown(records, scores),
                  "targets": target_results(metrics, clustering, config["target_tiers"]),
                  "direct_feature_sensitivity": {"status": "not_run", "reason": "Optional analysis excluded before outcomes; clustered-only/noise/majority diagnostics reported"},
                  "limitations": ["Synthetic compatibility labels; cosine is not a probability", bundle["cohort_manifest"]["limitations"],
                                  "Separate unsupervised map fitted per split without labels; axes not aligned", "Evaluation baseline sealed; never use it for model/threshold/experiment selection"]}
        report_path = Path("reports") / ("baseline_metrics.json" if split == "eval" else f"{split}_baseline_metrics.json")
        write_immutable(report_path, report)
        csv_path = Path("reports/baseline") / f"baseline_{split}_predictions.csv"
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(predictions[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(predictions)
        write_bytes_immutable(csv_path, buffer.getvalue().encode("utf-8"))
        paths = [report_path, csv_path, *baseline_plots(records, scores, coordinates, clusters, split, Path("reports/baseline"))]
        artifacts.update({str(path): file_hash(path) for path in paths})
        durations[split] = time.monotonic() - started
        print(f"BASELINE {split} SAVED; {durations[split]:.1f}s", flush=True)
    seal = {"version": "s4-seal-v1", "inputs": inputs, "artifacts": artifacts,
            "execution": execution, "split_elapsed_seconds": durations,
            "evaluation_policy": "Frozen assignment-required baseline; no selection/tuning from evaluation results"}
    seal["manifest_sha256"] = digest(seal)
    write_immutable(seal_path, seal)
    return verify_seal(seal_path, inputs)

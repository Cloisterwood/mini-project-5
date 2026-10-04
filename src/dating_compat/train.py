"""Gated SentenceTransformerTrainer runs and a non-scored resume/compute smoke test."""

import gc
import hashlib
import math
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from dating_compat.data import DataError, digest, file_hash, parse_json, write_immutable
from dating_compat.training_data import load_training_inputs


@contextmanager
def evaluation_read_guard():
    """Deny Python file opens of evaluation sources/artifacts inside this worker."""
    enabled = [True]
    def audit(event, args):
        if enabled[0] and event == "open" and isinstance(args[0], (str, bytes)):
            name = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve().name
            if name in {"eval_pairs.jsonl", "eval_pairs_metadata.json", "baseline_metrics.json"} or name.startswith("baseline_eval_"):
                raise DataError("Evaluation files are inaccessible to the training worker")
    sys.addaudithook(audit)
    try:
        yield
    finally:
        enabled[0] = False


def load_config(path: Path = Path("configs/train-default.yaml")) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    baseline = parse_json(Path("configs/baseline.json").read_text(encoding="utf-8"))
    if (config["loss"] != "CosineSimilarityLoss" or config["precision"] != "float32" or
        config["model_id"] != baseline["model_id"] or config["revision"] != baseline["revision"] or
        config["max_seq_length"] != baseline["preprocessing"]["max_seq_length"]):
        raise DataError("Training loss/model/preprocessing differs from frozen contract")
    if config["epochs"] <= 0 or config["batch_size"] <= 0 or not 0 < config["learning_rate"] < 1 or not 0 <= config["warmup_ratio"] <= 1:
        raise DataError("Invalid training hyperparameters")
    return config


def sample_rows(rows: list[dict], count: int, seed: int) -> list[dict]:
    if not 0 < count <= len(rows):
        raise DataError("Smoke sample exceeds allowed split")
    shuffled = rows.copy()
    random.Random(seed).shuffle(shuffled)
    return shuffled[:count]


def model_dataset(rows: list[dict]):
    from datasets import Dataset, Features, Value
    features = Features({"text_1": Value("string"), "text_2": Value("string"), "label": Value("float32")})
    return Dataset.from_dict({key: [row[key] for row in rows] for key in features}, features=features)


def state_hash(model) -> str:
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode()); result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def checkpoint_receipt(path: Path, expected_step: int) -> dict:
    required = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "model.safetensors"}
    if not all((path / name).is_file() for name in required):
        raise DataError("Checkpoint lacks model/optimizer/scheduler/RNG/Trainer state")
    state = parse_json((path / "trainer_state.json").read_text(encoding="utf-8"))
    if state["global_step"] != expected_step:
        raise DataError("Checkpoint step differs from expected resume point")
    import torch
    optimizer = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    scheduler = torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True)
    # Only deserialize our own freshly generated, hashed checkpoint.
    rng = torch.load(path / "rng_state.pth", map_location="cpu", weights_only=False)
    if not optimizer["state"] or not {"python", "numpy", "cpu"} <= set(rng) or scheduler["last_epoch"] != expected_step:
        raise DataError("Checkpoint resume state is incomplete")
    return {"path": str(path), "global_step": expected_step, "scheduler_last_epoch": scheduler["last_epoch"],
            "optimizer_parameter_states": len(optimizer["state"]), "rng_keys": sorted(rng),
            "files": {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob("*")) if p.is_file()}}


def _model(config, device: str, path: Path | None = None):
    import torch
    from sentence_transformers import SentenceTransformer
    if device not in {"cpu", "mps", "cuda"}:
        raise DataError("Explicit device must be cpu, mps or cuda")
    if device == "mps" and not torch.backends.mps.is_available() or device == "cuda" and not torch.cuda.is_available():
        raise DataError(f"Requested device unavailable: {device}")
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(config["seed"]); np.random.seed(config["seed"]); random.seed(config["seed"])
    if path is None:
        manifest = parse_json(Path("reports/base_model_manifest.json").read_text(encoding="utf-8"))
        path = Path("models/base-minilm") / config["revision"]
        if manifest["revision"] != config["revision"] or any(file_hash(path / name) != sha for name, sha in manifest["files"].items()):
            raise DataError("Base-model files differ from pinned revision manifest")
    model = SentenceTransformer(str(path), device=device, local_files_only=True, trust_remote_code=False).float()
    model.max_seq_length = config["max_seq_length"]
    return model


def _trainer(config, model, train_rows, validation_rows, output: Path, device: str,
             total_steps: int = -1, pause_step: int | None = None):
    import torch
    from transformers import TrainerCallback
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import CosineSimilarityLoss
    class Monitor(TrainerCallback):
        def __init__(self):
            self.steps = []
        def sync(self):
            if device == "mps": torch.mps.synchronize()
            if device == "cuda": torch.cuda.synchronize()
        def on_step_begin(self, args, state, control, **kwargs):
            self.sync(); self.started = time.monotonic()
        def on_step_end(self, args, state, control, **kwargs):
            self.sync(); self.steps.append({"step": state.global_step, "seconds": time.monotonic() - self.started})
            if state.global_step == pause_step:
                control.should_training_stop = True
                control.should_save = True
            return control
        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(not math.isfinite(float(v)) for k, v in (logs or {}).items() if k in {"loss", "eval_loss", "grad_norm"}):
                raise DataError("Nonfinite training loss/gradient")
    monitor = Monitor()
    args = SentenceTransformerTrainingArguments(output_dir=str(output),
        num_train_epochs=config["epochs"], max_steps=total_steps,
        per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"], warmup_ratio=config["warmup_ratio"],
        optim=config["optimizer"], weight_decay=config["weight_decay"], max_grad_norm=config["max_grad_norm"],
        seed=config["seed"], data_seed=config["seed"], fp16=False, bf16=False,
        use_cpu=device == "cpu", dataloader_num_workers=0, dataloader_pin_memory=False,
        logging_steps=1 if total_steps > 0 else config["logging_steps"], logging_nan_inf_filter=False,
        eval_strategy="steps" if total_steps > 0 else "epoch", eval_steps=config["smoke"]["pause_step"],
        save_strategy="steps" if total_steps > 0 else "epoch", save_steps=config["smoke"]["pause_step"],
        save_only_model=False, load_best_model_at_end=total_steps < 0,
        metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", push_to_hub=False, disable_tqdm=True,
        prompts={"text_1": "", "text_2": ""})
    if args.device.type != device:
        raise DataError(f"Trainer selected {args.device}, expected {device}")
    trainer = SentenceTransformerTrainer(model=model, args=args,
        train_dataset=model_dataset(train_rows), eval_dataset=model_dataset(validation_rows),
        loss=CosineSimilarityLoss(model), callbacks=[monitor])
    return trainer, monitor


def provenance(config, certificate, device, train_rows, val_rows) -> dict:
    return {"config": config, "device": device, "preflight_manifest_sha256": certificate["manifest_sha256"],
            "fit_ids": [r["record_id"] for r in train_rows], "validation_ids": [r["record_id"] for r in val_rows],
            "python": platform.python_version(), "platform": platform.platform(),
            "software": {name: version(name) for name in ("torch", "sentence-transformers", "transformers", "accelerate", "datasets")},
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "code_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in ("train.py", "training_data.py")}}


def train_smoke(device: str = "cpu") -> dict:
    with evaluation_read_guard():
        inputs, certificate = load_training_inputs()
        config = load_config()
        write_immutable(Path("reports/train_config_manifest.json"), {"config": config, "config_sha256": file_hash(Path("configs/train-default.yaml"))})
        report_path = Path("reports/compute_benchmark.json")
        if report_path.exists():
            report = parse_json(report_path.read_text(encoding="utf-8"))
            if report["provenance"]["preflight_manifest_sha256"] != certificate["manifest_sha256"] or report["provenance"]["config"] != config or report["provenance"]["device"] != device:
                raise DataError("Saved smoke report differs from requested inputs")
            for receipt in report["checkpoints"]:
                for name, sha in receipt["files"].items():
                    if file_hash(Path(receipt["path"]) / name) != sha:
                        raise DataError("Smoke checkpoint modified or missing")
            print("SMOKE SAVED CHECKPOINTS VERIFIED; no training repeated")
            return report
        root = Path("checkpoints/s5-smoke")
        if root.exists():
            raise DataError("Partial smoke output exists; preserve it and inspect before retry")
        fit = sample_rows(inputs["fit"], config["smoke"]["fit_pairs"], config["seed"])
        val = sample_rows(inputs["validation"], config["smoke"]["validation_pairs"], config["seed"])
        evidence = provenance(config, certificate, device, fit, val)
        write_immutable(Path("reports/s5-smoke/run_manifest.json"), evidence)
        total, pause = config["smoke"]["total_steps"], config["smoke"]["pause_step"]
        model = _model(config, device)
        lengths = [len(t) for t in model.tokenizer([r[k] for r in inputs["fit"] for k in ("text_1", "text_2")], truncation=False)["input_ids"]]
        truncation = {"max_fit_tokens": max(lengths), "fit_profile_occurrences_truncated": sum(n > config["max_seq_length"] for n in lengths)}
        initial_hash = state_hash(model)
        trainer, monitor = _trainer(config, model, fit, val, root / "resumed", device, total, pause)
        trainer.train()
        if trainer.state.global_step != pause or state_hash(model) == initial_hash:
            raise DataError("Smoke did not pause at expected step or weights did not change")
        first_checkpoint = root / "resumed" / f"checkpoint-{pause}"
        receipts = [checkpoint_receipt(first_checkpoint, pause)]
        probe = [r["text_1"] for r in fit[:16]]
        before = model.encode(probe, prompt="", normalize_embeddings=True, show_progress_bar=False)
        reload_model = _model(config, device, first_checkpoint)
        after = reload_model.encode(probe, prompt="", normalize_embeddings=True, show_progress_bar=False)
        reload_difference = float(np.max(np.abs(before - after)))
        if state_hash(model) != state_hash(reload_model) or reload_difference > 1e-6:
            raise DataError("Checkpoint model/inference round-trip differs")
        first_timings = monitor.steps
        del trainer, model, reload_model; gc.collect()
        model = _model(config, device)
        trainer, monitor = _trainer(config, model, fit, val, root / "resumed", device, total)
        trainer.train(resume_from_checkpoint=str(first_checkpoint))
        resumed_hash = state_hash(model)
        resumed_history = trainer.state.log_history
        resumed_timings = first_timings + monitor.steps
        receipts.append(checkpoint_receipt(root / "resumed" / f"checkpoint-{total}", total))
        del trainer, model; gc.collect()
        model = _model(config, device)
        trainer, monitor = _trainer(config, model, fit, val, root / "uninterrupted", device, total)
        trainer.train()
        reference_hash = state_hash(model)
        exact_resume = reference_hash == resumed_hash
        if not exact_resume:
            raise DataError("Resumed weights differ from uninterrupted smoke; investigate before scored training")
        receipts.append(checkpoint_receipt(root / "uninterrupted" / f"checkpoint-{total}", total))
        full_val_result = trainer.evaluate(model_dataset(inputs["validation"]))
        step_seconds = [item["seconds"] for item in monitor.steps if item["step"] > 2]
        median, mean = statistics.median(step_seconds), statistics.mean(step_seconds)
        steps = math.ceil(len(inputs["fit"]) / config["batch_size"]) * config["epochs"]
        estimate = steps * median + config["epochs"] * full_val_result["eval_runtime"]
        history = {"resumed": resumed_history, "uninterrupted": trainer.state.log_history,
                   "step_timings": {"resumed": resumed_timings, "uninterrupted": monitor.steps}}
        write_immutable(Path("reports/s5-smoke/loss_history.json"), history)
        report = {"version": "s5-smoke-v1", "status": "passed", "is_scored_experiment": False,
            "provenance": evidence, "initial_weights_sha256": initial_hash,
            "resumed_weights_sha256": resumed_hash, "uninterrupted_weights_sha256": reference_hash,
            "checks": {"finite_training_validation_loss": True, "weights_changed": True,
                       "checkpoint_reload_max_abs_difference": reload_difference, "resume_matches_uninterrupted_exactly": exact_resume,
                       "optimizer_scheduler_rng_present": True, "evaluation_source_access": "Denied by worker file-open audit guard"},
            "truncation": truncation, "checkpoints": receipts,
            "benchmark": {"measured_steps_excluding_two_warmup": len(step_seconds), "median_training_step_seconds": median,
                "mean_training_step_seconds": mean, "training_pairs_per_second": config["batch_size"] / mean,
                "full_validation_runtime_seconds": full_val_result["eval_runtime"], "full_fit_steps_four_epochs": steps,
                "estimated_four_epoch_seconds": estimate, "planning_upper_seconds": estimate * 2,
                "assumptions": "Batch16, same short fit profiles; timing excludes checkpoint I/O; estimate includes four validation passes, 2x planning allowance; not a completed experiment"},
            "compute_decision": {"selected": f"local_{device}", "paid_compute": False,
                "reason": "Real local training and exact resume verified; use measured estimate before full run"},
            "history_path": "reports/s5-smoke/loss_history.json"}
        write_immutable(report_path, report)
        return report


def train_run(run_id: str, config_path: Path, device: str, resume: Path | None = None) -> dict:
    """S6 interface; a user-authorized run uses full fit and validation only."""
    if not re.fullmatch(r"run-00[123]", run_id):
        raise DataError("Run ID must be one of the three planned experiments")
    with evaluation_read_guard():
        inputs, certificate = load_training_inputs()
        config = load_config(config_path)
        output, report_dir = Path("checkpoints") / run_id, Path("reports/runs") / run_id
        evidence = provenance(config, certificate, device, inputs["fit"], inputs["validation"])
        manifest_path = report_dir / "run_manifest.json"
        if resume is not None:
            saved = parse_json(manifest_path.read_text(encoding="utf-8"))
            for key in ("config", "device", "preflight_manifest_sha256", "fit_ids", "validation_ids", "code_sha256", "software"):
                if evidence[key] != saved[key]:
                    raise DataError("Resume configuration/data/code/software differs from frozen run")
            if resume.resolve().parent != output.resolve():
                raise DataError("Resume checkpoint must belong to this run")
            checkpoint_receipt(resume, int(resume.name.split("-")[-1]))
        elif output.exists() or manifest_path.exists():
            raise DataError("Run already exists; use explicit resume without overwriting")
        else:
            write_immutable(manifest_path, evidence)
        model = _model(config, device)
        trainer, _ = _trainer(config, model, inputs["fit"], inputs["validation"], output, device)
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        trainer.save_model(str(output / "selected"))
        report = {"status": "completed", "run_id": run_id, "global_step": trainer.state.global_step,
                  "best_model_checkpoint": trainer.state.best_model_checkpoint,
                  "best_validation_loss": trainer.state.best_metric, "selected_weights_sha256": state_hash(model),
                  "history": trainer.state.log_history, "selected_path": str(output / "selected")}
        write_immutable(report_dir / "training_result.json", report)
        return report

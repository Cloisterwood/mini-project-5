"""Offline environment checks; no dataset or model downloads."""

import importlib
import inspect
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

DEPENDENCIES = {
    "torch": "torch", "sentence-transformers": "sentence_transformers",
    "transformers": "transformers", "datasets": "datasets", "accelerate": "accelerate",
    "pydantic": "pydantic", "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "scikit-learn": "sklearn", "umap-learn": "umap", "hdbscan": "hdbscan",
    "matplotlib": "matplotlib", "seaborn": "seaborn", "PyYAML": "yaml", "pytest": "pytest",
}


def select_device(requested: str, available: dict[str, bool]) -> str:
    if requested == "auto":
        return next(device for device in ("cuda", "mps", "cpu") if available[device])
    if requested not in available or not available[requested]:
        raise ValueError(f"Requested device {requested!r} is unavailable")
    return requested


def tensor_check(torch, device: str) -> dict:
    """Verify float32 matmul and autograd rather than availability flags alone."""
    x = torch.tensor([[1., 2.], [3., 4.]], device=device, requires_grad=True)
    y = x @ x.T
    y.sum().backward()
    expected = torch.tensor([[5., 11.], [11., 25.]])
    if not torch.allclose(y.detach().cpu(), expected):
        raise RuntimeError("Tensor matmul produced unexpected values")
    if x.grad is None or not torch.isfinite(x.grad).all().item():
        raise RuntimeError("Tensor backward produced invalid gradients")
    return {"passed": True, "dtype": str(x.dtype), "device": str(x.device),
            "result": y.detach().cpu().tolist(), "backward_finite": True}


def diagnose(requested: str = "auto") -> dict:
    # Keep import-time font/JIT caches inside the project; respect user overrides.
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.cwd() / ".cache" / "numba"))
    disk = shutil.disk_usage(Path.cwd())
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "working_directory": str(Path.cwd()), "requested_device": requested,
        "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
        "dependencies": {}, "errors": [],
    }
    if not ((3, 12) <= sys.version_info[:2] < (3, 13)):
        report["errors"].append("This environment lock requires Python 3.12")
    for distribution, module in DEPENDENCIES.items():
        try:
            importlib.import_module(module)
            report["dependencies"][distribution] = {"version": version(distribution), "imported": True}
        except Exception as exc:
            report["dependencies"][distribution] = {"imported": False, "error": str(exc)}
            report["errors"].append(f"{distribution}: {type(exc).__name__}: {exc}")
    try:
        import torch
        from sentence_transformers import SentenceTransformerTrainer
        from sentence_transformers.losses import CosineSimilarityLoss
        from hdbscan import HDBSCAN

        signatures = {
            "SentenceTransformerTrainer": inspect.signature(SentenceTransformerTrainer.__init__),
            "CosineSimilarityLoss": inspect.signature(CosineSimilarityLoss.__init__),
        }
        for name, required in (("SentenceTransformerTrainer", {"model", "args", "train_dataset", "eval_dataset", "loss"}),
                               ("CosineSimilarityLoss", {"model"})):
            missing = required - signatures[name].parameters.keys()
            if missing:
                raise RuntimeError(f"{name} missing required parameters: {sorted(missing)}")
        report["training_api"] = {name: str(sig) for name, sig in signatures.items()}
        report["hdbscan"] = {"implementation": f"{HDBSCAN.__module__}.{HDBSCAN.__name__}",
                             "distribution": "hdbscan", "version": version("hdbscan")}
        available = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
        report["devices"] = {**available, "mps_built": torch.backends.mps.is_built()}
        report["selected_device"] = select_device(requested, available)
        report["tensor_check"] = tensor_check(torch, report["selected_device"])
    except Exception as exc:
        report["errors"].append(f"Device/API check: {type(exc).__name__}: {exc}")
    report["passed"] = not report["errors"]
    return report

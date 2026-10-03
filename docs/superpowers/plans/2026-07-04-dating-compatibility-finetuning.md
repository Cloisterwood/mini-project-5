# Dating Compatibility Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune all-MiniLM-L6-v2 with CosineSimilarityLoss so it separates compatible from incompatible dating pairs, and prove the improvement with margin, Cohen's d, FPR, cluster purity, and classification metrics.

**Architecture:** A small Python package (`src/`) with one module per pipeline stage: data loading (Pydantic-validated JSONL), data quality scoring, statistical metrics, model evaluation, training, and visualization. Thin runner scripts in `scripts/` orchestrate the stages and write JSON reports + PNG figures to `reports/`. Every training configuration change gets an entry in `docs/ITERATION_LOG.md`.

**Tech Stack:** Python 3.12, sentence-transformers (>=3.0, SentenceTransformerTrainer API), pydantic v2, numpy/pandas/scipy, scikit-learn, umap-learn, hdbscan, matplotlib/seaborn.

## Global Constraints

- Python 3.10+ (3.12.13 available via pyenv).
- Base model: `sentence-transformers/all-MiniLM-L6-v2`. Loss: `CosineSimilarityLoss`.
- Train ONLY on `data/dating_pairs.jsonl` (6,000 pairs). Evaluate ONLY on `data/eval_pairs.jsonl` (1,469 pairs). Never train on eval data.
- Quality gate: data quality Overall Score ≥ 60/100 before any training.
- Targets (eval set): Cohen's d ≥ 0.50, FPR ≤ 0.35 (stretch ≤ 0.10), cluster purity ≥ 0.65 (stretch ≥ 0.70), margin ≥ 0.05 (stretch ≥ 0.10); classification (threshold 0.5): accuracy/precision/recall/F1/AUC-ROC reported, stretch ≥ 0.90 each.
- Classification threshold for FPR and classification metrics: cosine similarity 0.5.
- Starting hyperparameters: 4 epochs, batch size 16, learning rate 2e-5, warmup ratio 0.1.
- All charts saved as PNG, colorblind-friendly palettes (Set2/tab10), descriptive titles, labeled axes, font ≥ 10pt, grid lines on.
- ≥ 3 iteration log entries; every final config decision traceable to a log entry.
- Deliverables to turn in: `README.md`, `docs/ITERATION_LOG.md`, `docs/SELF_EVALUATION.md`, `reports/*.json`, `reports/figures/*.png`.
- Commit after every task (this project gets its own git repo inside `mini-project-5/`).

## File Structure

```
mini-project-5/
├── README.md                      # how to run, how to interpret (Task 13)
├── requirements.txt               # pinned deps (Task 1)
├── .gitignore                     # venv, models, __pycache__ (Task 1)
├── data/                          # dataset files (user-provided, gitignored large files OK to commit — they're small)
│   ├── dating_pairs.jsonl
│   ├── eval_pairs.jsonl
│   ├── dating_pairs_metadata.json
│   └── eval_pairs_metadata.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Pydantic DatingPair + load_pairs() (Task 2)
│   ├── quality_evaluator.py       # 5-dimension quality scorer (Task 6)
│   ├── metrics.py                 # margin, Cohen's d, FPR, classification, purity (Tasks 4-5)
│   ├── evaluate.py                # encode + metrics + category breakdown → JSON/NPZ (Task 7)
│   ├── train.py                   # fine-tuning pipeline with loss logging (Task 9)
│   └── visualize.py               # all plotting functions (Task 8, 12)
├── scripts/
│   ├── check_data.py              # dataset presence/shape gate (Task 1)
│   ├── explore.py                 # distribution report (Task 3)
│   ├── run_quality_gate.py        # quality evaluator runner (Task 6)
│   ├── run_baseline.py            # baseline evaluation (Task 7)
│   ├── run_training.py            # training runner (Task 10)
│   ├── run_posteval.py            # post-training evaluation (Task 10)
│   └── compare.py                 # baseline vs fine-tuned comparison (Task 10)
├── reports/                       # all JSON reports
│   └── figures/                   # all PNGs
├── models/                        # saved fine-tuned models (gitignored)
├── docs/
│   ├── ITERATION_LOG.md           # required iteration log (Task 10-11)
│   ├── SELF_EVALUATION.md         # answers to the 6 self-eval questions (Task 13)
│   └── superpowers/plans/         # this plan
└── tests/
    ├── test_data_loader.py
    ├── test_metrics.py
    ├── test_cluster_purity.py
    └── test_quality_evaluator.py
```

---

### Task 1: Project scaffold + dataset gate

The project directory is currently **empty** — the dataset described in the spec is not present. This task sets up the environment and a hard gate that verifies the data files before anything else runs. **If the data files are missing, execution stops here until the user supplies them.**

**Files:**
- Create: `requirements.txt`, `.gitignore`, `src/__init__.py`, `scripts/check_data.py`, `data/` (dir), `reports/figures/` (dir), `models/` (dir), `docs/ITERATION_LOG.md` (skeleton)

**Interfaces:**
- Produces: a working venv at `.venv/`; `python scripts/check_data.py` exits 0 iff dataset is valid.

- [ ] **Step 1: Initialize a standalone git repo**

The parent directory is inside an unrelated home-directory git repo. This project should be its own repo for submission.

```bash
cd /Users/jeanneduke/newline-class-projects/mini-project-5
git init
```

- [ ] **Step 2: Write `requirements.txt`**

```text
sentence-transformers>=3.0,<6
datasets>=2.19
torch>=2.2
accelerate>=0.30
pydantic>=2.7
numpy>=1.26
pandas>=2.2
scipy>=1.13
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
umap-learn>=0.5.6
hdbscan>=0.8.36
pytest>=8.0
```

- [ ] **Step 3: Write `.gitignore`**

```text
.venv/
__pycache__/
*.pyc
models/
.pytest_cache/
.DS_Store
```

- [ ] **Step 4: Create venv and install**

```bash
cd /Users/jeanneduke/newline-class-projects/mini-project-5
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

Expected: installs succeed. (hdbscan compiles a C extension; if it fails on macOS, retry with `.venv/bin/pip install hdbscan --no-build-isolation` or fall back to `scikit-learn`'s `HDBSCAN` — sklearn>=1.3 ships `sklearn.cluster.HDBSCAN`, which is the preferred import in Task 5.)

- [ ] **Step 5: Create directories and package marker**

```bash
mkdir -p data src scripts reports/figures models docs tests
touch src/__init__.py
```

- [ ] **Step 6: Write `scripts/check_data.py`**

```python
"""Gate: verify the dataset files exist and have the expected shape."""
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
REQUIRED = {
    "dating_pairs.jsonl": 6000,
    "eval_pairs.jsonl": 1469,
}
FIELDS = {"text_1", "text_2", "label", "category", "subcategory", "pair_type"}


def main() -> int:
    ok = True
    for name, expected_count in REQUIRED.items():
        path = DATA / name
        if not path.exists():
            print(f"MISSING: {path}")
            ok = False
            continue
        n = 0
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                n += 1
                rec = json.loads(line)
                missing = FIELDS - rec.keys()
                if missing:
                    print(f"{name}:{i} missing fields {missing}")
                    ok = False
                if rec.get("label") not in (0, 1):
                    print(f"{name}:{i} bad label {rec.get('label')!r}")
                    ok = False
        print(f"{name}: {n} pairs (expected ~{expected_count})")
        if abs(n - expected_count) > expected_count * 0.05:
            print(f"WARNING: {name} count deviates >5% from spec")
    if not ok:
        print("\nDATASET GATE FAILED. Place the provided dataset files in data/ "
              "(dating_pairs.jsonl, eval_pairs.jsonl, and the two metadata JSON "
              "files) and re-run.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: Run the gate**

```bash
.venv/bin/python scripts/check_data.py
```

Expected right now: **FAIL with MISSING messages** — the data is not yet in place. **STOP and ask the user to drop the four dataset files into `data/`.** Do not proceed to Task 2 until this exits 0.

- [ ] **Step 8: Write `docs/ITERATION_LOG.md` skeleton**

```markdown
# Iteration Log — Dating Compatibility Fine-Tuning

Every training configuration change is recorded here. Final configuration
decisions must be traceable to a specific entry.

| # | Change | Reason | d Before | d After | Delta | Keep? |
|---|--------|--------|----------|---------|-------|-------|
| 1 | Baseline (no training) | Starting point | — | (filled by Task 7) | — | — |

## Detailed entries

(One section per entry, added as iterations run.)
```

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "chore: project scaffold, requirements, dataset gate"
```

---

### Task 2: Pydantic data model + JSONL loader

**Files:**
- Create: `src/data_loader.py`
- Test: `tests/test_data_loader.py`

**Interfaces:**
- Produces: `DatingPair` (pydantic BaseModel with fields `text_1: str, text_2: str, label: int, category: str, subcategory: str, pair_type: str`) and `load_pairs(path: str | Path) -> list[DatingPair]`. All later tasks consume `list[DatingPair]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data_loader.py
import json
import pytest
from src.data_loader import DatingPair, load_pairs

VALID = {
    "text_1": "boy: I love hiking",
    "text_2": "girl: I enjoy the outdoors",
    "label": 1,
    "category": "interests",
    "subcategory": "fitness",
    "pair_type": "compatible",
}


def test_valid_pair_parses():
    p = DatingPair(**VALID)
    assert p.label == 1
    assert p.text_1.startswith("boy:")


def test_label_must_be_binary():
    with pytest.raises(Exception):
        DatingPair(**{**VALID, "label": 2})


def test_empty_text_rejected():
    with pytest.raises(Exception):
        DatingPair(**{**VALID, "text_1": "   "})


def test_load_pairs_reads_jsonl(tmp_path):
    f = tmp_path / "pairs.jsonl"
    f.write_text(json.dumps(VALID) + "\n" + json.dumps({**VALID, "label": 0}) + "\n")
    pairs = load_pairs(f)
    assert len(pairs) == 2
    assert pairs[1].label == 0


def test_load_pairs_reports_bad_line(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_text(json.dumps(VALID) + "\n{not json}\n")
    with pytest.raises(ValueError, match="line 2"):
        load_pairs(f)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_data_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data_loader'`

- [ ] **Step 3: Implement `src/data_loader.py`**

```python
"""Pydantic-validated JSONL loading for dating pair records."""
import json
from pathlib import Path

from pydantic import BaseModel, field_validator


class DatingPair(BaseModel):
    text_1: str
    text_2: str
    label: int
    category: str
    subcategory: str
    pair_type: str

    @field_validator("label")
    @classmethod
    def label_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("label must be 0 or 1")
        return v

    @field_validator("text_1", "text_2")
    @classmethod
    def text_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must be non-empty")
        return v


def load_pairs(path: str | Path) -> list[DatingPair]:
    pairs: list[DatingPair] = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(DatingPair(**json.loads(line)))
            except Exception as e:
                raise ValueError(f"{path} line {i}: {e}") from e
    return pairs
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_data_loader.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_loader.py tests/test_data_loader.py
git commit -m "feat: pydantic DatingPair model and JSONL loader"
```

---

### Task 3: Data exploration report

**Files:**
- Create: `scripts/explore.py`
- Output: `reports/exploration_report.json`, `reports/figures/label_distribution.png`, `reports/figures/pair_type_distribution.png`, `reports/figures/category_distribution.png`

**Interfaces:**
- Consumes: `load_pairs` from Task 2.
- Produces: `reports/exploration_report.json` with keys `total`, `label_counts`, `pair_type_counts`, `category_counts`, `subcategory_counts`, `gender_first_counts`, `text_length_stats`, `duplicate_rate` for each of train/eval.

- [ ] **Step 1: Write `scripts/explore.py`**

```python
"""Explore dataset structure: labels, pair types, categories, text stats."""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data_loader import load_pairs

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)


def summarize(pairs):
    texts = [p.text_1 for p in pairs] + [p.text_2 for p in pairs]
    lengths = [len(t.split()) for t in texts]
    pair_keys = [(p.text_1, p.text_2, p.label) for p in pairs]
    dupes = len(pair_keys) - len(set(pair_keys))
    return {
        "total": len(pairs),
        "label_counts": dict(Counter(p.label for p in pairs)),
        "pair_type_counts": dict(Counter(p.pair_type for p in pairs)),
        "category_counts": dict(Counter(p.category for p in pairs)),
        "subcategory_counts": dict(Counter(p.subcategory for p in pairs)),
        "gender_first_counts": dict(
            Counter(p.text_1.split(":")[0].strip() for p in pairs)
        ),
        "text_length_stats": {
            "mean_words": float(np.mean(lengths)),
            "min_words": int(np.min(lengths)),
            "max_words": int(np.max(lengths)),
        },
        "duplicate_rate": dupes / len(pairs) if pairs else 0.0,
    }


def bar(counter: dict, title: str, xlabel: str, fname: str):
    items = sorted(counter.items(), key=lambda kv: -kv[1])
    keys = [str(k) for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=keys, y=vals, hue=keys, legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of pairs")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def main():
    report = {}
    train = load_pairs(ROOT / "data" / "dating_pairs.jsonl")
    evalp = load_pairs(ROOT / "data" / "eval_pairs.jsonl")
    report["train"] = summarize(train)
    report["eval"] = summarize(evalp)
    # train/eval overlap check
    train_keys = {(p.text_1, p.text_2) for p in train}
    eval_keys = {(p.text_1, p.text_2) for p in evalp}
    report["train_eval_overlap"] = len(train_keys & eval_keys)

    bar(report["train"]["label_counts"],
        "Training Label Distribution: Compatible (1) vs Incompatible (0)",
        "Label", "label_distribution.png")
    bar(report["train"]["pair_type_counts"],
        "Training Pair Type Distribution Across 9 Scenario Types",
        "Pair type", "pair_type_distribution.png")
    bar(report["train"]["category_counts"],
        "Training Category Distribution: Base and Compound Categories",
        "Category", "category_distribution.png")

    out = ROOT / "reports" / "exploration_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k]["total"] if isinstance(report[k], dict) else report[k]
                      for k in report}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python scripts/explore.py`
Expected: prints totals (~6000 train / ~1469 eval), `train_eval_overlap: 0`, writes JSON + 3 PNGs. Eyeball the printed distributions against the spec's metadata numbers (≈2997/3003 train labels).

- [ ] **Step 3: Commit**

```bash
git add scripts/explore.py reports/exploration_report.json reports/figures/*.png
git commit -m "feat: dataset exploration report and distribution plots"
```

---

### Task 4: Core statistical metrics (margin, Cohen's d, FPR, classification)

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces (all take `sims: np.ndarray` of cosine similarities and `labels: np.ndarray` of 0/1):
  - `cosine_similarities(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray` (row-wise)
  - `margin(sims, labels) -> float`
  - `cohens_d(sims, labels) -> float`
  - `false_positive_rate(sims, labels, threshold: float = 0.5) -> float`
  - `classification_metrics(sims, labels, threshold: float = 0.5) -> dict` with keys `accuracy, precision, recall, f1, auc_roc`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import numpy as np
import pytest
from src.metrics import (
    cosine_similarities, margin, cohens_d,
    false_positive_rate, classification_metrics,
)


def test_cosine_similarities_identical_and_orthogonal():
    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([[2.0, 0.0], [0.0, 3.0]])
    sims = cosine_similarities(a, b)
    assert sims == pytest.approx([1.0, 0.0], abs=1e-6)


def test_margin_is_mean_difference():
    sims = np.array([0.8, 0.9, 0.2, 0.3])
    labels = np.array([1, 1, 0, 0])
    assert margin(sims, labels) == pytest.approx(0.85 - 0.25)


def test_cohens_d_perfect_separation_is_large():
    sims = np.array([0.9, 0.91, 0.89, 0.1, 0.11, 0.09])
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert cohens_d(sims, labels) > 5


def test_cohens_d_no_separation_near_zero():
    rng = np.random.default_rng(0)
    sims = rng.normal(0.5, 0.1, 1000)
    labels = rng.integers(0, 2, 1000)
    assert abs(cohens_d(sims, labels)) < 0.15


def test_false_positive_rate():
    sims = np.array([0.9, 0.6, 0.4, 0.2])
    labels = np.array([1, 0, 0, 0])
    # incompatible sims: 0.6, 0.4, 0.2 -> one >= 0.5
    assert false_positive_rate(sims, labels) == pytest.approx(1 / 3)


def test_classification_metrics_perfect():
    sims = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])
    m = classification_metrics(sims, labels)
    for k in ("accuracy", "precision", "recall", "f1", "auc_roc"):
        assert m[k] == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.metrics'`

- [ ] **Step 3: Implement `src/metrics.py`**

```python
"""Statistical metrics for embedding-based compatibility evaluation."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def cosine_similarities(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    e1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    e2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    return np.sum(e1 * e2, axis=1)


def margin(sims: np.ndarray, labels: np.ndarray) -> float:
    return float(sims[labels == 1].mean() - sims[labels == 0].mean())


def cohens_d(sims: np.ndarray, labels: np.ndarray) -> float:
    a, b = sims[labels == 1], sims[labels == 0]
    na, nb = len(a), len(b)
    pooled = np.sqrt(
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
        / (na + nb - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def false_positive_rate(sims, labels, threshold: float = 0.5) -> float:
    incompat = sims[labels == 0]
    return float((incompat >= threshold).mean()) if len(incompat) else 0.0


def classification_metrics(sims, labels, threshold: float = 0.5) -> dict:
    preds = (sims >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(labels, sims)),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: core statistical metrics (margin, Cohen's d, FPR, classification)"
```

---

### Task 5: Pair features, UMAP projection, cluster purity

**Files:**
- Modify: `src/metrics.py` (append three functions)
- Test: `tests/test_cluster_purity.py`

**Interfaces:**
- Produces (appended to `src/metrics.py`):
  - `pair_features(emb1, emb2) -> np.ndarray` — symmetric per-pair feature `[|u−v| ; u*v]`, shape (n, 2·dim)
  - `umap_2d(features, seed: int = 42) -> np.ndarray` — shape (n, 2)
  - `cluster_purity(points_2d, labels, min_cluster_size: int = 15) -> float` — HDBSCAN clusters on the 2D projection; purity = Σ per-cluster majority-label count / n, noise points (-1) treated as one cluster

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cluster_purity.py
import numpy as np
import pytest
from src.metrics import pair_features, cluster_purity


def test_pair_features_symmetric():
    a = np.array([[1.0, 2.0]])
    b = np.array([[3.0, 1.0]])
    f_ab = pair_features(a, b)
    f_ba = pair_features(b, a)
    assert f_ab.shape == (1, 4)
    assert np.allclose(f_ab, f_ba)


def test_cluster_purity_separated_blobs_high():
    rng = np.random.default_rng(0)
    blob1 = rng.normal([0, 0], 0.1, (100, 2))
    blob2 = rng.normal([10, 10], 0.1, (100, 2))
    pts = np.vstack([blob1, blob2])
    labels = np.array([1] * 100 + [0] * 100)
    assert cluster_purity(pts, labels) > 0.9


def test_cluster_purity_mixed_blob_near_half():
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 1.0, (200, 2))
    labels = rng.integers(0, 2, 200)
    assert cluster_purity(pts, labels) < 0.7
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_cluster_purity.py -v`
Expected: FAIL with `ImportError: cannot import name 'pair_features'`

- [ ] **Step 3: Append to `src/metrics.py`**

```python
def pair_features(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Symmetric per-pair feature vector for clustering/UMAP."""
    return np.concatenate([np.abs(emb1 - emb2), emb1 * emb2], axis=1)


def umap_2d(features: np.ndarray, seed: int = 42) -> np.ndarray:
    import umap  # deferred: slow import

    return umap.UMAP(n_components=2, random_state=seed).fit_transform(features)


def cluster_purity(points_2d: np.ndarray, labels: np.ndarray,
                   min_cluster_size: int = 15) -> float:
    from sklearn.cluster import HDBSCAN  # sklearn>=1.3 native implementation

    assignments = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(points_2d)
    correct = 0
    for c in np.unique(assignments):
        mask = assignments == c
        counts = np.bincount(labels[mask].astype(int), minlength=2)
        correct += counts.max()
    return float(correct / len(labels))
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_cluster_purity.py -v`
Expected: 3 passed

- [ ] **Step 5: Run full suite and commit**

Run: `.venv/bin/python -m pytest -v` — expected: all pass.

```bash
git add src/metrics.py tests/test_cluster_purity.py
git commit -m "feat: pair features, UMAP helper, HDBSCAN cluster purity"
```

---

### Task 6: Data quality evaluator (5 dimensions + quality gate)

**Files:**
- Create: `src/quality_evaluator.py`, `scripts/run_quality_gate.py`
- Test: `tests/test_quality_evaluator.py`
- Output: `reports/data_quality_report.json`

**Interfaces:**
- Consumes: `list[DatingPair]` from Task 2.
- Produces: `evaluate_quality(pairs: list[DatingPair]) -> dict` with keys `data_quality, diversity, bias_detection, linguistic_quality, real_life_matching, overall` (each 0–100, `overall` = mean of the five) plus a `details` dict of sub-scores. Gate: `overall >= 60`.

Each dimension is the mean of 4 sub-scores, each sub-score clipped to [0, 100]:

- **data_quality**: completeness (% records with all 6 fields non-empty), label validity (% labels in {0,1}), duplicate score (`100·(1 − dup_rate/0.05)` so <5% dupes ⇒ high), format score (% texts matching `^(boy|girl):`)
- **diversity**: vocabulary richness (unique words / total words, scaled ×500, capped), category coverage (unique categories / 12 × 100), label balance (`100 − |pct_compatible − 50|·(100/10)` clipped — full marks at 50/50, zero at 40/60 boundary breach), pair-type variety (unique pair types / 9 × 100)
- **bias_detection**: gender-order balance (100 − |boy-first% − 50|×4), length bias (100·(1 − |pointbiserial r(word count, label)|)), per-category label skew (mean over categories with ≥30 pairs of `100 − |pct_compat − 50|·2`), vocabulary bias (100·(1 − Jaccard-gap) where gap = fraction of top-50 words exclusive to one label class)
- **linguistic_quality**: length sanity (% texts with 4–60 words), repetition (100·unique-bigram ratio averaged over texts), lexical naturalness (% texts containing at least one first-person token `i|my|me`), punctuation sanity (% texts without repeated punctuation like `!!` or `??`)
- **real_life_matching**: realistic pair presence (min(realistic_count,20)/20 × 100 using pair types `curated_realistic`), dealbreaker coverage (min(dealbreaker%,15)/15 × 100), subtle mismatch coverage (min(subtle%,19)/19 × 100), compound category presence (min(compound_category_count,7)/7 × 100)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_quality_evaluator.py
from src.data_loader import DatingPair
from src.quality_evaluator import evaluate_quality


def make_pairs(n=100):
    pairs = []
    subjects = ["hiking", "cooking", "jazz music", "travel", "reading",
                "yoga", "gardening", "board games", "photography", "running"]
    for i in range(n):
        g1, g2 = ("boy", "girl") if i % 2 == 0 else ("girl", "boy")
        pairs.append(DatingPair(
            text_1=f"{g1}: I really enjoy {subjects[i % 10]} on my weekends",
            text_2=f"{g2}: my favorite thing is {subjects[(i + 3) % 10]} with friends",
            label=i % 2,
            category=["lifestyle", "interests", "values", "dealbreakers",
                      "interests_and_values"][i % 5],
            subcategory=f"sub_{i % 8}",
            pair_type=["compatible", "incompatible", "dealbreaker",
                       "subtle_mismatch", "curated_realistic"][i % 5],
        ))
    return pairs


def test_returns_all_dimensions():
    r = evaluate_quality(make_pairs())
    for k in ("data_quality", "diversity", "bias_detection",
              "linguistic_quality", "real_life_matching", "overall"):
        assert k in r
        assert 0 <= r[k] <= 100


def test_overall_is_mean_of_dimensions():
    r = evaluate_quality(make_pairs())
    dims = [r["data_quality"], r["diversity"], r["bias_detection"],
            r["linguistic_quality"], r["real_life_matching"]]
    assert abs(r["overall"] - sum(dims) / 5) < 0.01


def test_balanced_wellformed_data_scores_reasonably():
    r = evaluate_quality(make_pairs(200))
    assert r["data_quality"] > 80
    assert r["overall"] > 50


def test_duplicates_lower_data_quality():
    pairs = make_pairs(50)
    duped = pairs + pairs[:25]  # 33% duplicates
    assert (evaluate_quality(duped)["data_quality"]
            < evaluate_quality(pairs)["data_quality"])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_quality_evaluator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.quality_evaluator'`

- [ ] **Step 3: Implement `src/quality_evaluator.py`**

```python
"""Score dataset quality across 5 dimensions; gate at overall >= 60/100."""
import re
from collections import Counter

import numpy as np
from scipy.stats import pointbiserialr

from src.data_loader import DatingPair

GENDER_RE = re.compile(r"^(boy|girl):")
FIRST_PERSON_RE = re.compile(r"\b(i|my|me)\b", re.IGNORECASE)
BAD_PUNCT_RE = re.compile(r"[!?.]{2,}")


def _clip(x: float) -> float:
    return float(np.clip(x, 0.0, 100.0))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _score_data_quality(pairs: list[DatingPair]) -> dict:
    n = len(pairs)
    complete = sum(
        1 for p in pairs
        if all([p.text_1.strip(), p.text_2.strip(), p.category.strip(),
                p.subcategory.strip(), p.pair_type.strip()])
        and p.label in (0, 1)
    ) / n * 100
    valid_labels = sum(1 for p in pairs if p.label in (0, 1)) / n * 100
    keys = [(p.text_1, p.text_2, p.label) for p in pairs]
    dup_rate = (len(keys) - len(set(keys))) / n
    dup_score = _clip(100 * (1 - dup_rate / 0.05))
    fmt = sum(1 for p in pairs
              if GENDER_RE.match(p.text_1) and GENDER_RE.match(p.text_2)) / n * 100
    return {"completeness": _clip(complete), "label_validity": _clip(valid_labels),
            "duplicates": dup_score, "format": _clip(fmt)}


def _score_diversity(pairs: list[DatingPair]) -> dict:
    all_words = [w for p in pairs for w in _words(p.text_1) + _words(p.text_2)]
    ttr = len(set(all_words)) / len(all_words) if all_words else 0
    vocab = _clip(ttr * 500)
    categories = {p.category for p in pairs}
    cat_cov = _clip(len(categories) / 12 * 100)
    pct_compat = sum(p.label for p in pairs) / len(pairs) * 100
    balance = _clip(100 - abs(pct_compat - 50) * 10)
    types = {p.pair_type for p in pairs}
    variety = _clip(len(types) / 9 * 100)
    return {"vocabulary": vocab, "category_coverage": cat_cov,
            "label_balance": balance, "pair_type_variety": variety}


def _score_bias(pairs: list[DatingPair]) -> dict:
    boy_first = sum(1 for p in pairs if p.text_1.startswith("boy:")) / len(pairs) * 100
    gender = _clip(100 - abs(boy_first - 50) * 4)

    lengths = np.array([len(_words(p.text_1)) + len(_words(p.text_2)) for p in pairs])
    labels = np.array([p.label for p in pairs])
    if len(set(labels)) > 1 and lengths.std() > 0:
        r, _ = pointbiserialr(labels, lengths)
        length_bias = _clip(100 * (1 - abs(r)))
    else:
        length_bias = 100.0

    cat_scores = []
    for cat, cnt in Counter(p.category for p in pairs).items():
        if cnt < 30:
            continue
        pct = sum(p.label for p in pairs if p.category == cat) / cnt * 100
        cat_scores.append(_clip(100 - abs(pct - 50) * 2))
    cat_skew = float(np.mean(cat_scores)) if cat_scores else 100.0

    def top_words(label):
        c = Counter(w for p in pairs if p.label == label
                    for w in _words(p.text_1) + _words(p.text_2))
        return {w for w, _ in c.most_common(50)}
    t1, t0 = top_words(1), top_words(0)
    union = t1 | t0
    exclusive = (len(t1 - t0) + len(t0 - t1)) / len(union) if union else 0
    vocab_bias = _clip(100 * (1 - exclusive))
    return {"gender_balance": gender, "length_bias": length_bias,
            "category_skew": cat_skew, "vocabulary_bias": vocab_bias}


def _score_linguistic(pairs: list[DatingPair]) -> dict:
    texts = [p.text_1 for p in pairs] + [p.text_2 for p in pairs]
    n = len(texts)
    length_ok = sum(1 for t in texts if 4 <= len(_words(t)) <= 60) / n * 100

    def bigram_ratio(t):
        ws = _words(t)
        if len(ws) < 2:
            return 1.0
        bigrams = list(zip(ws, ws[1:]))
        return len(set(bigrams)) / len(bigrams)
    repetition = float(np.mean([bigram_ratio(t) for t in texts]) * 100)
    first_person = sum(1 for t in texts if FIRST_PERSON_RE.search(t)) / n * 100
    punct = sum(1 for t in texts if not BAD_PUNCT_RE.search(t)) / n * 100
    return {"length_sanity": _clip(length_ok), "repetition": _clip(repetition),
            "naturalness": _clip(first_person), "punctuation": _clip(punct)}


def _score_realism(pairs: list[DatingPair]) -> dict:
    n = len(pairs)
    types = Counter(p.pair_type for p in pairs)
    realistic = _clip(min(types.get("curated_realistic", 0), 20) / 20 * 100)
    dealbreaker_pct = types.get("dealbreaker", 0) / n * 100
    dealbreakers = _clip(min(dealbreaker_pct, 15) / 15 * 100)
    subtle_pct = types.get("subtle_mismatch", 0) / n * 100
    subtle = _clip(min(subtle_pct, 19) / 19 * 100)
    compound = len({p.category for p in pairs if "_and_" in p.category})
    compound_score = _clip(min(compound, 7) / 7 * 100)
    return {"realistic_pairs": realistic, "dealbreaker_coverage": dealbreakers,
            "subtle_mismatch_coverage": subtle, "compound_categories": compound_score}


def evaluate_quality(pairs: list[DatingPair]) -> dict:
    details = {
        "data_quality": _score_data_quality(pairs),
        "diversity": _score_diversity(pairs),
        "bias_detection": _score_bias(pairs),
        "linguistic_quality": _score_linguistic(pairs),
        "real_life_matching": _score_realism(pairs),
    }
    result = {dim: float(np.mean(list(subs.values())))
              for dim, subs in details.items()}
    result["overall"] = float(np.mean(list(result.values())))
    result["details"] = details
    return result
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_quality_evaluator.py -v`
Expected: 4 passed. (If `test_balanced_wellformed_data_scores_reasonably` fails on a sub-score, inspect `details` and tune the offending heuristic constant — the synthetic fixture is well-formed and balanced, so it should score comfortably.)

- [ ] **Step 5: Write `scripts/run_quality_gate.py`**

```python
"""Run the 5-dimension quality evaluator on the training set; enforce gate."""
import json
import sys
from pathlib import Path

from src.data_loader import load_pairs
from src.quality_evaluator import evaluate_quality

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    pairs = load_pairs(ROOT / "data" / "dating_pairs.jsonl")
    report = evaluate_quality(pairs)
    out = ROOT / "reports" / "data_quality_report.json"
    out.write_text(json.dumps(report, indent=2))
    for dim in ("data_quality", "diversity", "bias_detection",
                "linguistic_quality", "real_life_matching", "overall"):
        print(f"{dim:22s} {report[dim]:6.1f} / 100")
    if report["overall"] < 60:
        print("\nQUALITY GATE FAILED (overall < 60). Inspect the lowest "
              "dimension's sub-scores in reports/data_quality_report.json "
              "and document findings before proceeding.")
        return 1
    print("\nQuality gate PASSED (overall >= 60). OK to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the quality gate on the real dataset**

Run: `.venv/bin/python scripts/run_quality_gate.py`
Expected: prints five dimension scores + overall; exits 0 with "PASSED". If it fails, investigate the lowest dimension per the spec (document findings in `docs/ITERATION_LOG.md` under a "Data findings" note) before proceeding — do not train on gated-out data.

- [ ] **Step 7: Commit**

```bash
git add src/quality_evaluator.py tests/test_quality_evaluator.py scripts/run_quality_gate.py reports/data_quality_report.json
git commit -m "feat: 5-dimension data quality evaluator with 60/100 gate"
```

---

### Task 7: Evaluation runner + baseline metrics

**Files:**
- Create: `src/evaluate.py`, `scripts/run_baseline.py`
- Output: `reports/baseline_metrics.json`, `reports/baseline_arrays.npz`

**Interfaces:**
- Consumes: `load_pairs`, everything in `src/metrics.py`.
- Produces: `evaluate_model(model_name_or_path: str, pairs: list[DatingPair], seed: int = 42) -> tuple[dict, dict]` returning `(report, arrays)`:
  - `report` keys: `model`, `margin`, `effect_size`, `false_positive_rate`, `cluster_purity`, `compatible_mean_similarity`, `incompatible_mean_similarity`, `accuracy`, `precision`, `recall`, `f1`, `auc_roc`, `by_base_category` (dict: category → {n, effect_size, false_positive_rate}), `by_pair_type` (dict: pair_type → {n, false_positive_rate, mean_similarity})
  - `arrays` keys: `sims` (n,), `labels` (n,), `umap_xy` (n, 2) — saved as NPZ for the visualization suite
- Base-category attribution: a pair counts under every base word (`dealbreakers`, `values`, `lifestyle`, `interests`, `realistic_example`) contained in its `category` string, so `interests_and_values` contributes to both `interests` and `values`.

- [ ] **Step 1: Implement `src/evaluate.py`**

```python
"""Evaluate any sentence-transformer on the eval pairs; produce full report."""
import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_loader import DatingPair
from src.metrics import (
    classification_metrics, cluster_purity, cohens_d, cosine_similarities,
    false_positive_rate, margin, pair_features, umap_2d,
)

BASE_CATEGORIES = ("dealbreakers", "values", "lifestyle", "interests",
                   "realistic_example")


def evaluate_model(model_name_or_path: str, pairs: list[DatingPair],
                   seed: int = 42) -> tuple[dict, dict]:
    model = SentenceTransformer(model_name_or_path)
    t1 = [p.text_1 for p in pairs]
    t2 = [p.text_2 for p in pairs]
    e1 = model.encode(t1, batch_size=64, convert_to_numpy=True,
                      show_progress_bar=True)
    e2 = model.encode(t2, batch_size=64, convert_to_numpy=True,
                      show_progress_bar=True)
    sims = cosine_similarities(e1, e2)
    labels = np.array([p.label for p in pairs])

    feats = pair_features(e1, e2)
    xy = umap_2d(feats, seed=seed)

    report = {
        "model": model_name_or_path,
        "margin": margin(sims, labels),
        "effect_size": cohens_d(sims, labels),
        "false_positive_rate": false_positive_rate(sims, labels),
        "cluster_purity": cluster_purity(xy, labels),
        "compatible_mean_similarity": float(sims[labels == 1].mean()),
        "incompatible_mean_similarity": float(sims[labels == 0].mean()),
        **classification_metrics(sims, labels),
    }

    by_cat = {}
    for base in BASE_CATEGORIES:
        idx = np.array([base in p.category for p in pairs])
        if idx.sum() < 10 or len(set(labels[idx])) < 2:
            continue
        by_cat[base] = {
            "n": int(idx.sum()),
            "effect_size": cohens_d(sims[idx], labels[idx]),
            "false_positive_rate": false_positive_rate(sims[idx], labels[idx]),
        }
    report["by_base_category"] = by_cat

    by_type = {}
    for ptype in sorted({p.pair_type for p in pairs}):
        idx = np.array([p.pair_type == ptype for p in pairs])
        sub_labels = labels[idx]
        by_type[ptype] = {
            "n": int(idx.sum()),
            "mean_similarity": float(sims[idx].mean()),
            "false_positive_rate": (
                false_positive_rate(sims[idx], sub_labels)
                if (sub_labels == 0).any() else None
            ),
        }
    report["by_pair_type"] = by_type

    arrays = {"sims": sims, "labels": labels, "umap_xy": xy}
    return report, arrays
```

- [ ] **Step 2: Write `scripts/run_baseline.py`**

```python
"""Baseline: pre-trained all-MiniLM-L6-v2 on the eval set."""
import json
from pathlib import Path

import numpy as np

from src.data_loader import load_pairs
from src.evaluate import evaluate_model

ROOT = Path(__file__).resolve().parents[1]
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    pairs = load_pairs(ROOT / "data" / "eval_pairs.jsonl")
    report, arrays = evaluate_model(MODEL, pairs)
    (ROOT / "reports" / "baseline_metrics.json").write_text(
        json.dumps(report, indent=2))
    np.savez(ROOT / "reports" / "baseline_arrays.npz", **arrays)
    print(json.dumps({k: v for k, v in report.items()
                      if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test on 20 pairs first**

Run:
```bash
.venv/bin/python - <<'EOF'
from src.data_loader import load_pairs
from src.evaluate import evaluate_model
pairs = load_pairs("data/eval_pairs.jsonl")[:20]
report, arrays = evaluate_model("sentence-transformers/all-MiniLM-L6-v2", pairs)
assert arrays["sims"].shape == (20,)
assert arrays["umap_xy"].shape == (20, 2)
print("smoke OK:", {k: round(v, 3) for k, v in report.items() if isinstance(v, float)})
EOF
```
Expected: downloads the model (~90 MB) on first run, prints `smoke OK` with plausible numbers. (UMAP on 20 points may warn about n_neighbors — fine for a smoke test.)

- [ ] **Step 4: Run the full baseline**

Run: `.venv/bin/python scripts/run_baseline.py`
Expected: full 1,469-pair evaluation (a few minutes on CPU), writes `reports/baseline_metrics.json` and `reports/baseline_arrays.npz`. Sanity-check the numbers against spec expectations: effect_size ≈ 0.0–0.1 (near-zero — the model can't tell compatibility), FPR ≈ 0.2–0.5, auc_roc ≈ 0.5. **If effect_size > 0.3, flag it**: the spec says the dataset may then be too easy; note this in the iteration log before continuing.

- [ ] **Step 5: Record baseline in the iteration log**

Fill entry 1's "d After" column in `docs/ITERATION_LOG.md` with the measured baseline effect size, and add a detailed section:

```markdown
### Entry 1 — Baseline (no training)
- Config: pre-trained sentence-transformers/all-MiniLM-L6-v2, no fine-tuning
- Metrics: d=<measured>, margin=<measured>, FPR=<measured>, purity=<measured>, AUC=<measured>
- Notes: establishes the comparison point; saved to reports/baseline_metrics.json
```

- [ ] **Step 6: Commit**

```bash
git add src/evaluate.py scripts/run_baseline.py reports/baseline_metrics.json reports/baseline_arrays.npz docs/ITERATION_LOG.md
git commit -m "feat: evaluation runner and baseline metrics"
```

---

### Task 8: Visualization suite (baseline plots now, comparison plots wired for later)

**Files:**
- Create: `src/visualize.py`, `scripts/make_baseline_figures.py`
- Output: `reports/figures/similarity_distribution_baseline.png`, `reports/figures/umap_baseline.png`

**Interfaces:**
- Consumes: NPZ arrays (`sims`, `labels`, `umap_xy`) from Task 7; metrics JSONs.
- Produces functions (all save PNG at 150 dpi and take `out_path: Path`):
  - `plot_similarity_distributions(runs: list[tuple[str, np.ndarray, np.ndarray]], out_path)` — one subplot per run (name, sims, labels), overlaid histograms per label
  - `plot_umap(runs: list[tuple[str, np.ndarray, np.ndarray]], out_path)` — one subplot per run (name, umap_xy, labels)
  - `plot_roc(runs: list[tuple[str, np.ndarray, np.ndarray]], out_path)` — overlaid ROC curves (name, sims, labels), AUC in legend
  - `plot_category_comparison(baseline_by_cat: dict, finetuned_by_cat: dict, out_path)` — grouped bars of effect size per base category
  - `plot_delta(baseline: dict, finetuned: dict, out_path)` — bar chart of metric deltas with zero line
  - `plot_loss_curve(history: list[dict], out_path)` — training loss vs step, epoch boundaries marked

- [ ] **Step 1: Implement `src/visualize.py`**

```python
"""All required PNG visualizations. Colorblind-friendly, gridded, >=10pt."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
COMPAT_COLOR, INCOMPAT_COLOR = sns.color_palette("Set2")[0], sns.color_palette("Set2")[1]
CORE_METRICS = ("margin", "effect_size", "false_positive_rate", "cluster_purity")


def _save(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_similarity_distributions(runs, out_path: Path):
    fig, axes = plt.subplots(1, len(runs), figsize=(7 * len(runs), 5),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (name, sims, labels) in zip(axes[0], runs):
        ax.hist(sims[labels == 1], bins=40, alpha=0.6, label="Compatible",
                color=COMPAT_COLOR)
        ax.hist(sims[labels == 0], bins=40, alpha=0.6, label="Incompatible",
                color=INCOMPAT_COLOR)
        ax.axvline(0.5, color="gray", linestyle="--", label="Threshold 0.5")
        ax.set_title(f"Cosine Similarity Distribution: Compatible vs "
                     f"Incompatible Pairs ({name})")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Pair count")
        ax.legend()
    _save(fig, out_path)


def plot_umap(runs, out_path: Path):
    fig, axes = plt.subplots(1, len(runs), figsize=(7 * len(runs), 6),
                             squeeze=False)
    for ax, (name, xy, labels) in zip(axes[0], runs):
        for lab, color, text in ((1, COMPAT_COLOR, "Compatible"),
                                 (0, INCOMPAT_COLOR, "Incompatible")):
            m = labels == lab
            ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.5, c=[color], label=text)
        ax.set_title(f"UMAP Projection of Pair Embeddings by "
                     f"Compatibility Label ({name})")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(markerscale=3)
    _save(fig, out_path)


def plot_roc(runs, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, sims, labels in runs:
        fpr, tpr, _ = roc_curve(labels, sims)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set_title("ROC Curves: Baseline vs Fine-Tuned Compatibility Scoring")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    _save(fig, out_path)


def plot_category_comparison(baseline_by_cat: dict, finetuned_by_cat: dict,
                             out_path: Path):
    cats = [c for c in baseline_by_cat if c in finetuned_by_cat]
    x = np.arange(len(cats))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [baseline_by_cat[c]["effect_size"] for c in cats],
           width, label="Baseline", color=COMPAT_COLOR)
    ax.bar(x + width / 2, [finetuned_by_cat[c]["effect_size"] for c in cats],
           width, label="Fine-tuned", color=INCOMPAT_COLOR)
    ax.axhline(0.5, color="gray", linestyle="--", label="Target d = 0.50")
    ax.set_xticks(x, cats, rotation=20)
    ax.set_title("Effect Size (Cohen's d) per Preference Category: "
                 "Baseline vs Fine-Tuned")
    ax.set_xlabel("Preference category")
    ax.set_ylabel("Cohen's d")
    ax.legend()
    _save(fig, out_path)


def plot_delta(baseline: dict, finetuned: dict, out_path: Path):
    deltas = [finetuned[m] - baseline[m] for m in CORE_METRICS]
    colors = [COMPAT_COLOR if d >= 0 else INCOMPAT_COLOR for d in deltas]
    # for FPR, a decrease is good — flip its color logic
    fpr_i = CORE_METRICS.index("false_positive_rate")
    colors[fpr_i] = COMPAT_COLOR if deltas[fpr_i] <= 0 else INCOMPAT_COLOR
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(CORE_METRICS, deltas, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Metric Improvement After Fine-Tuning "
                 "(Fine-Tuned minus Baseline)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Delta")
    _save(fig, out_path)


def plot_loss_curve(history: list[dict], out_path: Path):
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, losses, color=COMPAT_COLOR)
    ax.set_title("Training Loss per Step: CosineSimilarityLoss Convergence")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    if history and "epoch" in history[0]:
        max_epoch = int(max(h["epoch"] for h in history))
        for e in range(1, max_epoch + 1):
            boundary = next((h["step"] for h in history if h["epoch"] >= e), None)
            if boundary:
                ax.axvline(boundary, color="gray", linestyle=":", alpha=0.6)
    _save(fig, out_path)
```

- [ ] **Step 2: Write `scripts/make_baseline_figures.py`**

```python
"""Baseline-only figures (before/after versions regenerated in Task 12)."""
from pathlib import Path

import numpy as np

from src.visualize import plot_similarity_distributions, plot_umap

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"

data = np.load(ROOT / "reports" / "baseline_arrays.npz")
runs = [("Baseline", data["sims"], data["labels"])]
plot_similarity_distributions(runs, FIG / "similarity_distribution_baseline.png")
plot_umap([("Baseline", data["umap_xy"], data["labels"])],
          FIG / "umap_baseline.png")
print("Baseline figures written to", FIG)
```

- [ ] **Step 3: Run and inspect**

Run: `.venv/bin/python scripts/make_baseline_figures.py`
Expected: two PNGs; the histogram shows heavily overlapping distributions and the UMAP shows mixed colors (that's the point of the baseline).

- [ ] **Step 4: Commit**

```bash
git add src/visualize.py scripts/make_baseline_figures.py reports/figures/*.png
git commit -m "feat: visualization suite and baseline figures"
```

---

### Task 9: Fine-tuning pipeline

**Files:**
- Create: `src/train.py`

**Interfaces:**
- Consumes: `load_pairs`.
- Produces: `train_model(config: dict) -> tuple[str, list[dict]]` returning `(model_output_dir, loss_history)`. `config` keys: `run_id: str`, `epochs: int`, `batch_size: int`, `learning_rate: float`, `warmup_ratio: float`, `base_model: str` (default all-MiniLM-L6-v2), `loss: str` ("cosine" | "mnrl"), `max_pairs: int | None` (for smoke tests). Saves model to `models/<run_id>/`, loss history to `reports/loss_history_<run_id>.json`, config to `reports/train_config_<run_id>.json`.

- [ ] **Step 1: Implement `src/train.py`**

```python
"""Fine-tune a sentence transformer on dating pairs with contrastive loss."""
import json
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments, losses,
)

from src.data_loader import load_pairs

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "epochs": 4,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "loss": "cosine",
    "max_pairs": None,
}


def train_model(config: dict) -> tuple[str, list[dict]]:
    cfg = {**DEFAULTS, **config}
    run_id = cfg["run_id"]
    out_dir = ROOT / "models" / run_id

    pairs = load_pairs(ROOT / "data" / "dating_pairs.jsonl")
    if cfg["max_pairs"]:
        pairs = pairs[: cfg["max_pairs"]]

    if cfg["loss"] == "mnrl":
        # MNRL uses only positive pairs (in-batch negatives)
        positives = [p for p in pairs if p.label == 1]
        train_ds = Dataset.from_dict({
            "anchor": [p.text_1 for p in positives],
            "positive": [p.text_2 for p in positives],
        })
    else:
        train_ds = Dataset.from_dict({
            "sentence1": [p.text_1 for p in pairs],
            "sentence2": [p.text_2 for p in pairs],
            "score": [float(p.label) for p in pairs],
        })

    model = SentenceTransformer(cfg["base_model"])
    loss = (losses.MultipleNegativesRankingLoss(model) if cfg["loss"] == "mnrl"
            else losses.CosineSimilarityLoss(model))

    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=25,
        save_strategy="no",
        report_to=[],
        seed=42,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_ds, loss=loss,
    )
    trainer.train()
    model.save(str(out_dir))

    history = [
        {"step": h["step"], "epoch": h.get("epoch"), "loss": h["loss"]}
        for h in trainer.state.log_history if "loss" in h
    ]
    (ROOT / "reports" / f"loss_history_{run_id}.json").write_text(
        json.dumps(history, indent=2))
    (ROOT / "reports" / f"train_config_{run_id}.json").write_text(
        json.dumps(cfg, indent=2))
    return str(out_dir), history


if __name__ == "__main__":
    import sys
    train_model(json.loads(sys.argv[1]))
```

- [ ] **Step 2: Smoke-test with 64 pairs, 1 epoch**

Run:
```bash
.venv/bin/python -m src.train '{"run_id": "smoke", "epochs": 1, "max_pairs": 64}'
```
Expected: trains in under a minute, `models/smoke/` contains a saved model, `reports/loss_history_smoke.json` has ≥1 logged loss entry. Then clean up: `rm -rf models/smoke reports/loss_history_smoke.json reports/train_config_smoke.json`.

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: fine-tuning pipeline with CosineSimilarityLoss and loss logging"
```

---

### Task 10: Iteration 2 — first full training run + post-eval + comparison

**Files:**
- Create: `scripts/run_training.py`, `scripts/run_posteval.py`, `scripts/compare.py`
- Output: `reports/finetuned_metrics_<run_id>.json`, `reports/finetuned_arrays_<run_id>.npz`, `reports/comparison.json`
- Modify: `docs/ITERATION_LOG.md`

**Interfaces:**
- Consumes: `train_model` (Task 9), `evaluate_model` (Task 7).
- Produces: `scripts/compare.py` writes `reports/comparison.json` shaped `{"baseline": {...}, "finetuned": {...}, "delta": {...}, "targets_met": {...}, "run_id": "..."}` and prints a markdown table.

- [ ] **Step 1: Write `scripts/run_training.py`**

```python
"""Run a training iteration. Usage: run_training.py <run_id> [json-overrides]"""
import json
import sys

from src.train import train_model


def main():
    run_id = sys.argv[1]
    overrides = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    out_dir, history = train_model({"run_id": run_id, **overrides})
    print(f"Model saved to {out_dir}; {len(history)} loss points logged.")
    if len(history) >= 2 and history[-1]["loss"] >= history[0]["loss"]:
        print("WARNING: loss did not decrease — per spec, reduce lr to 1e-5 "
              "or increase batch size to 32 in the next iteration.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `scripts/run_posteval.py`**

```python
"""Evaluate a fine-tuned model on the eval set. Usage: run_posteval.py <run_id>"""
import json
import sys
from pathlib import Path

import numpy as np

from src.data_loader import load_pairs
from src.evaluate import evaluate_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    run_id = sys.argv[1]
    pairs = load_pairs(ROOT / "data" / "eval_pairs.jsonl")
    report, arrays = evaluate_model(str(ROOT / "models" / run_id), pairs)
    (ROOT / "reports" / f"finetuned_metrics_{run_id}.json").write_text(
        json.dumps(report, indent=2))
    np.savez(ROOT / "reports" / f"finetuned_arrays_{run_id}.npz", **arrays)
    print(json.dumps({k: v for k, v in report.items()
                      if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `scripts/compare.py`**

```python
"""Compare a fine-tuned run against the baseline. Usage: compare.py <run_id>"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "margin": ("gte", 0.05),
    "effect_size": ("gte", 0.50),
    "false_positive_rate": ("lte", 0.35),
    "cluster_purity": ("gte", 0.65),
}
METRICS = ["margin", "effect_size", "false_positive_rate", "cluster_purity",
           "accuracy", "precision", "recall", "f1", "auc_roc"]


def main():
    run_id = sys.argv[1]
    base = json.loads((ROOT / "reports" / "baseline_metrics.json").read_text())
    fine = json.loads(
        (ROOT / "reports" / f"finetuned_metrics_{run_id}.json").read_text())
    delta = {m: fine[m] - base[m] for m in METRICS}
    targets_met = {}
    for m, (op, t) in TARGETS.items():
        targets_met[m] = fine[m] >= t if op == "gte" else fine[m] <= t

    out = {"run_id": run_id, "baseline": {m: base[m] for m in METRICS},
           "finetuned": {m: fine[m] for m in METRICS},
           "delta": delta, "targets_met": targets_met}
    (ROOT / "reports" / "comparison.json").write_text(json.dumps(out, indent=2))

    print("| Metric | Baseline | Fine-Tuned | Delta | Target Met? |")
    print("|---|---|---|---|---|")
    for m in METRICS:
        met = ("✅" if targets_met[m] else "❌") if m in targets_met else "—"
        print(f"| {m} | {base[m]:.4f} | {fine[m]:.4f} | {delta[m]:+.4f} | {met} |")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run iteration 2 (spec starting config: 4 epochs, bs 16, lr 2e-5)**

```bash
.venv/bin/python scripts/run_training.py run1
.venv/bin/python scripts/run_posteval.py run1
.venv/bin/python scripts/compare.py run1
```
Expected: training takes tens of minutes on CPU (6,000 pairs × 4 epochs; Apple Silicon MPS will be used automatically by torch if available). Comparison table prints with deltas. Effect size should jump well above baseline (spec example: 0.015 → ~0.38+).

- [ ] **Step 5: Add iteration log entry 2**

Append to the table and detailed entries in `docs/ITERATION_LOG.md` (fill in measured numbers):

```markdown
| 2 | 4 epochs, lr=2e-5, batch=16, warmup 0.1, CosineSimilarityLoss | Spec default starting config | <baseline d> | <run1 d> | <delta> | <Yes/No + why> |

### Entry 2 — run1: spec default config
- Change: fine-tuned with 4 epochs, batch 16, lr 2e-5, warmup_ratio 0.1, CosineSimilarityLoss
- Reason: spec-recommended starting hyperparameters
- Metrics before: d=<baseline>, FPR=<baseline>, margin=<baseline>, purity=<baseline>
- Metrics after: d=<run1>, FPR=<run1>, margin=<run1>, purity=<run1>
- Delta: d <+x>, FPR <−x>, margin <+x>, purity <+x>
- Keep?: <Yes/No — e.g. "Yes, but d below 0.50 target; iterate">
```

- [ ] **Step 6: Commit**

```bash
git add scripts/run_training.py scripts/run_posteval.py scripts/compare.py reports/finetuned_metrics_run1.json reports/finetuned_arrays_run1.npz reports/comparison.json reports/loss_history_run1.json reports/train_config_run1.json docs/ITERATION_LOG.md
git commit -m "feat: iteration 2 - first fine-tuning run with default config"
```

---

### Task 11: Iterations 3+ — tune until targets met (≥ 3 total log entries)

**Files:**
- Modify: `docs/ITERATION_LOG.md`; new `reports/finetuned_metrics_<run_id>.json` etc. per run.

**Interfaces:**
- Consumes: Task 10 scripts unchanged — each iteration is `run_training.py runN '<overrides>'` → `run_posteval.py runN` → `compare.py runN` → log entry.

Follow the spec's decision tree. Run at least one more iteration even if run1 already meets all targets (the spec requires ≥ 3 log entries showing experimentation; entry 1 = baseline, entry 2 = run1, entry 3+ = these).

- [ ] **Step 1: Pick the next change based on run1's numbers**

Decision rules (from the spec, in priority order):
- Loss didn't decrease after epoch 1 → `{"learning_rate": 1e-5}` or `{"batch_size": 32}`
- Loss decreased but eval metrics worse than expected (overfitting) → `{"epochs": 2}` (or 3)
- Effect size < 0.50 → `{"epochs": 6}` first; if still short, try `{"loss": "mnrl"}` (note: MNRL trains on positives only with in-batch negatives — a legitimate spec-suggested alternative)
- All targets met → run a deliberate probe anyway (e.g. `{"epochs": 6}` to check for overfitting headroom, or `{"loss": "mnrl"}` to test the alternative loss) so the final config is validated against an alternative, not just the first thing that worked

- [ ] **Step 2: Run iteration 3**

```bash
.venv/bin/python scripts/run_training.py run2 '{"epochs": 6}'   # or chosen override
.venv/bin/python scripts/run_posteval.py run2
.venv/bin/python scripts/compare.py run2
```

- [ ] **Step 3: Add iteration log entry 3** (same format as entry 2 — record change, reason, before/after, delta, keep/revert decision)

- [ ] **Step 4: Repeat until every core target is met** (d ≥ 0.50, FPR ≤ 0.35, purity ≥ 0.65, margin ≥ 0.05) or until a config is demonstrably best. Each run gets its own `runN` id and log entry. If FPR stays > 0.35, inspect `by_pair_type` in the metrics JSON — the spec predicts dealbreaker pairs are the worst offenders — and record the analysis in the log entry.

- [ ] **Step 5: Declare the final configuration**

Add to `docs/ITERATION_LOG.md`:

```markdown
## Final configuration

**Chosen run:** run<N> (entry <M>)
**Config:** <epochs / batch / lr / warmup / loss>
**Justification:** entry <M> shows the best effect size (<d>) with FPR <fpr>
and no overfitting signal (loss curve flat by final epoch, eval metrics
improved alongside training loss). Entries <...> show the rejected
alternatives and why.
```

Re-run `compare.py <final_run>` so `reports/comparison.json` reflects the final model.

- [ ] **Step 6: Commit**

```bash
git add reports/ docs/ITERATION_LOG.md
git commit -m "feat: tuning iterations and final configuration selection"
```

---

### Task 12: Final visualization suite (all 6 required charts)

**Files:**
- Create: `scripts/make_final_figures.py`
- Output: `reports/figures/similarity_distribution_before_after.png`, `umap_before_after.png`, `roc_curves.png`, `category_effect_size.png`, `training_loss_curve.png`, `improvement_delta.png`

**Interfaces:**
- Consumes: all plotting functions from Task 8; `baseline_arrays.npz`, `finetuned_arrays_<final>.npz`, both metrics JSONs, `loss_history_<final>.json`, `comparison.json`.

- [ ] **Step 1: Write `scripts/make_final_figures.py`**

```python
"""Generate all six required before/after figures for the final model.

Usage: make_final_figures.py <final_run_id>
"""
import json
import sys
from pathlib import Path

import numpy as np

from src.visualize import (
    plot_category_comparison, plot_delta, plot_loss_curve, plot_roc,
    plot_similarity_distributions, plot_umap,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"


def main():
    run_id = sys.argv[1]
    base = np.load(ROOT / "reports" / "baseline_arrays.npz")
    fine = np.load(ROOT / "reports" / f"finetuned_arrays_{run_id}.npz")
    base_m = json.loads((ROOT / "reports" / "baseline_metrics.json").read_text())
    fine_m = json.loads(
        (ROOT / "reports" / f"finetuned_metrics_{run_id}.json").read_text())
    history = json.loads(
        (ROOT / "reports" / f"loss_history_{run_id}.json").read_text())

    sim_runs = [("Baseline", base["sims"], base["labels"]),
                ("Fine-tuned", fine["sims"], fine["labels"])]
    plot_similarity_distributions(
        sim_runs, FIG / "similarity_distribution_before_after.png")
    plot_umap([("Baseline", base["umap_xy"], base["labels"]),
               ("Fine-tuned", fine["umap_xy"], fine["labels"])],
              FIG / "umap_before_after.png")
    plot_roc(sim_runs, FIG / "roc_curves.png")
    plot_category_comparison(base_m["by_base_category"],
                             fine_m["by_base_category"],
                             FIG / "category_effect_size.png")
    plot_loss_curve(history, FIG / "training_loss_curve.png")
    plot_delta(base_m, fine_m, FIG / "improvement_delta.png")
    print("Wrote 6 figures to", FIG)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and visually inspect every PNG**

Run: `.venv/bin/python scripts/make_final_figures.py <final_run_id>`
Checklist per the spec: descriptive titles ✓, labeled axes ✓, legends ✓, grid ✓, overlap visibly reduced in the before/after histogram, UMAP shows separated regions after fine-tuning, ROC fine-tuned curve dominates baseline, loss curve converges.

- [ ] **Step 3: Commit**

```bash
git add scripts/make_final_figures.py reports/figures/
git commit -m "feat: final before/after visualization suite (6 required charts)"
```

---

### Task 13: Submission documentation — README, self-evaluation, final checklist

**Files:**
- Create: `README.md`, `docs/SELF_EVALUATION.md`
- Modify: `docs/ITERATION_LOG.md` (final polish)

- [ ] **Step 1: Write `README.md`**

Structure (fill measured numbers from `reports/comparison.json`):

```markdown
# Mini Project 5 — Dating Compatibility Fine-Tuning

Fine-tunes `all-MiniLM-L6-v2` with CosineSimilarityLoss to distinguish
compatible from incompatible dating pairs, and proves the improvement with
statistical metrics.

## Results (baseline → fine-tuned)

| Metric | Baseline | Fine-Tuned | Delta | Target | Met? |
|---|---|---|---|---|---|
| Effect size (Cohen's d) | <x> | <x> | <+x> | ≥ 0.50 | ✅/❌ |
| Margin | <x> | <x> | <+x> | ≥ 0.05 | |
| False positive rate | <x> | <x> | <−x> | ≤ 0.35 | |
| Cluster purity | <x> | <x> | <+x> | ≥ 0.65 | |
| Accuracy / Precision / Recall / F1 / AUC | ... | ... | ... | report | |

## Setup

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    # place dataset files in data/ then verify:
    .venv/bin/python scripts/check_data.py

## Pipeline (run in order)

    .venv/bin/python scripts/explore.py            # 1. dataset exploration
    .venv/bin/python scripts/run_quality_gate.py   # 2. quality gate (>= 60/100)
    .venv/bin/python scripts/run_baseline.py       # 3. baseline metrics
    .venv/bin/python scripts/make_baseline_figures.py
    .venv/bin/python scripts/run_training.py run1  # 4. fine-tune
    .venv/bin/python scripts/run_posteval.py run1  # 5. post-training metrics
    .venv/bin/python scripts/compare.py run1       # 6. side-by-side comparison
    .venv/bin/python scripts/make_final_figures.py run1  # 7. all charts

## Interpreting the results

- **Cohen's d** is the primary metric: <0.2 = no separation, 0.5 = medium,
  0.8 = large. The baseline (~<x>) means the pre-trained model cannot tell
  compatible from incompatible; the fine-tuned <x> means the two similarity
  distributions are clearly separated.
- **FPR** at threshold 0.5 = how often incompatible people would be wrongly
  matched. **Cluster purity** = how cleanly pairs separate in UMAP space.
- Category breakdown and failure analysis: `reports/finetuned_metrics_<run>.json`
  (`by_base_category`, `by_pair_type`); dealbreaker pairs are the hardest.

## Deliverables map

| Requirement | Location |
|---|---|
| Data quality evaluator | `src/quality_evaluator.py`, `reports/data_quality_report.json` |
| Baseline evaluation | `scripts/run_baseline.py`, `reports/baseline_metrics.json` |
| Fine-tuning pipeline | `src/train.py`, `scripts/run_training.py` |
| Post-training evaluation | `scripts/run_posteval.py`, `scripts/compare.py`, `reports/comparison.json` |
| Visualization suite | `reports/figures/*.png` |
| Metrics reports (JSON) | `reports/*.json` |
| Iteration log | `docs/ITERATION_LOG.md` |
| Self-evaluation | `docs/SELF_EVALUATION.md` |
```

- [ ] **Step 2: Write `docs/SELF_EVALUATION.md`**

Answer the spec's six self-evaluation questions **with measured data**, not intuition:

```markdown
# Self-Evaluation

1. **Does the pre-trained model actually struggle?**
   Baseline Cohen's d = <x> (< 0.3 threshold): yes/no + one sentence.
2. **Is the improvement real or overfitting?**
   All reported metrics computed on held-out eval_pairs.jsonl (never trained
   on); training loss curve (reports/figures/training_loss_curve.png) shows
   <convergence description>; eval d improved <x> → <x>.
3. **Where does the model still fail?**
   Highest-FPR pair types from by_pair_type: <list with numbers>.
4. **Did you validate the data first?**
   Quality gate scores: <five dimensions + overall>; weakest dimension was
   <x> because <sub-score detail>.
5. **Would a human agree with the labels?**
   Manual spot-check of 20 random eval pairs (seed 42): <n>/20 labels
   reasonable; disagreements: <describe>.
6. **Does the UMAP visualization show meaningful structure?**
   <describe before/after separation, reference umap_before_after.png>.
```

For question 5, actually run the spot check:

```bash
.venv/bin/python - <<'EOF'
import random
from src.data_loader import load_pairs
pairs = load_pairs("data/eval_pairs.jsonl")
random.seed(42)
for p in random.sample(pairs, 20):
    print(f"[{p.label}] ({p.pair_type}/{p.category})\n  A: {p.text_1}\n  B: {p.text_2}\n")
EOF
```

Read each pair and judge whether the label is reasonable; record the count and any disagreements.

- [ ] **Step 3: Run the full test suite one last time**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests pass.

- [ ] **Step 4: Verify the spec's final success criteria**

Walk the spec's "Final Success Criteria" checklist section by section (data/quality gate, baseline, fine-tuning, post-training metrics, visualizations, iteration logs, documentation) and confirm each box against an artifact in the repo. Fix anything missing before the final commit.

- [ ] **Step 5: Final commit**

```bash
git add README.md docs/SELF_EVALUATION.md docs/ITERATION_LOG.md
git commit -m "docs: README, self-evaluation, and final iteration log"
```

---

## Self-Review Notes

- **Spec coverage:** exploration (T3), quality evaluator + gate (T6), baseline + saved JSON + baseline UMAP (T7–T8), CosineSimilarityLoss fine-tuning with documented hyperparameters and loss curve (T9–T10, T12), post-training eval with side-by-side deltas and category/pair-type breakdown (T10), all 6 required visualizations (T8, T12), ≥3 iteration log entries with final-config traceability (T7 entry 1, T10 entry 2, T11 entries 3+), classification metrics at threshold 0.5 (T4, in every report), README + self-evaluation answers (T13). Optional enhancements (instructional fine-tuning, cross-domain robustness, TF-IDF, Plotly dashboards) are deliberately out of scope — YAGNI; add only if targets are met early.
- **Known judgment calls:** cluster purity is computed via HDBSCAN on a UMAP projection of symmetric pair features `[|u−v|; u·v]` with noise treated as one cluster — the spec doesn't pin down the exact recipe, so the same procedure is applied identically to baseline and fine-tuned models, which is what makes the comparison valid. Spec's stated baseline numbers are inconsistent between sections (~0.35 vs ~0.52 purity); we trust our own measured baseline.
- **Blocking risk:** the dataset is NOT yet in `data/` — Task 1 Step 7 is a hard stop until the user provides the four files.

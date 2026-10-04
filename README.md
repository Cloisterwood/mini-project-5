# Mini Project 5: Dating Compatibility

Staged fine-tuning coursework for `sentence-transformers/all-MiniLM-L6-v2` with
`CosineSimilarityLoss`. Stage 1 provides a reproducible Python environment and
environment diagnostics. Data processing and training are implemented in later stages.

## Setup

Run these commands in Terminal or the VS Code integrated terminal from the project
root. Prerequisites: Python 3.12 (verified locally: 3.12.13) and uv >=0.10.5.
`uv.lock` pins the complete runtime and development dependency graph, including
package hashes. `--locked` refuses to update an out-of-date lock.

```bash
uv sync --locked --python 3.12.13 --cache-dir .cache/uv
.venv/bin/python -m dating_compat doctor --output reports/environment.json
.venv/bin/python -m pytest -q
```

Expected doctor marker: `ENVIRONMENT PASS` and exit status 0. A missing dependency,
unsupported API, unavailable requested device, or failed tensor operation produces
`ENVIRONMENT FAIL` and exit status 1. No model download or API inference occurs.
The report records versions, interpreter, disk space, device availability, training
API signatures, and a float32 matrix multiplication with backward gradients.

```bash
.venv/bin/python -m dating_compat doctor --device cpu
.venv/bin/python -m dating_compat doctor --device mps
```

Automatic device selection prefers CUDA, then MPS, then CPU. Explicit unavailable
devices fail instead of falling back. A successful tensor check does not establish
that the full Trainer workload is supported or fast enough; that is measured in S5.

## VS Code interpreter

Open this folder as the workspace. `.vscode/settings.json` sets the default to
`${workspaceFolder}/.venv/bin/python`. If VS Code has a previously cached selection,
run **Python: Select Interpreter**, choose **Enter interpreter path**, and select:

```text
/Users/jeanneduke/newline-class-projects/mini-project-5/.venv/bin/python
```

`uv sync` installs `dating_compat` in editable mode from `src/`; edits take effect
without reinstalling. Use `.venv/bin/python` for every project command.

## Implementation choices

Python is constrained to 3.12 for this lock. The sentence-transformers 5.x and
transformers 4.x API family is constrained in `pyproject.toml`; exact installed
versions are pinned in `uv.lock` and reported by doctor. Training will use
`SentenceTransformerTrainer` and `CosineSimilarityLoss`, as described in the
[official training documentation](https://sbert.net/docs/sentence_transformer/training_overview.html).
Clustering uses the separate **hdbscan** package (`hdbscan.hdbscan_.HDBSCAN`),
not the scikit-learn implementation. No implicit implementation switch is allowed.
Required figures use Matplotlib/Seaborn PNGs; Plotly is not required.

Verified locally October 4, 2026: macOS 27.0, ARM64 Mac14,2, 24 GiB RAM,
Python 3.12.13, torch 2.14.1, sentence-transformers 5.7.0, transformers 4.57.6,
and hdbscan 0.8.44. CPU and MPS tensor/autograd checks passed; CUDA is unavailable.
Four diagnostic tests passed. The Codex sandbox reported MPS unavailable; the
approved outside-sandbox check verified MPS and saved `reports/environment.json`.
Run doctor in the normal Terminal environment when checking actual Mac hardware.

## Stages and resources

Start with [START_HERE.md](START_HERE.md) and the [current handoff](docs/implementation/HANDOFF.md).
Every stage ends with a checkpoint and stop; context usage at 75% also triggers a stop.

- [Stage plan](docs/implementation/01-STAGES.md)
- [Frozen experiment contract](docs/implementation/02-EXPERIMENT-CONTRACT.md)
- [Dataset source, revision, license declaration, acquisition paths, and issues](docs/implementation/RESOURCES.md)
- [Linear project](https://linear.app/cloisterwood/project/mini-project-5-dating-compatibility-1a9f2484672e)

Raw data, caches, credentials, and model weights are excluded from Git. No dataset
has been validated, no training experiment has run, and no performance targets are
claimed in Stage 1. Paid compute decisions wait for measured need in Stage 5.

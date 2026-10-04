# Mini Project 5: Dating Compatibility

Staged fine-tuning coursework for `sentence-transformers/all-MiniLM-L6-v2` with
`CosineSimilarityLoss`. Stage 1 provides a reproducible Python environment and
environment diagnostics. Stage 2 adds data processing; training awaits later stages.

Stage 2 implements acquisition, strict schema/metadata validation, data exploration,
and a seeded profile-component split. The approved quarantine excludes 9 training
rows that overlap evaluation, leaving **5,991 eligible rows**. Original files and
all 1,469 evaluation rows are preserved. See [the data audit](docs/DATA_AUDIT.md).
The structural data gate passes; the S3 human quality gate is still required.

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

## Acquire and audit data

Run from the project root with the locked environment. Acquisition uses network
access; exploration is offline. `acquire` downloads the four required files and
the dataset card at revision `69b750dd024d5732ca19596d86bffc253abee92e`, verifies
the recorded original SHA-256 checksums and Apache-2.0 card declaration, and writes
`reports/acquisition_manifest.json`. Existing source files are never overwritten.

```bash
.venv/bin/python -m dating_compat acquire
.venv/bin/python -m dating_compat explore
```

Expected markers: `ACQUISITION PASS` and `DATA EXPLORATION PASS`, both exit **0**.
Explore defaults to the user-approved `configs/quarantine-v1.json` policy and
writes active reports under `reports/quarantine-v1/`: `cohort_manifest.json`,
`exploration.json`, and `split_manifest.json`. Original blocked reports remain
at the report root as historical evidence.

The active manifest contains exactly the 5,991 eligible IDs in a 4,792 fit / 1,199
validation split, seed 42. The nine quarantined IDs are excluded from both partitions.
Normalized profile connected components prevent repeated people
and reversed pairs from crossing these partitions. Stratification uses label and
pair_type after reporting rare category strata; grouping is retained throughout.
Raw profile text is preserved for model input; NFKC/casefold/whitespace normalization
is only for audits and grouping. Labels are strict integers 0/1; pair_type is metadata.

Reports are frozen: identical reruns are allowed, different contents fail instead
of replacing earlier evidence. Use a new report directory for a documented protocol
revision. The approved nine-row quarantine is applied only to eligibility; original
data bytes and the entire evaluation set remain unchanged. Initial reports
under `reports/s2-initial-audit/` are preserved coding-repair evidence and must not
be used for training. Quality/training must use `load_eligible_bundle()` from
`dating_compat.cohort`. It reloads original data, verifies acquisition and approval
hashes, recomputes the exact exclusion set, verifies both active manifests, and
calls `assert_data_ready`. Training must also pass the separate S3 human gate.
There are still 519 normalized individual profiles shared with evaluation; pair
quarantine does not establish independence between people or writing templates.

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

Raw data, caches, credentials, and model weights are excluded from Git. Schema,
counts, and metadata have been validated; the approved derived cohort passes the
structural leakage gate.
No training experiment has run and no performance targets are claimed. Paid compute
decisions wait for measured need in Stage 5.

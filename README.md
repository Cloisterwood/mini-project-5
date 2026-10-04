# Mini Project 5: Dating Compatibility

Staged fine-tuning coursework for `sentence-transformers/all-MiniLM-L6-v2` with
`CosineSimilarityLoss`. Stage 1 provides a reproducible Python environment and
environment diagnostics. Stage 2 adds data processing; Stage 6 training is in progress.

Stage 2 implements acquisition, strict schema/metadata validation, data exploration,
and a seeded profile-component split. The approved quarantine excludes 9 training
rows that overlap evaluation, leaving **5,991 eligible rows**. Original files and
all 1,469 evaluation rows are preserved. See [the data audit](docs/DATA_AUDIT.md).
The structural data gate and the S3 human quality gate pass.

Stages 3–5 are complete. All 20 human reviews are recorded and the verified
quality gate passes at **94.60/100**. The pinned MiniLM baseline is measured and
sealed. CPU training, checkpoint reload, and exact resume passed a bounded smoke
test. Stage 6 is complete: three training experiments, with **run-002 selected** by
the frozen validation rule. Its validation metrics are d=2.2064, FPR=21.04%,
AUC=0.9165, and coverage-adjusted purity=83.40%.
All three runs meet all four primary validation targets.
Final held-out comparison awaits Stage 7; selection is already frozen.

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

## Quality gate and human review

```bash
.venv/bin/python -m dating_compat quality
```

This uses only the approved eligible cohort and the frozen
`configs/quality-rubric.yaml`. It writes `reports/quality_report.json`, preserving
individual components, evidence, numerators/denominators, undefined reasons,
mapping rules, both score scales, and source/cohort/split/rubric/review hashes.
The rubric hash is frozen in `reports/quality_rubric_manifest.json` before scoring.

The completed review currently returns `QUALITY GATE PASS` with exit **0**.
Scores are data quality 19.33/20, diversity 19.48/20, bias diagnostics 17.04/20,
linguistic quality 20.00/20, and real-life matching 18.75/20; overall 94.60/100.
Human label agreement is 15 acceptable, 3 ambiguous, and 2 unacceptable; all other
human fields are acceptable on all 20 pairs. Original judgments and notes are
preserved. A high heuristic score does not prove absence of bias or real-world
relationship validity. Missing human evidence returns pending with exit **1**.

Run this in the VS Code terminal from this project folder for guided entry:

```bash
.venv/bin/python -m dating_compat review
```

The program shows each pair and explains the six ratings. Enter `1` for acceptable,
`2` for ambiguous, or `3` for unacceptable. It requires notes for ambiguous or
unacceptable judgments and your name/alias. Type `yes` at the pair summary to
confirm your human judgments and save that pair. Enter `q` at any prompt to pause;
rerunning resumes at the first unfinished pair. Each confirmed pair is saved to
`docs/human_review_responses.csv`. To revisit a saved pair, use
`.venv/bin/python -m dating_compat review --pair 7` (choose its sample number).
Existing answers are shown and Enter keeps them. Concurrent CSV edits cause a
stop without overwriting those edits. The program does not generate ratings or
change the selected pairs. Run `quality` afterward to update the quality report.

You can also read [the 20-pair human review sheet](docs/HUMAN_REVIEW.md) and edit
`docs/human_review_responses.csv` directly. Each row requires six explicit
acceptable/ambiguous/unacceptable ratings, a reviewer name/alias, and
`human_reviewed=yes`. Add notes for every ambiguous or unacceptable judgment.
The completed ratings are saved. The seeded sample covers all nine pair types,
uses fit IDs only, and is diagnostic rather than a population estimate.

Rerun quality after entering the ratings. Passing requires every dimension >=12/20,
overall >=60/100, complete human evidence, unchanged valid source data, and
`QUALITY GATE PASS` / exit 0. If it fails, keep the frozen thresholds and seek
instructor/data-owner guidance. Future training must call `require_quality_gate()`
from dating_compat.quality; it revalidates structural provenance, verifies the
sample, recomputes scores from actual responses, and rejects stale or modified
quality reports. Changing a JSON passed flag cannot bypass the gate.

## Baseline metrics and sealed evidence

Run from the project folder:

```bash
.venv/bin/python -m dating_compat baseline
```

The completed run returns `BASELINE SEALED ARTIFACTS VERIFIED` and `BASELINE PASS`
without repeating inference. Changed source, quality, configuration, or sealed
artifacts cause a failing exit. Removing an artifact does not authorize replacing it.
`configs/baseline.json` is frozen by `reports/baseline_config_manifest.json` before
outcomes. Model revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` was verified
with the Hugging Face API. Eleven downloaded files have checksums in
`reports/base_model_manifest.json`; weights stay ignored under `models/base-minilm/`.

Inference uses CPU, four Torch threads, float32, batch 32, max sequence length 256,
raw profile texts with gender prefixes preserved, no instruction prefix, and unit
embeddings. No profile was truncated (maximum observed token lengths 36/33/35 for
fit/validation/evaluation). Embeddings are cached under `.cache/embeddings/`, bound
to ordered record IDs, raw text hash, model files, preprocessing, device, and versions.

| Split | Pairs | Margin | Signed Cohen's d | FPR at .5 | AUC | Coverage-adjusted purity |
|---|---:|---:|---:|---:|---:|---:|
| Fit | 4,792 | -0.08437 | -0.39050 | 0.49562 | 0.39924 | 0.45785 |
| Validation | 1,199 | -0.08771 | -0.40483 | 0.50751 | 0.40162 | 0.49625 |
| Evaluation | 1,469 | -0.07875 | -0.36405 | 0.48026 | 0.40076 | 0.51804 |

The base semantic-similarity model separates these compatibility labels poorly.
The frozen scoring direction/threshold remains unchanged. None of the baseline
final-checklist performance targets are met; baseline measurement is complete.
This does not predict whether fine-tuning will attain the targets.

All classification metrics, confusion counts, denominators, explicit undefined
reasons, exact category/pair-type breakdowns, and both target tiers are in
`reports/{fit,validation}_baseline_metrics.json` and
[`reports/baseline_metrics.json`](reports/baseline_metrics.json) (evaluation).
Row-aligned prediction CSVs and nine inspected similarity/UMAP/ROC PNGs are under
[`reports/baseline/`](reports/baseline/). The hash seal for all 17 baseline artifacts
is `reports/baseline_manifest.json`.

Clustering uses symmetric `[abs(u-v), u*v]` features with separately fitted,
label-free UMAP/HDBSCAN per split. Purity gives noise zero credit and uses all
pairs as denominator. Clustered-only purity, noise coverage, cluster sizes, and
observed-label majority baselines are also reported. Direct-feature sensitivity
was excluded before outcomes as an optional analysis. Separate UMAP axes are not
aligned. The evaluation baseline is sealed and must never guide model selection,
thresholds, early stopping, or targeted augmentation; development uses validation.
Bootstrap uncertainty and paired fine-tuned comparisons belong to Stage 7.

Implementation follows the installed APIs and official
[Sentence Transformers](https://sbert.net/docs/package_reference/sentence_transformer/model.html),
[UMAP reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html),
and [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/api.html) documentation.

## Training smoke test and compute decision

Run the separate governance preflight, then the isolated training worker:

```bash
.venv/bin/python -m dating_compat prepare-training
.venv/bin/python -m dating_compat train-smoke --device cpu
```

The preflight freshly verifies data eligibility, real human quality evidence, and
the sealed baseline, exporting only fit/validation rows to ignored
`data/training-ready/`. Its certificate is `reports/training_preflight_manifest.json`.
The worker validates the certificate, current permitted evidence and package
hashes. It denies Python file opens of evaluation sources and baseline evaluation
artifacts; evaluation examples/labels never enter its datasets. Trainer receives
only `text_1`, `text_2`, and float32 labels. Run the preflight before each experiment.

`configs/train-default.yaml` fixes four epochs, batch 16, learning rate 2e-5, warmup
ratio 0.1, seed 42, float32, max sequence length 256, AdamW and CosineSimilarityLoss. Raw texts, gender
prefixes, model revision and inference scoring match baseline. Full runs monitor
validation loss and reload the lowest-validation-loss checkpoint. Resume requires
the same frozen run configuration, IDs, source certificate, code and software.

The completed smoke used 256 fit pairs and 64 validation pairs, a fixed 16-step
schedule paused at 8, resumed to 16, and an uninterrupted 16-step reference. Weights
changed, losses were finite, and checkpoint reload reproduced embeddings with
maximum absolute difference 0. Resumed final weights and all 16 per-step training
losses matched the uninterrupted run exactly. Checkpoints include model,
optimizer (101 parameter states), scheduler and Python/NumPy/CPU RNG states.
This verifies CPU resume; other devices have not been benchmarked or verified.

[`reports/compute_benchmark.json`](reports/compute_benchmark.json) records actual
measurements: median 0.28487 seconds/step, mean 0.30800 seconds/step over 14 steps
after two warmup steps; about 51.95 pairs/second. Full validation loss evaluation
took 8.2427 seconds. Extrapolating 1,200 fit steps plus four validation passes gives
**374.8 seconds (6.25 minutes)**, with a **749.6-second (12.49-minute)** planning
allowance. Checkpoint I/O and initialization add overhead; this is an estimate,
not an actual completed four-epoch run. Local CPU is the selected environment.

Checkpoints (about 1 GiB) stay ignored under `checkpoints/s5-smoke/`; hashes and
resume evidence are tracked in the benchmark. Run provenance and stepwise losses
are in `reports/s5-smoke/`. Repeating `train-smoke` verifies saved checkpoint
hashes and prints `TRAINING SMOKE PASS` without training again. The smoke is
explicitly **not a scored experiment** and does not count toward the three runs.

Stage 6's implemented interface is `train --run-id run-001 --device cpu`; it uses
full fit and validation inputs. An existing run requires explicit
`--resume checkpoints/run-001/checkpoint-N` and cannot be overwritten by a new
run. After training, `validate-run --run-id run-001` compares the selected checkpoint
with the sealed validation baseline using the same IDs, preprocessing, threshold,
UMAP and HDBSCAN configuration. Repeated validation verifies saved hashes without
new inference. See `reports/runs/run-001/` and [the iteration log](docs/ITERATION_LOG.md).
All three runs are complete. Run-002 uses `--config configs/train-run-002.yaml`
for six epochs; its validation report and `comparison_run_001.json` record small
gains in d/FPR/AUC/purity and slight declines in accuracy/recall/F1. Final selection
is frozen in `reports/selected_run_manifest.json`: run-002, then run-001, then
run-003. Run-003 lowered LR to1e-5 for six epochs and had lower d/higher FPR.
`select-run` verifies all three saved validation seals and freezes the decision;
repeat verifies the same evidence without inference. Stage 7 remains pending
user continuation. No further training or retraining on all data is scheduled.

The implementation uses [Sentence Transformer training arguments](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html)
and [Trainer checkpoint resume](https://sbert.net/docs/package_reference/sentence_transformer/trainer.html),
checked against the installed APIs and actual saved states.

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
Human quality scoring passes and the measured baseline misses performance targets.
The local CPU training smoke passed; no scored experiment has run. The measured
local estimate supports proceeding on CPU when Stage 6 is authorized.

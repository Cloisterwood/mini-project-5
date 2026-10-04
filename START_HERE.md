# Mini-project 5: staged implementation

Planning date: October 3, 2026. Status: Stages 0–3 complete; Stage 4 awaits authorization.

Build a reproducible dating compatibility experiment using `sentence-transformers/all-MiniLM-L6-v2` and `CosineSimilarityLoss`. OpenAI models assist with development; the assignment explicitly specifies the embedding model being trained.

## Start in VS Code

Open `/Users/jeanneduke/newline-class-projects/mini-project-5` as the folder. Paste this into Codex chat:

> Read START_HERE.md, docs/implementation/01-STAGES.md, docs/implementation/02-EXPERIMENT-CONTRACT.md, docs/implementation/03-OPERATING-RULES.md, and docs/implementation/HANDOFF.md. Execute only the next incomplete stage. Inspect the workspace before changes. Stop at stage completion or 75% context utilization, whichever comes first. Save a handoff, update Linear, and commit/push reviewed project files to the confirmed GitHub repository. If an integration is unavailable, record pending synchronization honestly. Ask only for information, access, or human decisions that block this stage. Do not start the next stage until I say continue.

The pipeline commands in the stage plan are **interfaces to implement**, not commands that work yet. Terminal commands run in the VS Code integrated terminal; prompts run in Codex chat. Use `.venv/bin/python` locally and select that same interpreter in VS Code.

## Read in this order

1. [Stage plan](docs/implementation/01-STAGES.md): ten bounded build stages, checks, deliverables, and stops.
2. [Experiment contract](docs/implementation/02-EXPERIMENT-CONTRACT.md): metric definitions, assignment conflicts, data quality, and leakage controls.
3. [Operating rules](docs/implementation/03-OPERATING-RULES.md): context checkpoints, models, retries, Linear, GitHub, and compute.
4. [Current handoff](docs/implementation/HANDOFF.md): the next action and unresolved setup details.
5. [Original assignment](docs/implementation/assignment-original.txt): preserved verbatim.

The existing `docs/superpowers/plans/2026-07-04-dating-compatibility-finetuning.md` is preserved as historical material. This new plan supersedes its execution instructions, including repeated evaluation-set tuning, automatic model/device assumptions, and unbounded retries. Do not execute both plans.

## Confirmed resources and remaining setup

See [RESOURCES.md](docs/implementation/RESOURCES.md) for the private Cloisterwood GitHub repository, Linear project and ten issues, and the verified Hugging Face dataset revision and file names. No additional source links are needed.

Stages 0–3 are complete. All 20 actual human reviews are recorded; the frozen quality gate and fresh recomputation guard pass at 94.60/100, with all five dimensions above 12/20. Say “Continue with Stage 4” to authorize metrics and baseline; no model download or training has started. The approved quarantine remains active: 5,991 eligible rows (4,792 fit / 1,199 validation), with original sources and all evaluation rows preserved. The locked Python 3.12.13 environment passes doctor on CPU and MPS. The user confirmed Sol 6.1 is available in VS Code; use it as the default development assistant. GPU spending decisions are deferred until measured need in Stage 5.

The pinned original dataset is downloaded into ignored data/ and preserved unchanged. Environment setup, diagnostics, acquisition, and exploration commands work; see README.md. No model weights or training code are present. The historical July plan remains preserved.

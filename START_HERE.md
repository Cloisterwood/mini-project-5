# Mini-project 5: staged implementation

Planning date: October 3, 2026. Status: Stages 0–5 complete; Stage 6 run-001 complete (1/3).

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

Stages 0–5 are complete. The quality gate passes at 94.60/100 and the pinned baseline is sealed. Stage 6 run-001 completed four epochs on local CPU in 274.03 seconds. On the frozen 1,199 validation pairs, d=2.1841, FPR=0.2170, margin=0.5792 and coverage-adjusted purity=0.8315 meet all four primary targets. These are development validation results; final evaluation awaits S7 after all three runs and frozen selection. Say “Continue Stage 6 with run-002” to authorize only the second experiment. The approved quarantine remains 5,991 eligible rows (4,792 fit / 1,199 validation), with original/evaluation sources preserved. Sol 6.1 remains the user's preferred development assistant. Training and validation workers consume certified inputs and deny evaluation file reads.

Original data, isolated training inputs, model weights, embedding caches and smoke checkpoints stay ignored locally; checksums are in reports. Environment, diagnostics, acquisition, exploration, quality, guided review, baseline, training preflight, smoke and run interfaces work; see README.md. Smoke weights are not a submitted/scored model. The historical July plan remains preserved.

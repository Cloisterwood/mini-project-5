# Mini-project 5: staged implementation

Planning date: October 3, 2026. Status: plan ready; implementation has not started.

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

Stage 0 is in progress. The next VS Code session should confirm its available Codex model choice and Linear access, then finish the Stage 0 checkpoint. The user prefers Sol; use the actual model offered in the picker and record it. GPU spending decisions are deferred until a measured need in Stage 5.

No dataset files have been downloaded and no training code has been implemented. The historical July plan remains preserved.

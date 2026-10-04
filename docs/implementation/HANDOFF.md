# Current handoff

Updated: 2026-10-04. Status: STAGE 1 COMPLETE — STAGE 2 NOT STARTED.

## Next single action

Read the shared operating rules, experiment contract, S2 in `01-STAGES.md`, and
`RESOURCES.md`. Download the four original dataset files from the pinned revision
into ignored `data/`, then hash and validate them. Execute only S2 / [CLO-44](https://linear.app/cloisterwood/issue/CLO-44/mp5-s2-dataset-validation-exploration-and-split).
Do not start quality scoring, model download, or training.

Resume prompt: “Continue with Stage 2. Read the handoff and shared rules, acquire
and validate the pinned dataset, implement exploration and leakage-safe splitting,
verify the loader/split tests and reports, save Linear/GitHub evidence, then stop.”

## Stage 1 checkpoint

- Stage/issue: S1 / [CLO-43](https://linear.app/cloisterwood/issue/CLO-43/mp5-s1-reproducible-environment-and-package-skeleton).
- Stop reason: stage complete. Context indicator unavailable; exact 75% boundary cannot be measured.
- Assistant: GPT-6 family per session instructions; exact picker model/reasoning not exposed. Sol 6.1 remains the user's saved preference.
- Host: macOS 27.0, arm64, Mac14,2, 24 GiB RAM (sysctl verified), approximately 154 GiB free at diagnostics.
- Interpreter: `/Users/jeanneduke/newline-class-projects/mini-project-5/.venv/bin/python`, Python 3.12.13 from the existing pyenv installation. uv 0.10.5.
- Selected device: MPS verified outside sandbox; CPU verified inside sandbox; CUDA unavailable. Float32 matmul and autograd passed. Full Trainer smoke testing and throughput await S5.
- Environment: `pyproject.toml` API-family constraints and complete `uv.lock` with hashes. `uv sync --locked --python 3.12.13 --cache-dir .cache/uv` installed 78 packages, exit 0. Separate hdbscan implementation `hdbscan.hdbscan_.HDBSCAN`, version 0.8.44; no implicit switch to sklearn.
- Core versions: torch 2.14.1, sentence-transformers 5.7.0, transformers 4.57.6, datasets 4.8.5, accelerate 1.15.0, numpy 2.5.3. All required imports succeeded. Doctor checks installed Trainer/loss signature parameters without downloading a model.
- VS Code workspace default interpreter set in `.vscode/settings.json`. The GUI's cached active selection was not inspected; README includes the exact selection path if it overrides the default.
- Package skeleton: src/dating_compat, configs, tests, reports, models. Model README tracked, binaries ignored. No training or data commands implemented yet.

## Checks and artifacts

- Initial Git status/diff clean; previous verified checkpoint `7323f3959d1956cdaf614a2ce7c953e4ade5ef4d` on main.
- `.venv/bin/python -m dating_compat doctor --output reports/environment.json`: ENVIRONMENT PASS, exit 0, CPU selected in sandbox. The initial font-cache build emitted permission warnings and took extra time.
- Approved `.venv/bin/python -m dating_compat doctor --device mps --output reports/environment.json`: ENVIRONMENT PASS, exit 0. Final saved report is the actual MPS check.
- `.venv/bin/python -m dating_compat doctor --device cpu --output reports/environment-cpu.json`: ENVIRONMENT PASS, exit 0; CPU report saved separately.
- `.venv/bin/python -m pytest -q`: exit 0, 4 passed. Tests cover accelerator selection, explicit unavailable-device failure, missing-dependency failure, and CLI nonzero failure status.
- `uv pip check --python .venv/bin/python --cache-dir .cache/uv`: exit 0, all installed packages compatible.
- `.venv/bin/python -c 'import dating_compat; print(dating_compat.__file__)'`: exit 0, points to this project's src/dating_compat/__init__.py (editable install).
- `git diff --check`: exit 0. Ignore checks confirmed .venv, cache, and future model weights excluded.
- Lock/report hashes and final verified publishing SHA will be recorded on CLO-43; this commit cannot contain its own SHA. Final checkpoint is the latest main commit in the confirmed private repository https://github.com/Cloisterwood/mini-project-5.
- Temporary blockers: sandbox DNS denied PyPI; approved resolution/install succeeded. Sandbox hardware query denied; approved sysctl succeeded. Sandbox MPS unavailable; approved real-device check succeeded. No unresolved code/dependency error.
- No dataset, model weights, training process, GPU rental, API inference, or subscription change. Install and diagnostic processes completed before checkpoint.
- Linear: CLO-43 moved In Progress at start; completion and verified publishing receipts are written at closeout. No pending sync expected after successful receipts.

## Decisions preserved

- Train all-MiniLM-L6-v2 with CosineSimilarityLoss using SentenceTransformerTrainer.
- Frozen experiment contract in `02-EXPERIMENT-CONTRACT.md`; report working and stricter targets separately.
- Three actual training configurations; fixed training-derived validation; final held-out comparison after selection.
- Stop at every stage, every S6 experiment, or 75% context use, whichever comes first.
- Plus and Mac first; paid compute/subscription only after measured need and user decision.
- Historical July plan preserved; the staged implementation plan governs execution.

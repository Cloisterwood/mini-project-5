# Build stages

Every stage ends with the checkpoint procedure in `03-OPERATING-RULES.md`, then STOP. An earlier 75% context stop leaves that stage In Progress. Resume it before advancing. Each stage is one Linear issue; Stage 6 contains separate stops for each training experiment.

## Stage 0 — access, destinations, and decisions

Model: Sol. Dependencies: none.

- Confirm dataset source, Linear team/project, GitHub destination and visibility, and available Codex model choices.
- Inspect existing files and any applicable AGENTS.md. Preserve the July plan.
- Connect Linear in VS Code through its supported MCP integration; user handles browser OAuth. Create/reuse a project named `Mini-project 5 — Dating compatibility`; create issues MP5-S0 through MP5-S9 using the titles here and copy their acceptance checks. These are local planning keys, not invented Linear issue IDs. Record real IDs/URLs in the handoff.
- Initialize a standalone Git repository here after verifying the target. Add ignores before staging. Publish the plan to the confirmed repository; record its URL and pushed commit.
- Freeze the experiment contract; flag grading ambiguities for instructor clarification when available. Proceed with the documented default interpretation if clarification is unavailable.

Acceptance: real Linear issue links and verified GitHub plan commit recorded; dataset acquisition path known. If auth/destination is missing, checkpoint as Blocked and list the exact human action. No fabricated integration success.

**STOP S0.**

## Stage 1 — reproducible environment and package skeleton

Model: Sol; Luna may handle straightforward documentation. Dependency: S0.

- Inspect Python and Mac hardware. Choose a supported Python 3.10+ version (prefer 3.11 or 3.12 after dependency compatibility checks), create project-local `.venv`, and set the VS Code interpreter explicitly.
- Create `pyproject.toml`, pinned dependency lock, `.gitignore`, `README.md`, `configs/`, `src/dating_compat/`, `tests/`, `reports/`, and `models/`.
- Dependencies: PyTorch, sentence-transformers, datasets/accelerate as required by Trainer, Pydantic, NumPy, Pandas, SciPy, scikit-learn, UMAP, HDBSCAN, Matplotlib, Seaborn, pytest. Plotly is optional; PNG output is mandatory.
- Use `SentenceTransformerTrainer` with `CosineSimilarityLoss`; confirm installed APIs. Pick one HDBSCAN implementation and record its version; do not silently switch implementations.
- Implement `python -m dating_compat doctor`: interpreter, dependency imports, selected device, CPU/MPS/CUDA availability, disk space, and package versions. Verify a small real tensor operation on the selected device.

Acceptance: `.venv/bin/python -m dating_compat doctor` prints `ENVIRONMENT PASS`; editable package imports work from the documented working directory. Record the exact environment, not a claim that all tests pass when none exist.

**STOP S1.**

## Stage 2 — dataset validation, exploration, and split

Model: Sol. Dependency: S1 and the four original dataset files.

- Implement `data.py`, strict Pydantic schema, stable record IDs, file hashes, and JSONL errors with filename/line. Reject blank required text, invalid labels (including boolean/coerced strings), malformed JSON, and missing metadata files.
- Record observed counts versus expected 6,000/1,469; compare metadata to actual data. Investigate discrepancies rather than silently dropping rows.
- Explore label/category/subcategory/pair-type distributions, vocabulary and length statistics, prefix order, exact/reversed/normalized duplicate pairs, contradictory labels, and train/eval overlap. `label` is the target; `pair_type` never determines it.
- Create reproducible fit/validation manifests from training data only, targeting 80/20, seed 42, stratified where feasible. Keep duplicate/reversed pairs in the same partition; report shared profiles/templates and consider grouped splitting. Never silently abandon grouping when rare strata make it difficult.
- Write immutable split IDs and hash manifest. Only profile text enters model input; metadata stays in analysis.

Acceptance: loader/split tests catch malformed records, reversed overlap, contradictory duplicates, and leakage; `python -m dating_compat explore` writes `reports/exploration.json` and `reports/split_manifest.json`. Missing data or cross-file pair leakage blocks training until resolved with provenance.

**STOP S2.**

## Stage 3 — quality gate and human review

Model: Sol for scoring design; Terra for bounded implementation once contract is fixed. Dependency: S2.

- Implement `quality.py` and the five-dimension rubric in the experiment contract. Save each subscore, evidence, denominator, and scoring rule.
- Prepare a seeded 20-pair training-only review sheet with diverse pair types where feasible. User rates label agreement, naturalness, and realism; record ambiguous judgments and sample selection. An LLM opinion cannot stand in for completed human review.
- Audit gender-prefix/label association, category label imbalance, text-length correlation, and vocabulary shortcuts. Provide diagnostics; do not call them proof of absence of bias.
- Save `reports/quality_report.json` and `docs/HUMAN_REVIEW.md`. Implement a failing exit status if the gate is not met; training must check the report against current dataset hashes.

Acceptance: tests show missing review or failed gate prevents training. Overall score is at least 60/100 and all five dimensions at least 12/20 under the conservative working rubric. If not, document findings, request instructor/data-owner guidance, and stop; never weaken scoring to pass.

**STOP S3.**

## Stage 4 — metrics and baseline

Model: Sol; use a stronger available OpenAI model for unresolved statistical design. Dependency: S3.

- Implement `metrics.py`, `evaluate.py`, and `cluster.py` from the experiment contract. Test hand-calculated examples and undefined cases before loading the real model.
- Load the exact base-model revision, cache embeddings by model/data/preprocessing hashes, and calculate fit/validation baseline metrics plus the assignment-required baseline on the supplied evaluation file.
- Seal evaluation baseline predictions/report against modification. Do not use final-evaluation examples or results to choose experiments. Development decisions use validation only.
- Save `reports/baseline_metrics.json` (evaluation), validation metrics separately, row-aligned predictions, cluster coverage, configuration, software versions, and baseline PNGs. Measure actual numbers; assignment examples are not expected outputs.

Acceptance: tests verify Cohen's d, FPR, threshold ties, AUC, noise/one-class/zero-variance cases, pair-order symmetry, and prediction IDs. Baseline reports are reproducible and labeled by split. Record a high baseline honestly.

**STOP S4.**

## Stage 5 — training smoke test and compute decision

Model: Sol. Dependency: S4.

- Implement `train.py`, `configs/train-default.yaml`, run manifests, training/validation loss logging, checkpoint save/resume, and explicit device selection.
- Initial config: 4 epochs, batch 16, learning rate 2e-5, warmup ratio 0.1, seed 42, float32 initially. Record max sequence length and truncation rates; freeze preprocessing across baseline and fine-tuning.
- Trainer receives two text columns and float labels 0/1 only. Training uses fit IDs; monitoring/checkpoint selection uses validation IDs; eval file is inaccessible to the training command.
- Run a small training-only smoke sample, verify finite loss, actual weight updates, checkpoint reload, inference, and resume state (optimizer/scheduler/RNG included where supported).
- Benchmark representative steps on the Mac and estimate a full run from measured throughput. If unsupported/too slow, present one Colab or RunPod alternative with current cost and a capped budget. User chooses paid provisioning.

Acceptance: checkpoint round-trip and split/gate protection tests pass; smoke result is explicitly not a scored experiment. Save `reports/compute_benchmark.json` and the selected execution environment.

**STOP S5.**

## Stage 6 — three bounded training experiments

Model: Sol for diagnosis; Terra for running an already-verified configuration. Dependency: S5.

Run three real training configurations, each from the same base revision and frozen fit/validation split. Baseline and smoke tests do not count toward the three experiments. Stop after each experiment, even if context is below 75%.

1. `run-001`: specified default configuration. Report validation metrics versus validation baseline. **STOP S6A.**
2. `run-002`: one evidence-based change; e.g., lower learning rate to 1e-5 if unstable, or fewer epochs if validation worsens while fit loss improves. Record rationale before launching. **STOP S6B.**
3. `run-003`: one further justified comparison, such as 6 epochs if underfitting or batch 32 if memory allows. Keep `CosineSimilarityLoss` for required experiments. **STOP S6C.**

Each run writes immutable `reports/runs/<run-id>/` config, provenance, loss history, validation predictions/metrics, duration, and checkpoint location/checksum. Append `docs/ITERATION_LOG.md` and structured `reports/iterations.jsonl`: change, reason, before/after, signed delta, keep/reject, seed, code commit, model/device, and failure status if applicable.

Select among successful runs by validation: satisfy core targets if possible, then highest d, lower FPR as tie-break. If none meets all targets, choose highest d among runs satisfying FPR <= .35; if none, explicitly report unmet constraints and choose the highest d as a diagnostic result. Require user agreement before extending the three-run budget. Do not promise target attainment.

Acceptance: three completed experiment entries, honest failures separately logged, selected configuration traceable to evidence, no test-based selection. Additional repetitions for seed robustness are optional and budgeted separately.

## Stage 7 — frozen final comparison and uncertainty

Model: Sol. Dependency: completed S6 and frozen run selection.

- Record selected checkpoint hash before opening evaluation results. Do not retrain on all 6,000 records by default; the validation-selected checkpoint remains final.
- Evaluate it once on `eval_pairs.jsonl` using identical baseline metrics/preprocessing and threshold .5. Write `reports/finetuned_metrics.json` and `reports/comparison.json` with deltas and both target tiers.
- Compute paired uncertainty estimates and category/pair-type breakdowns with support counts. Examine false positives, dealbreakers, subtle mismatches, and both-class support.
- Save examples of errors and explanations as post-selection analysis. Changes prompted by these results require a new untouched test set or disclosure that results are exploratory, not a fresh holdout claim.

Acceptance: both reports reference identical evaluation IDs/hashes; confidence intervals and assumptions recorded; achieved/unmet targets explicit; test results never silently trigger further tuning.

**STOP S7.**

## Stage 8 — all figures and interpretation

Model: Terra for plotting; Sol for conclusions. Dependency: S7.

- Implement `visualize.py` and output six PNGs: `similarity_distribution_before_after.png`, `umap_before_after.png`, `category_effect_size.png`, `roc_curves.png`, `training_loss_curve.png`, and `improvement_delta.png`.
- Use Matplotlib/Seaborn (Plotly optional), labeled axes, descriptive titles, colorblind-friendly palette, fonts >=10pt, and grids where appropriate. Show validation curves alongside training loss; training loss alone cannot establish lack of overfitting.
- Make UMAP points represent pairs, not individual profiles assigned conflicting labels. Disclose separate projections and their arbitrary coordinate systems. Do not infer significance from attractive clusters.
- Answer all six assignment self-evaluation questions in `docs/SELF_EVALUATION.md`, with measured evidence and human review results. State that synthetic labels do not establish real-world romantic compatibility.

Acceptance: inspect every PNG visually, verify labels/units/split/counts against JSON, and ensure missing subgroup metrics are marked unavailable rather than zero.

**STOP S8.**

## Stage 9 — reproducibility, GitHub delivery, and closeout

Model: Sol for final audit; Luna for small editorial fixes. Dependency: S8.

- Complete README with exact setup and CLI commands, dataset acquisition/license, interpreter selection, compute options, targets, measured results, limitations, and artifact retrieval instructions.
- Audit every checkbox in the original assignment against a real artifact in `docs/DELIVERABLE_CHECKLIST.md`; do not mark unmet performance targets complete.
- Run relevant full tests and clean-environment smoke validation. Verify saved-model inference and regenerate reports/figures from stored predictions without costly retraining.
- Save model artifacts in a durable approved location outside ordinary Git, with hashes and reproducible download/rebuild instructions. GitHub holds source/configs/locks/docs/small reports/PNGs; model weights and caches stay excluded unless Git LFS or a release artifact is deliberately chosen.
- Push the final reviewed commit, verify remote SHA matches, and record the GitHub URL. Update Linear with evidence, close only completed issues, and leave unmet criteria explicit.

Acceptance: another user can reproduce the pipeline from README, obtain data/model artifacts, identify all three experiments, and verify the submitted commit. No credentials, caches, or unintended files in the repository.

**STOP S9 — project delivered, or delivered with explicitly listed unmet criteria.**

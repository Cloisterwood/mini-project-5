# Current handoff

Updated: 2026-10-04. Status: S6B COMPLETE — runs001–002 complete; S6 In Progress (2/3).

## Next single action: S6C only

Wait for “Continue Stage 6 with run-003.” Run only the third experiment, then
freeze run selection using the S6 rule and stop. Do not start S7.
Read the stage plan/experiment contract/operating rules, freshly prepare-training,
verify smoke receipts. Proposed single change from run-002: reduce LR2e-5->1e-5
while retaining six epochs and all other values. Small gains from extra training
justify testing smaller updates as the final bounded alternative; no instability
was observed. Freeze rationale/config before launch from the same pinned base,
never initialize from previously trained weights. No extra run budget authorized.
After three successful runs: core targets first, highest validation d, lower FPR
tie-break; if unmet use documented fallback. Final evaluation/uncertainty in S7
only after selection is frozen and separately authorized.

## S6B execution and verification

- Pre-launch plan/config committed b2cbf4e53f74a1f7235b59379969cc5c3554c425;
  reports/run-002-plan.json and configs/train-run-002.yaml. Only config value
  changed: epochs4->6. Ratio warmup/linear decay extends too:180/1800 steps
  versus120/1200. This is a six-epoch schedule comparison, not an identical prefix.
- Fresh prepare-training and saved train-smoke verification passed, no smoke
  retraining. Original sources, human gate, baseline and run-001 unchanged.
- Actual run-002 from pinned base completed1800 steps on local CPU4threads,
  4792 fit/1199 validation, batch16/LR2e-5/warmup.1/seed42/float32/max256,
  CosineSimilarityLoss. No failed/interrupted training attempt or paid compute.
- Trainer runtime463.3596 seconds (7.72min), validation generation40.124987s.
  Epoch validation losses .1311007142, .1167688221, .1155654490, .1153148040,
  .1148024052, .1139603481. Lowest loss at epoch6/checkpoint-1800 selected.
- Selected model checkpoints/run-002/selected; state SHA
  8c79ccf853ccd4e24e64e55b637b68f2d7847c6208ca2911cc7e69ea9e2a59f4.
  Full model/optimizer/scheduler/RNG receipts in selected_checkpoint_manifest.json.
  Weights/checkpoints retained ignored locally, no remote model backup.
- Validation run-001 -> run-002: d2.1841410550 ->2.2064050189;
  FPR .2170283806 -> .2103505843; AUC .9136700056 -> .9164718976;
  margin .5792293519 -> .5767153208; accuracy .8315262719 -> .8306922435;
  precision .8024316109 -> .8058551618; recall .88 -> .8716666667;
  F1 .8394276630 -> .8374699760; coverage-adjusted purity .8315262719 -> .8340283570.
- All four primary validation targets pass. Earlier strict FPR and classification
  targets except AUC remain unmet. Small gains, no uncertainty/significance claim.
  Both meet primary targets; run-002 provisionally leads by higher d. No final
  selection before run-003; run-001 retained. Validation is development evidence.
- TN473/FP126/FN77/TP523.19 clusters,8 noise/1199; clustered-only purity .8396306;
  all-pair denominator1199, noise zero credit. Frozen threshold.5 and preprocessing,
  same label-free UMAP/HDBSCAN config. Independently fitted UMAP axes unaligned.
- No evaluation source/results opened by training/validation workers; no evaluation
  inference with selected model. Training code and dependencies/lock unchanged.
- Immutable reports/runs/run-002/ contains config/provenance/losses/predictions,
  metrics/breakdowns/target tiers, checkpoint receipt, four figures and nine-file
  validation seal. Separately sealed comparison_run_001.json records signed deltas.
  docs/ITERATION_LOG.md and reports/iterations.jsonl updated.
- validate-run passed; repeat verified nine artifact hashes without inference.
  Independent CSV audit reproduces metrics/purity and exact validation IDs/labels;
  manifests confirm same preflight/IDs/config except epochs and current source hashes.
  One audit initially resolved code basenames from the wrong directory; corrected
  to src/dating_compat and passed without product changes. Four plots visually
  checked; temporary plotting/font-cache warnings did not prevent output.
- Full pytest112 passed in14.23s. No product source edits or new tests this run.
- No active training process/paid resources. Context telemetry unavailable;
  sequential GPT-6 family, exact picker unexposed; user preference Sol6.1.
- Reviewed named config/reports/docs committed and pushed to private main;
  final delivery SHA and live remote verification on CLO-48; worktree clean.
  S6 stays In Progress until third run/selection complete. Commit cannot contain
  its own SHA. No run-003 started.

STOP: S6B complete. Next prompt: “Continue Stage 6 with run-003.”

## Historical S6A checkpoint (superseded next action)

Updated: 2026-10-04. Status: S6A COMPLETE — run-001 complete; S6 remains In Progress (1/3).

## Next single action: S6B only

Wait for “Continue Stage 6 with run-002.” Do not start run-003 or S7.
Read S6 and the experiment contract/operating rules. Freshly run prepare-training
and verify smoke receipts. Freeze one evidence-based config change and rationale
before training run-002 from the same pinned base, never from run-001 weights.
Proposed comparison: extend epochs from4 to6, all other defaults unchanged.
Rationale: validation loss improved each epoch (.132871, .118419, .116256,
.115483), with diminishing gains and no observed epoch loss reversal. Test
whether more training helps; this does not establish underfitting or promise gains.
Save run-002 validation comparison/iteration receipt, update tracking, STOP S6B.

## S6A actual execution and result

- Fresh prepare-training: pass; existing train-smoke --device cpu receipts verified.
- Default run-001: four epochs, batch16, LR2e-5, warmup.1, seed42, float32,
  max256, CosineSimilarityLoss, local CPU4threads; fit4792/validation1199.
- Training completed1200 steps, train_runtime274.0303 seconds (4.57min).
  Lowest epoch validation loss .11548302322626114 at checkpoint-1200.
  Selected inference weights state SHA:
  ed38a7f19c3c3d26541f4ab0571913429f94c0738acf7d3a52785a0a26aa3b2c.
- Training execution commit481864bf32a62bb89ca1680995b22991327bf81c;
  exact training source hashes and software in run_manifest.json. Train code
  unchanged during execution. New validation helper has its own code hash.
- validate-run --run-id run-001: VALIDATION RUN PASS;40.32 seconds for
  inference/clustering/report generation. Repeat verifies nine sealed artifacts.
- Same validation rows/order, raw text, preprocessing, normalization, threshold.5,
  seed42 label-free symmetric pair UMAP/HDBSCAN settings as baseline.
- Validation before -> after: margin -.0877096 -> .5792294;
  d -.4048329 ->2.1841411; FPR .5075125 -> .2170284;
  AUC .4016194 -> .9136700; accuracy .3753128 -> .8315263;
  coverage-adjusted purity .4962469 -> .8315263.
- Counts TN469/FP130/FN72/TP528. Twenty clusters;15/1199 noise rows,
  clustered-only purity .8420608; all-row denominator1199 gives noise no credit.
- All four primary checklist validation targets met. Earlier strict FPR<=.1
  and accuracy/precision/recall/F1>=.9 unmet; report both target tiers honestly.
- Keep as candidate; no final selection until all three planned runs. Validation
  was used for checkpoint selection; these are not final holdout estimates or
  significance claims. S7 paired uncertainty and final evaluation remain pending.
- No evaluation examples/results opened by training/validation workers. Existing
  evaluation baseline sealed; no evaluation inference for this selected model.
  Original sources, quarantine/split/human-review/baseline evidence unchanged.
- reports/runs/run-001/ contains immutable config, run/training provenance,
  full loss history, row-aligned predictions, metrics/breakdowns/target tiers,
  checkpoint file receipts and four PNGs. docs/ITERATION_LOG.md and
  reports/iterations.jsonl record signed deltas and provisional keep decision.
- Full pytest:112 passed in12.41s. Focused validation tests3 passed.
  Independent CSV audit reproduces metrics/purity and verifies labels/IDs/order.
  All four plots visually checked. A one-off audit initially used the wrong CSV
  column name 'cluster'; corrected to 'cluster_id' and passed without code changes.
  Plotting used a temporary font cache after sandbox cache warnings; plots saved.
- Model/optimizer/RNG/checkpoints retained locally under ignored checkpoints/run-001/;
  selected checkpoint is checkpoints/run-001/selected. GitHub stores hashes/reports,
  not model binaries; local machine currently holds the model, no remote backup.
- No training process or paid resource active. Dependencies/lock unchanged.
- Sequential execution; assistant GPT-6 family, exact picker/reasoning unexposed;
  user preference Sol6.1. Context telemetry unavailable; checkpoint at S6A stop.
- Named source/tests/reports/docs reviewed, committed/pushed to private main;
  final delivery SHA and live remote verification recorded on CLO-48. A commit
  cannot include its own SHA. S6 remains In Progress, one of three runs complete.

STOP: S6A complete. Next prompt: “Continue Stage 6 with run-002.”

## Historical S5 checkpoint (superseded next action)

Updated: 2026-10-04. Status: S5 COMPLETE — CPU SMOKE AND EXACT RESUME VERIFIED.

## Next single action

Wait for “Continue with Stage 6.” Execute ONLY S6A/run-001, then stop.
Read 01-STAGES.md S6, 02-EXPERIMENT-CONTRACT.md, and 03-OPERATING-RULES.md.
Run prepare-training to freshly validate governance, then train-smoke --device cpu
to verify existing evidence without retraining. Use the frozen default config for
run-001: four epochs, batch16, LR2e-5, warmup.1, seed42, float32, max256,
CosineSimilarityLoss, local CPU. Do not open evaluation results for development.

The full training interface is implemented:
.venv/bin/python -m dating_compat train --run-id run-001 --device cpu
It has NOT been executed. S6 must additionally compute selected-checkpoint
validation metrics/plots versus validation baseline, save row-aligned validation
predictions/config/checkpoint hashes, append iteration receipts, update Linear
and GitHub, then STOP S6A. Run-002 and run-003 need separate user continuation.

Resume prompt: “Continue with Stage 6. Execute only run-001 from the frozen base
and fit/validation split, save actual validation comparison and iteration evidence,
update Linear/GitHub, then stop after S6A.”

## Stage 5 completed implementation

- S5/CLO-47 Done. Added train.py, training_data.py, frozen train-default.yaml,
  prepare-training/train-smoke/train CLI interfaces and protection tests.
- Separate governance preparation reverified original sources, eligible grouped
  split, actual human quality gate and sealed baseline. It exported ONLY fit4792
  and validation1199 rows into ignored data/training-ready/.
- Training worker verifies certificate and current allowed evidence/package
  hashes without reading evaluation sources or baseline evaluation report/PNGs.
  An active Python file-open audit guard rejects evaluation source/artifact names,
  including resolved aliases. It is an application guard, not a separate OS account.
- Trainer receives only text_1/text_2 and float32 binary labels; IDs kept in
  provenance, metadata excluded. Eligibility, exact split IDs/order, exclusion
  IDs, evaluation IDs, disjointness and float/text schema checked before model.
- S3 quality remains94.59591920379897/100 from20 actual human reviews.
  Sealed S4 baseline unchanged. Training worker uses certified preflight evidence;
  full source revalidation is the separate governance step before each run.
- Defaults: four epochs, batch16, LR2e-5, warmup ratio.1, seed42, float32,
  max sequence length256, AdamW (non-fused adamw_torch), weight_decay0,
  max_grad_norm1, logging every10 steps, epoch validation/save, lowest validation
  loss checkpoint selection for full runs. Run selection uses S6's validation
  metric rule. Raw profiles/gender prefixes/no instructions match baseline.
- Explicit device argument. Actual training/round-trip/resume verified on CPU
  with four Torch threads; no claims of MPS/CUDA training or resume support.
- Full-run manifests freeze config/device/receipt/IDs/code/software. Existing run
  cannot be overwritten; explicit resume verifies same run and complete checkpoint.
  Models/checkpoints and optimizer state stay ignored under checkpoints/.

## Actual non-scored smoke

- Seed42 sample256 fit pairs,64 validation pairs, from their frozen allowed sets.
  Fixed total16-step schedule (two-step warmup), paused8 then resumed16.
  Reference starts from identical base for uninterrupted16 steps.
  This is32 actual optimizer steps across the two equivalent trajectories;
  it is NOT a scored experiment and counts toward none of the three runs.
- Finite per-step training and validation losses/gradients.
- Initial weights SHA:
  2c705a5e3279febcd4b210fc95a6231ced10be71e400e863ab41b623d5799508.
- Resumed AND uninterrupted final weights SHA:
  a951e7e580b2a6515e516150465ce3704749f26739ed2287596cee3f0d82b28a.
  Weights changed and final state hashes match exactly.
- Checkpoint8 model/inference reload: exact model state; maximum absolute
  embedding difference0 on16 fit-only probe profiles.
- Required resume files present: model.safetensors, optimizer.pt, scheduler.pt,
  rng_state.pth, trainer_state.json.101 optimizer parameter states;
  scheduler last_epoch equals8/16; Python/NumPy/CPU RNG keys present.
- Checkpoint receipts/hash lists for resumed8/resumed16/uninterrupted16 saved in
  reports/compute_benchmark.json. Smoke outputs about1GiB locally.
- Independent audit: smoke IDs belong to correct splits and are disjoint; all16
  resumed per-step training losses equal uninterrupted reference. Do not compare
  Hugging Face resumed aggregate train_loss directly with the full reference
  average; continuation aggregate divides by cumulative global steps.
- Loss history and exact run/code/software provenance in reports/s5-smoke/.
  Full validation LOSS forward pass used only to measure compute, not to score
  or select a submitted model. No evaluation examples or labels accessed by worker.

## Actual compute benchmark and selection

- CPU reference measured14 training steps after excluding first two warmup steps.
- Median step0.28486654144944623s; mean0.30800131242722273s;
  approximately51.94783059173042 pairs/s at batch16.
- Full1199-row validation loss pass runtime8.2427s.
- Four-epoch full-fit step count1200. Extrapolated compute374.8106497393355s
  (6.25min), including four validation passes; planning allowance749.621299478671s
  (12.49min). Actual longer-run throughput may vary; checkpoint/init/I/O overhead
  excluded from measured step time, covered approximately by2x planning allowance.
  This is NOT an observed completed full run.
- Selected environment local_cpu, paid_compute=false. Local measured training
  and exact resume are viable; no rental alternative or provisioning needed.
- No training process or paid resources active at this checkpoint.

## Validation and preserved evidence

- Initial focused training suite9 passed; initial full suite105 passed.
- Final full .venv/bin/python -m pytest -q: exit0,109 passed in11.55s
  (96 prior +13 training isolation/schema/checkpoint/gate cases).
- Real train-smoke --device cpu: exit0, TRAINING SMOKE PASS.
- Repeat same command: exit0, SMOKE SAVED CHECKPOINTS VERIFIED; no retraining.
- Repeated baseline: exit0, BASELINE SEALED ARTIFACTS VERIFIED; no inference.
- Tests reject absent/failed/stale preflight, changed human/source/package evidence,
  evaluation proof paths, integer/bool labels or extra metadata columns,
  missing checkpoint state and invalid run IDs. Worker explicitly denies
  evaluation-file opens; data sampling/model schema tests pass.
- Original6,000-row training and1,469-row evaluation and S2/S3/S4 artifacts
  unchanged. Approved9-row quarantine =>5,991 eligible; profile-grouped
  fit4792/validation1199, seed42. Residual profile/template overlap disclosed.
- Baseline validation d -0.4048328864/FPR .5075125209/AUC .4016193656.
  All baseline performance targets unmet. S5 supplies no scored improvements.
  Cosine is not calibrated probability; synthetic labels are not real romance.

## Provenance and saving

- reports/training_preflight_manifest.json SHA:
  4cba812338d9ee7b02269db1537322c07399e1c8e45563c47964ef2b103d5507.
- reports/train_config_manifest.json SHA:
  fdfc8e79983986b04dae9f0154bdfffbcc57ac1f60257c371dcea3ddfc9c2fa2.
- reports/compute_benchmark.json SHA:
  a82d25a1762853d66907e18fb3ac51e8b68aec0b2aa2ae246b8e9dbb970a36f3.
- reports/s5-smoke/loss_history.json SHA:
  7a25e8f86cf5fe94471e330d3aa26e6003118a72943d6035e8aa55d02f2056be.
- Local Mac14,2 arm64/macOS27.0/24GiB, project Python3.12.13 and locked environment
  unchanged; library versions in run manifest. No dependencies/lock changed.
- Starting and smoke execution commit44204fe99ec69b61a8b8abe735aa5365a2221892,
  plus exact uncommitted code SHA hashes in smoke manifest/benchmark.
  Final delivery commit and live remote SHA recorded on CLO-47 at closeout;
  a commit cannot contain its own SHA.
- Assistant GPT-6 family per instructions; picker/reasoning not exposed; user
  preference Sol6.1. Context telemetry unavailable; exact75% limit unmeasurable.
  Sequential work, no agents. All model training stayed local CPU.
- Branch main, private https://github.com/Cloisterwood/mini-project-5. Reviewed
  named source/config/small reports/docs/tests committed/pushed; remote verified
  and worktree clean at closeout. Data, cached embeddings, model weights and
  checkpoints excluded; local checkpoint receipts describe retention.
- Linear completion receipt recorded on CLO-47; S0–S5 Done.
  S6 not started. STOP: S5 complete.

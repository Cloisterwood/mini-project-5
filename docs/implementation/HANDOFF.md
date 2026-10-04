# Current handoff

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

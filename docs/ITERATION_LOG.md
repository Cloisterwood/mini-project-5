# Training iteration log

## run-003 and final selection — completed 2026-10-04 (S6C)

Single change from run-002: LR2e-5->1e-5, six epochs/all other values unchanged.
Pre-launch rationale/config committed1acaf14 (full SHA in run_manifest.json):
compare smaller updates after small gains from longer training, not a repair for
instability. Same pinned base and frozen fit4792/validation1199, CPU4threads,
seed42, batch16, float32, max256, CosineSimilarityLoss,180 warmup/1800 total steps.
Completed1800 steps in492.7345 Trainer seconds; validation generation57.471869s.
Failure/interruption status: none. Lowest validation loss .1173410416 at epoch6,
checkpoint-1800 selected. A small epoch5 reversal did not alter the planned budget.

| Validation metric | run-002 | run-003 | Signed delta |
|---|---:|---:|---:|
| Margin | 0.576715 | 0.579754 | +0.003039 |
| Cohen's d | 2.206405 | 2.147573 | -0.058832 |
| FPR | 0.210351 | 0.230384 | +0.020033 |
| AUC | 0.916472 | 0.909958 | -0.006514 |
| Accuracy | 0.830692 | 0.829858 | -0.000834 |
| Precision | 0.805855 | 0.794643 | -0.011212 |
| Recall | 0.871667 | 0.890000 | +0.018333 |
| F1 | 0.837470 | 0.839623 | +0.002153 |
| Coverage-adjusted purity | 0.834028 | 0.814012 | -0.020017 |

Reject run-003 for final selection: all four primary targets pass, but d is lower
than both prior runs. Higher recall/F1/margin do not override the frozen rule.
TN461/FP138/FN66/TP534;20 clusters,34 noise/1199, clustered-only purity .837768.
Earlier strict FPR and accuracy/precision/recall/F1 targets remain unmet.
Selected run-003 model state SHA
f052a9e9e6f59bdc7bbef0a970da2911f7377cbf5ec653cd266af3677a28d9ff;
weights and full training checkpoint retained locally under checkpoints/run-003/.
Hash receipts and immutable report/predictions/config/plots in reports/runs/run-003/;
signed comparison against run-002 in comparison_run_002.json, against baseline in
validation_metrics.json. No original source, baseline or prior run changed.

Final frozen ranking: run-002, run-001, run-003. All three meet all four primary
validation targets; choose highest d, then lower FPR tie-break. **Select run-002**:
six epochs, LR2e-5, best-loss checkpoint-1800; inference checkpoints/run-002/selected,
state SHA8c79ccf853ccd4e24e64e55b637b68f2d7847c6208ca2911cc7e69ea9e2a59f4.
Decision binds all three reports, training results, model receipts, cohort and
selected config in reports/selected_run_manifest.json; canonical manifest SHA
e428c305c13473d5c6863c9d1c987ac766c8633373d455dab2e4e9d7d05e60be.
No final evaluation opened for selection, no significance claim from one seed.
No retraining on all data: this checkpoint remains final for S7.

Verification: focused selection5 passed; full117 passed in21.94s. Independent CSV
audit reproduces metrics/purity and exact validation IDs/labels. Training config,
cohort, source and preflight hashes checked. Four plots visually reviewed.
select-run repeated: same frozen decision and all three validation seals verified
without inference. Model binaries ignored and retained locally, no remote backup.
No active training process/paid compute. STOP S6 complete; S7 pending continuation.

## run-002 — completed 2026-10-04 (S6B)

Single config change: epochs4->6 from the same pinned base, all other defaults
unchanged. Rationale recorded before launch in reports/run-002-plan.json and
configs/train-run-002.yaml, committed at b2cbf4e53f74a1f7235b59379969cc5c3554c425.
All four run-001 epoch validation losses improved with diminishing gains; test
additional training without claiming established underfitting. The ratio-based
warmup/linear decay schedule extends with the total steps:180 warmup/1800 total,
versus120/1200. This is not an identical four-epoch prefix plus two epochs.

Completed1800 steps in463.3596 Trainer seconds; validation artifact generation
40.124987 seconds. Lowest epoch validation loss .1139603481 selected epoch6,
checkpoint-1800. Failure/interruption status: none. Seed42, CPU4threads, float32,
batch16, LR2e-5, warmup.1, max256, CosineSimilarityLoss; fit4792/validation1199.

| Validation metric | run-001 | run-002 | Signed delta |
|---|---:|---:|---:|
| Margin | 0.579229 | 0.576715 | -0.002514 |
| Cohen's d | 2.184141 | 2.206405 | +0.022264 |
| FPR | 0.217028 | 0.210351 | -0.006678 |
| AUC | 0.913670 | 0.916472 | +0.002802 |
| Accuracy | 0.831526 | 0.830692 | -0.000834 |
| Precision | 0.802432 | 0.805855 | +0.003424 |
| Recall | 0.880000 | 0.871667 | -0.008333 |
| F1 | 0.839428 | 0.837470 | -0.001958 |
| Coverage-adjusted purity | 0.831526 | 0.834028 | +0.002502 |

Keep as provisional leader: both runs meet all four primary targets, and run-002
has higher d under the frozen selection rule. FPR is also lower. Gains are small;
accuracy, recall, F1 and margin slightly decline. Earlier strict FPR and
accuracy/precision/recall/F1 targets remain unmet. No significance claim or final
selection until run-003. TN473/FP126/FN77/TP523;19 clusters,8 noise rows,
clustered-only purity .839631. Baseline comparisons and exact deltas in
reports/runs/run-002/validation_metrics.json; run-001 comparison separately sealed
in comparison_run_001.json. These are development validation results only.

Selected model checkpoints/run-002/selected, weights state SHA
8c79ccf853ccd4e24e64e55b637b68f2d7847c6208ca2911cc7e69ea9e2a59f4.
Optimizer, scheduler and RNG checkpoint hashes verified. Local weights retained
and ignored; GitHub saves reports/hash receipts, not model binaries. Original
sources, baseline, run-001 and frozen split unchanged. No evaluation inference.

Verification: full112 tests passed in14.23s; independent CSV audit reproduces
metrics/purity and validates IDs/labels/order. Frozen config/source/preflight/IDs
verified. An audit initially resolved source basenames from the wrong directory;
corrected to src/dating_compat and passed without product changes. Nine validation
artifact hashes verified on repeat; four plots visually checked. Temporary font
cache warnings did not prevent plot output. No dependencies or source changed.

Proposed run-003, pending continuation: keep six epochs and reduce LR2e-5->1e-5,
all other values unchanged. Given small gains from extending training, compare
smaller parameter updates as the final bounded alternative; this is not a fix for
observed instability. Record rationale/config before launch from the same base.
Then freeze selection among all three by the specified rule and STOP S6C.

STOP S6B; run-003 and S7 not started.

## run-001 — completed 2026-10-04 (S6A)

First specified default comparison: four epochs, batch16, learning rate2e-5,
warmup.1, seed42, float32, max sequence256, CosineSimilarityLoss, local CPU4threads.
Starts from pinned base MiniLM revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41,
using frozen fit4792/validation1199. No change to defaults; establish the required
first trained comparison. Completed1200 steps in274.0303 Trainer seconds.
Validation artifact generation took40.3194 seconds. Failure status: none.

| Validation metric | Baseline | run-001 | Signed delta |
|---|---:|---:|---:|
| Margin | -0.087710 | 0.579229 | +0.666939 |
| Cohen's d | -0.404833 | 2.184141 | +2.588974 |
| FPR | 0.507513 | 0.217028 | -0.290484 |
| AUC | 0.401619 | 0.913670 | +0.512051 |
| Accuracy | 0.375313 | 0.831526 | +0.456214 |
| Precision | 0.337691 | 0.802432 | +0.464741 |
| Recall | 0.258333 | 0.880000 | +0.621667 |
| F1 | 0.292729 | 0.839428 | +0.546699 |
| Coverage-adjusted purity | 0.496247 | 0.831526 | +0.335279 |

Keep as candidate: all four primary validation targets pass. Earlier strict FPR
and accuracy/precision/recall/F1 targets remain unmet. Twenty clusters,15 noise
rows,1,184 clustered rows; clustered-only purity .842061. Same threshold.5 and
frozen label-free clustering config; UMAP maps have independent, unaligned axes.
Validation is development evidence, not final evaluation. No final selection or
uncertainty claim until remaining runs and S7. No evaluation inference occurred.

Best validation loss .1154830232 selected checkpoint-1200 (epoch4), with optimizer,
scheduler and RNG receipts verified. Model saved locally at
checkpoints/run-001/selected; state SHA
ed38a7f19c3c3d26541f4ab0571913429f94c0738acf7d3a52785a0a26aa3b2c.
Training code commit481864bf32a62bb89ca1680995b22991327bf81c; exact source hashes,
config/software, predictions and figures in reports/runs/run-001/.
Model binaries stay ignored locally; GitHub receives provenance/report files.

Verification:112 tests pass; independent prediction audit reproduces all aggregate
metrics and cluster purity and checks exact validation ID/label alignment.
Repeat validate-run verifies all nine saved artifact hashes without inference.
Four figures visually checked. No failed/interrupted training attempts.

Proposed next comparison, pending continuation: six epochs with everything else
unchanged. All four epoch validation losses improved with diminishing gains;
test whether further training helps without claiming underfitting. Freeze config
and rationale before run-002. STOP S6A; no run-002 started.

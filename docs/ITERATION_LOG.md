# Training iteration log

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

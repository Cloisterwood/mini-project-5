# Final deliverable audit

Audited against every checkbox under the original assignment's Final Success
Criteria. Checked means supported by an artifact, not blanket scientific success.
The preserved original assignment remains authoritative; actual baseline d differs
from its illustrative ~.015. Final result uses run-002 selected before evaluation.

## Data and quality gate

- [x] Provided training/evaluation data loaded: acquisition_manifest.json and original SHA receipts;6000/1469 source rows.
- [x] Pair-type/category/label exploration: reports/quarantine-v1/exploration.json and docs/DATA_AUDIT.md.
- [x] All five quality dimensions: src/dating_compat/quality.py and reports/quality_report.json.
- [x] Overall>=60 before training:94.595919/100, all dimensions>=12/20; preflight/report hashes in each run.
- [x] Never trained on evaluation rows: nine pair overlaps quarantined, fit4792/validation1199, isolated certified packages and file-open guards. Residual shared profiles/templates disclosed; statistical independence not claimed.

## Baseline

- [x] Four core baseline metrics on pinned MiniLM: evaluate.py/metrics.py/cluster.py and reports/baseline_metrics.json.
- [x] Saved baseline JSON: reports/baseline_metrics.json and sealed baseline_manifest.json.
- [x] Baseline UMAP saved: reports/baseline/baseline_eval_umap.png and submission before/after map.

## Fine-tuning

- [x] Actual CosineSimilarityLoss pipeline trained: train.py, three completed training_result.json reports; smoke not counted.
- [ ] Literal claim of convergence **without overfitting**: fit/validation loss PNG is complete and improves, but cannot establish absence of overfitting; overlap/one-seed limitations remain. Treat this wording as partially satisfied, not proven.
- [x] Hyperparameters documented: three YAML configs/run manifests. Run001:4epochs/batch16/LR2e-5/120 warmup of1200steps; run002:6epochs/LR2e-5/180 of1800; run003:6epochs/LR1e-5/180 of1800; seed42/float32/max256.

## Post-training metrics

- [x] Same four core metrics: final_evaluation.py and reports/finetuned_metrics.json, identical1469 evaluation IDs/source hash.
- [x] d>=.50 achieved: actual baseline-.364048 -> final2.158427; delta2.522475 with paired95% interval[2.303166,2.738972]. The assignment's baseline~.015 is illustrative, not substituted for observed baseline.
- [x] FPR<=.35 achieved:.227632 (173/760).
- [x] Coverage-adjusted purity>=.65 achieved:.833901, noise given zero credit.
- [x] Margin>=.05 achieved:.572008.
- [x] Accuracy/precision/recall/F1/AUC reported:.833220/.786420/.898449/.838710/.911777.
- [x] Category-wise metrics: exact source categories and pair types with class supports/undefined reasons in both metric JSONs.
- [x] Before/after table and every metric delta: docs/FINAL_COMPARISON.md and reports/comparison.json.

## Visualizations

- [x] Before/after cosine histograms: figures/submission/similarity_distribution_before_after.png.
- [x] Before/after pair UMAP: figures/submission/umap_before_after.png; unaligned axes explicitly disclosed.
- [x] Category grouped bar chart: figures/submission/category_effect_size.png; unavailable dealbreaker d is not zero. Stage8's earlier dot chart preserved at figures/.
- [x] Overlaid ROC/AUC: figures/submission/roc_curves.png.
- [x] Training loss curve: figures/submission/training_loss_curve.png, raw window losses, mean logged-window loss per epoch, and full validation loss. Epoch summary is not a fresh full-fit evaluation.
- [x] Delta bar chart/zero line: figures/submission/improvement_delta.png; intervals only for the four bootstrapped metrics.
- [x] Six PNGs with clear titles/axes/fonts>=10/colorblind palette/grids: all visually inspected; hashes in figures/submission/manifest.json. Regenerated PNG bytes match in fresh environment.

## Iteration logs and traceability

- [x] Every change/reason/before/after/signed delta: docs/ITERATION_LOG.md and reports/iterations.jsonl.
- [x] At least three experimentation entries: exactly run001/run002/run003, all actual completed training runs, no failed or smoke attempts counted.
- [x] Final decisions traceable: selected_run_manifest.json ranks002,001,003 by frozen validation rule and binds checkpoint/config/report hashes before evaluation.
- [x] Structured change/reason/metrics/delta records: reports/iterations.jsonl.

## Documentation

- [x] README explains complete command sequence, frozen artifact reuse, data acquisition/license declarations, interpreter/device/compute choices and model retrieval.
- [x] JSON automated reports: baseline_metrics.json, finetuned_metrics.json, comparison.json, quality_report.json and supporting seals.
- [x] Before/after metric tables: docs/FINAL_COMPARISON.md and docs/ITERATION_LOG.md.
- [x] Data-grounded explanation of improvement: docs/SELF_EVALUATION.md answers all six questions. Compatible/incompatible score direction changes and paired uncertainty support dataset improvement; no causal or real-world compatibility claim.

## Other required deliverables and explicit exceptions

- [x] Required stack: locked Python3.12, sentence-transformers/MiniLM/CosineSimilarityLoss, Pydantic, NumPy/Pandas/SciPy, Matplotlib, UMAP, external HDBSCAN.
- [x] Quality/baseline/train/post-training/visualization implementations, JSON reports, iteration log and README supplied.
- [x] Five dimension targets met at both score scales; primary final four targets met.
- [x] Stricter margin>=.10, d>=.50, purity>=.70, AUC>=.90 met.
- [ ] Stricter FPR<=.10: actual.227632.
- [ ] Stricter accuracy>=.90: actual.833220.
- [ ] Stricter precision>=.90: actual.786420.
- [ ] Stricter recall>=.90: actual.898449.
- [ ] Stricter F1>=.90: actual.838710.
- [ ] Literal uniform random20 human sample: completed review was seeded stratified diagnostic;15 acceptable/3 ambiguous/2 unacceptable. This departure is disclosed, not silently relabeled.
- [x] Saved selected inference model outside ordinary Git: private GitHub release v0.1.0, archive/file/state SHA receipts in model_release_manifest.json. Optimizer/resume files remain local; release contains inference files and model-card fit examples, not the raw dataset.
- [x] Fresh locked environment smoke,121 tests, restored model inference, three reports and six figures reproduced: reports/reproducibility/. Same host/cached wheels; no cross-platform equivalence or repeat-training guarantee claimed.

Optional instruction prefixes, extra training data, alternate losses, dashboards,
cross-domain evaluation and seed repetitions were not implemented. No final outcome
triggered tuning. Submission is complete with the unchecked criteria above explicit.

# Stage 7 frozen final comparison

Run-002 was selected on validation before any trained-model evaluation outcomes.
Its exact six-epoch/LR2e-5 checkpoint remains final; no refit or tuning followed.
Evaluation uses the same1,469 original IDs/labels, raw preprocessing and threshold.5
as the sealed baseline (709 positive,760 negative). Reports bind identical source
hashes and ordered IDs. These synthetic labels do not establish real relationship
compatibility, and residual cross-split profile/template overlap remains disclosed.

| Evaluation metric | Baseline | Final run-002 | Signed delta |
|---|---:|---:|---:|
| Margin | -0.078752 | 0.572008 | +0.650760 |
| Cohen's d | -0.364048 | 2.158427 | +2.522475 |
| False positive rate | 0.480263 | 0.227632 | -0.252632 |
| AUC | 0.400757 | 0.911777 | +0.511020 |
| Accuracy | 0.405037 | 0.833220 | +0.428182 |
| Precision | 0.353982 | 0.786420 | +0.432437 |
| Recall | 0.282087 | 0.898449 | +0.616361 |
| F1 | 0.313972 | 0.838710 | +0.524738 |
| Coverage-adjusted purity | 0.518039 | 0.833901 | +0.315861 |

All four primary checklist targets pass: margin>=.05, d>=.5, FPR<=.35,
purity>=.65. Earlier strict FPR<=.1 and accuracy/precision/recall/F1>=.9 remain
unmet; strict margin/d/AUC/purity pass. Confusion counts TN587/FP173/FN72/TP637.
Twenty clusters,13 noise rows,1,456 clustered pairs; clustered-only purity .841346.
Noise gets zero credit in the primary all-pair denominator. UMAP axes are unaligned
between independently fitted baseline/final maps; no label inputs to clustering.

## Paired uncertainty

Protocol frozen at commit6f7b7d2486870b100e202f1a5182b4f831730682, before final
inference. Two thousand seeded42 bootstrap draws, same component sample for both
models, percentile95% intervals. Evaluation has1,396 normalized profile connected
components (1,332 size1,55 size2,9 size3); largest3 pairs (0.204%). This meets the
predeclared minimum30 components/maximum25% dominance rule. Resample components
uniformly with replacement and include all pairs within each selected component.
All2,000 resamples defined for all four reported metrics/deltas.

| Metric | Final 95% interval | Delta 95% interval |
|---|---|---|
| Margin | [0.543768, 0.598405] | [0.610211, 0.687241] |
| Cohen's d | [1.991090, 2.334103] | [2.303166, 2.738972] |
| FPR | [0.199468, 0.259013] | [-0.306186, -0.193543] |
| AUC | [0.896504, 0.925144] | [0.475343, 0.543752] |

Delta d excludes zero and supports improvement for this synthetic evaluation
under the stated resampling assumptions. Component grouping handles repeated exact
normalized profiles within this set; shared templates and train/evaluation profile
overlap still limit independence/generalization. Intervals omit training-seed and
run-selection uncertainty. No purity interval: fixed cluster labels were not used
as a substitute for refitting UMAP/HDBSCAN in every bootstrap sample.

## Post-selection error analysis

| Source pair_type | Positive | Negative | FP | FN |
|---|---:|---:|---:|---:|
| compatible | 307 | 8 | 8 | 12 |
| complex_compatible | 92 | 0 | 0 | 0 |
| complex_incompatible | 0 | 58 | 0 | 0 |
| dealbreaker | 0 | 215 | 0 | 0 |
| incompatible | 0 | 210 | 2 | 0 |
| llm_judged_compatible | 151 | 0 | 0 | 56 |
| llm_judged_incompatible | 0 | 148 | 48 | 0 |
| subtle_mismatch | 159 | 121 | 115 | 4 |

Subtle mismatches account for115/173 total false positives:115/121 negatives in
that group cross.5 (95.04%). For example, source line2 pairs enthusiasm for belief
systems with entrepreneurship (cosine .705820), and line27 pairs financial
independence with close family bonds (.656265). The source labels these incompatible;
the model's scores do not discriminate these labeled differences. These excerpts
support a descriptive weakness, not a causal explanation or proof of incompatibility.

Dealbreakers show0/215 false positives; this subgroup has only negative labels,
so recall/AUC/d are undefined, not perfect general-purpose performance. Source
pair_type metadata is distinct from labels: eight negatives carry type compatible
and all eight are false positives. Some example pairs share similar dislikes,
highlighting ambiguity between textual similarity and source compatibility labels.
The llm_judged groups contribute48 false positives and56 false negatives. Category
and pair-type reports include class support and explicit undefined metrics.

reports/final_error_analysis.json saves source-order excerpts (up to three per
type/error) with IDs, true labels, both scores and threshold explanations. This
is a descriptive convenience sample, not an exhaustive or random qualitative review.
No changes follow these outcomes. Any future tuning needs a new untouched test set
or explicit exploratory disclosure.

## Verification and artifacts

Full121 tests passed in13.52s. Independent CSV audit reproduces metrics, all
breakdowns and cluster purity; IDs/labels and both comparison source hashes align.
The final command ran inference once; repeat verifies four sealed artifact hashes
without encoding, clustering or bootstrap. Actual evaluation/report runtime27.6341s.
No training/paid resources active. Model stays locally retained; GitHub stores hashes.

reports/finetuned_metrics.json, reports/comparison.json,
reports/final_error_analysis.json and reports/final/finetuned_eval_predictions.csv
are bound by reports/final_evaluation_manifest.json. Selected file/state receipts
were written to reports/final_pre_evaluation_manifest.json before evaluation reads.
Stage8 figures/interpretation remain pending. STOP S7 complete.

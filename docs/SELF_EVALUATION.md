# Assignment self-evaluation

The six questions below reproduce the topics in the original assignment's Step6.
Answers use frozen final evaluation, the validation-selected run-002, and the
actual pre-training human review. Cosine is not calibrated probability; synthetic
labels do not establish real-world romantic compatibility.

## 1. Does the pre-trained model actually struggle?

Yes on this labeled task. On1,469 evaluation pairs, baseline signed Cohen's d is
-.364048, margin -.078752, AUC .400757, FPR .480263 and accuracy .405037. The
compatible class has lower mean similarity than the incompatible class. This
is below the assignment's d>.3 concern about a trivially separable baseline.
Do not reverse the score after observing results: the fixed positive direction
and threshold.5 remain unchanged. Text similarity does not directly encode these
compatibility labels. See reports/baseline_metrics.json and the similarity/ROC PNGs.

## 2. Is the improvement real or overfitting?

Improvement appears on the supplied held-out evaluation, not just the fit set:
final d2.158427, margin .572008, AUC .911777, FPR .227632, accuracy .833220.
Validation-selected run-002 had d2.206405/FPR .210351/AUC .916472 on1,199
validation pairs; no evaluation results selected the run or checkpoint.
Paired delta d is2.522475, with95% component-bootstrap interval[2.303166,2.738972],
excluding zero under its stated assumptions (2000 seed42 draws,1396 components).
FPR delta is-.252632, interval[-.306186,-.193543].

This supports improvement on this synthetic dataset, not proof that overfitting
is absent. Run-002 epoch validation loss decreased .131101->.113960 across six
epochs, while fit window loss also decreased. The training figure shows both.
There is residual profile/template overlap across supplied sources, despite
quarantining nine exact pair overlaps and using a grouped internal validation
split. Bootstrap handles exact within-evaluation profile components; shared
templates, training-seed and run-selection variability remain unmeasured.
See reports/comparison.json and docs/FINAL_COMPARISON.md.

## 3. Where does the model still fail?

The highest supported pair-type FPR is1.0 for source type compatible, but it has
only8 negative rows; this type name does not override its binary labels.
Subtle mismatch is the main substantial weakness:115FP/121 negatives (95.04%),
accounting for115/173 total false positives. LLM-judged incompatible is48/148
(32.43%); LLM-judged compatible has56FN/151 positives (37.09%). Dealbreakers
have0FP/215 negatives; single-class recall/AUC/d are unavailable, not perfect.

Three exact source categories exceed FPR.40: dealbreakers_and_interests6/12
(.50), interests_and_values31/67 (.462687), values53/119 (.445378). Categories
are kept separate rather than silently merging names. Their class support is
shown in the category chart/report; small high-d categories are exploratory.
Examples of failures and both scores are saved in reports/final_error_analysis.json.
The subtle-mismatch examples pair spirituality with entrepreneurship and financial
independence with family bonds: scores above.5 miss the source-labeled distinctions.
These labels themselves can be ambiguous; this is descriptive evidence, not a
causal diagnosis. Additional clearly adjudicated mismatch pairs could be a future
research hypothesis, but no augmentation/tuning follows these final outcomes.
It would need a new untouched test set or an explicit exploratory designation.

## 4. Did you validate the data first?

Yes. The original train/evaluation files were pinned and hashed, schema/metadata
checked, category/pair-type support measured, duplicates/profile overlap audited,
and a pre-training quality rubric frozen. Approved quarantine leaves5,991 eligible
training rows, split4792 fit/1199 grouped validation. Original6000 training and1469
evaluation rows remain preserved. The quality gate passed at94.595919/100, above60,
with each dimension above12/20 and all20 human responses complete.

| Dimension | Score /20 | Scaled /100 |
|---|---:|---:|
| Data quality | 19.3323 | 96.6617 |
| Bias detection | 17.0356 | 85.1781 |
| Diversity | 19.4780 | 97.3898 |
| Linguistic quality | 20.0000 | 100.0000 |
| Real-life matching | 18.7500 | 93.7500 |

Strengths include structural validity, broad coverage and acceptable human language
ratings. Weaknesses include source label ambiguity, repeated templates/profiles and
heuristic bias/diversity proxies. Scores are project screening measures, not
validated fairness or romantic-compatibility instruments. A passing aggregate
score does not mean every label is correct. See reports/quality_report.json and
docs/DATA_AUDIT.md.

## 5. Would a human agree with the labels?

The completed real human review (reviewer jsd) rated15/20 labels acceptable (75%),
3/20 ambiguous (15%), and2/20 unacceptable (10%). All20 preferences were rated
realistic and hierarchy-consistent. These responses were collected before training
using the guided CSV program; no assistant fabricated or replaced the judgments.

The frozen sample is seeded and stratified across diagnostic categories/types,
with rare groups overrepresented. It is not a uniform random20-pair sample; this
is a departure from the assignment's literal random-sampling wording. The
unweighted75% agreement is diagnostic, not a population accuracy estimate.
No new human judgments were collected or inferred during Stage8. Individual
responses, IDs and sample provenance are preserved in the S3 quality/review
artifacts; label agreement contributes to the quality score.

## 6. Does the UMAP visualization show meaningful structure?

The final pair map shows clearer label concentrations than the baseline map,
with remaining mixed points. This agrees descriptively with coverage-adjusted
purity improving .518039->.833901. Final20 clusters contain1456 of1469 pairs,
with13 noise points given zero credit; clustered-only purity is.841346.
The largest cluster and detailed class counts remain in the report.

Each point is a pair represented symmetrically by normalized embeddings
[abs(u-v),u*v]. Labels are used only for coloring and purity after label-free
UMAP/HDBSCAN. Baseline/final transforms were independently fitted with identical
settings, so axes are arbitrary and unaligned; location or distance across panels
cannot be interpreted as movement. Attractive regions do not establish significance
or real romance. No bootstrap purity interval is shown because transforms/clusters
were not refitted for each sample. See figures/submission/umap_before_after.png.

## Figure receipts

All six required PNGs and their input/output SHA receipts are in figures/submission/manifest.json.
They use saved predictions and loss history only, with no new inference/clustering
or bootstrap. The improvement figure shows intervals only for the four metrics
bootstrapped in S7; other deltas are explicitly point estimates. Stage9 reproducibility
and deliverable audit remain pending. Both target tiers and unmet criteria stay visible.

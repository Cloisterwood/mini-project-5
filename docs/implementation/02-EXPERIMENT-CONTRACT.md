# Experiment contract

Freeze this contract before looking at model outcomes. Record later changes as protocol revisions, not retroactive improvements.

## Resolve contradictory assignment requirements

| Metric | Final-checklist working target | Earlier strict/stretch target |
|---|---:|---:|
| Compatibility margin | >= 0.05 | >= 0.10 |
| Cohen's d (primary) | >= 0.50 | >= 0.50 |
| FPR at cosine .5 | <= 0.35 | <= 0.10 |
| Cluster purity | >= 0.65 | >= 0.70 |
| Accuracy, precision, recall, F1, AUC | Report all | >= 0.90 each |

Report both columns. The instructor may clarify grading; these defaults do not erase the earlier targets. The assignment's baseline numbers conflict and must not be copied as results. Cohen's d measures standardized effect magnitude, not statistical significance. Cosine similarity is not a calibrated compatibility probability.

## Data and holdout policy

- Original JSONL and metadata are immutable. Record SHA-256, source URL, acquisition date, and verified license; the supplied Hugging Face dataset card declares Apache-2.0; see RESOURCES.md for its verified revision and download paths.
- `dating_pairs.jsonl`: split into fit and validation; both originate exclusively from the allowed training file. Fix split before any tuning. Record achieved counts if grouping prevents exactly 80/20.
- `eval_pairs.jsonl`: supplied held-out evaluation. Structural validation and the required baseline are allowed; prohibit model selection, early stopping, threshold tuning, or targeted training augmentation based on its results.
- Detect pair leakage with a normalized, unordered pair key independent of label. Report contradictory labels separately. Preserve raw text; normalization used for auditing is not automatic training preprocessing.
- Report repeated individual profiles and shared templates across partitions. Where feasible, group connected repeated-profile components for validation and confidence intervals. If a giant component makes this infeasible, disclose the limitation and narrower claim (new pairs, not necessarily new people/templates).
- Model input contains only the two profile texts. Category/subcategory/pair_type/label must not be embedded. Preserve gender prefixes for the main required run; any removal/swap audit is clearly separated.

## Exact primary metric definitions

Normalize each profile embedding to unit length; score each pair with cosine similarity. Predict compatible when score **>= 0.5**. Use identical policy before/after.

- Margin: mean score for label 1 minus mean score for label 0.
- Cohen's d: margin divided by pooled sample standard deviation, with sample variances (`ddof=1`), weighted by `(n1-1)` and `(n0-1)`, divided by `n1+n0-2` before square root. Preserve sign.
- FPR: FP / (FP + TN), not FP / all pairs. Report numerator and denominator.
- Accuracy, precision, recall, F1, confusion matrix: threshold .5. AUC: continuous cosine scores, not predictions.
- Undefined denominators, missing classes, fewer than two observations per class for d, or zero pooled variance: emit JSON `null` plus reason; never NaN/Infinity or a fabricated zero. Include counts and excluded/undefined bootstrap samples.
- Report by exact category and pair_type. If producing a base-category rollup for compound categories, specify mapping, overlapping membership, and denominators. Do not imply overlapping rollup counts sum to dataset size.

## Clustering and UMAP

Compatibility labels belong to pairs. Form symmetric, label-free pair features `[abs(u-v), u*v]` from normalized profile embeddings. This is a documented interpretation of the assignment's unspecified pair-level clustering method; it is not another trained classifier.

For the assignment-facing purity, use UMAP to 2D (initial frozen settings: n_neighbors=15, min_dist=.1, metric=euclidean, random_state=42), followed by HDBSCAN (min_cluster_size=15, min_samples=5, Euclidean metric). Fix settings on validation before final evaluation. Fit unsupervised transforms separately before/after; never provide labels to UMAP or clustering. Label coloring happens afterward.

For each non-noise cluster, count its majority label. Primary coverage-adjusted purity = sum of these majority counts / all pairs, treating noise as unassigned (zero credit). Also report clustered-only purity, noise fraction, cluster count, and largest cluster size. All-noise: primary purity 0, clustered-only purity null. Do not treat each noise point as a perfect singleton or silently exclude noise from the primary denominator.

Purity depends on this choice and is not necessarily comparable with unspecified assignment examples. Report the balanced-label majority baseline and sensitivity as descriptive evidence. Optional direct-feature HDBSCAN is a separately named sensitivity metric; do not replace the frozen primary method after seeing results. UMAP coordinates across separately fitted maps are not aligned axes.

## Statistical evidence

Use 2,000 seeded paired bootstrap resamples of evaluation units, evaluating baseline and fine-tuned predictions on the same sampled IDs; report 95% percentile intervals for margin/d/FPR/AUC and their deltas. If repeated profiles induce dependence, resample profile components when enough groups exist; otherwise label pair bootstrap intervals as conditional/approximate and disclose dependence. Do not bootstrap purity from fixed labels as if the entire clustering procedure had been refit.

Use the CI for delta d as primary evidence of improvement, alongside practical targets. Report uncertainty honestly if the interval includes zero. Category comparisons are exploratory; small groups may have no usable interval. Fixed-seed experiments do not measure training-seed variance; optional repeat seeds assess that separately.

## Five-dimension quality rubric

The source lists four dimensions in one table but requires five in its deliverables and final checklist. Implement all five. Each dimension has four subscores from 0 to 5; dimension total is /20, overall is their sum /100. Also emit each dimension scaled to /100 for comparison with the earlier table. Proposed thresholds below must be frozen before scoring data.

| Dimension | Four evidence-backed components |
|---|---|
| Data quality | Schema/completeness; binary-label/metadata consistency; unordered duplicate rate; contradictory-pair rate |
| Diversity | Label balance; required category coverage; pair-type coverage; unique normalized profile ratio |
| Bias detection | Gender-prefix/label association; category label skew; length/label association; vocabulary shortcut concentration |
| Linguistic quality | Human-rated naturalness; human-rated coherence; template repetition; empty/garbled/truncated text checks |
| Real-life matching | Human label agreement; plausible preferences; consistency with stated preference hierarchy; human ambiguity/contradiction audit |

Scoring specification to implement in `configs/quality-rubric.yaml` during S3: every component records raw statistic, denominator, range, and explicit mapping to 0–5. Fraction-of-valid/acceptable components use `5 * acceptable_fraction`; required-category/pair-type coverage uses fraction present. Label balance gets 5 in [.4,.6], declining linearly to 0 at 0 or 1. Duplicate rate gets 5 at 0, declining to 0 at .05 (clip to [0,5]). For association, repetition, and vocabulary heuristics, define justified thresholds in the frozen YAML before evaluating, with examples; do not claim these are validated fairness or linguistic instruments. Human ratings use a written acceptable/ambiguous/unacceptable rubric and preserve individual responses.

Conservative working gate: overall >=60 and every dimension >=12/20. Show earlier dimension targets (80/60/50/50 out of 100) separately, because they differ. Missing human review leaves gate pending. Structural invalidity, contradictory unresolved records, and cross-file pair leakage block training regardless of aggregate score. The rubric is a project heuristic; a high score is not proof of unbiased or realistic matchmaking.

## Bound the scope

Required: MiniLM, CosineSimilarityLoss, five quality dimensions, all metrics/charts, three actual training configurations, reproducible artifacts. No OpenAI API calls are required for this pipeline. Optional instruction prefixes, alternate losses, extra generated data, dashboards, and cross-domain evaluation come only after required deliverables and a new scoped decision. MultipleNegativesRankingLoss is not a drop-in replacement for labeled negative pairs and can create false in-batch negatives; keep it out of the core pathway.

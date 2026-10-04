# Current handoff

Updated: 2026-10-04. Status: S4 COMPLETE — BASELINE MEASURED AND SEALED.

## Next single action

Wait for “Continue with Stage 5.” Read S5 in 01-STAGES.md, the experiment contract,
and operating rules. Implement only train.py, frozen default training config,
checkpoint/resume, fit-only smoke test, and measured compute benchmark. Preserve
CosineSimilarityLoss, base revision, 256-token raw-text preprocessing, gender
prefixes, normalized cosine scoring and threshold .5. Smoke training uses only fit
IDs; monitoring/checkpoint choice uses validation; training must not read eval.
No paid provisioning without user choice. Stop after S5 or context checkpoint.

Resume prompt: “Continue with Stage 5. Read the handoff, reverify data/quality and
sealed baseline, implement training smoke test and checkpoint round-trip, measure
local throughput, save Linear/GitHub evidence, then stop.”

## Stage 4 completed work

- CLO-46 / S4 Done. Implemented metrics.py, cluster.py, evaluate.py and baseline CLI.
- S3 quality gate freshly reverified: passed, 94.59591920379897/100, 20 real reviews.
- configs/baseline.json frozen BEFORE model outcomes with config manifest.
  SHA-256 f15fb4da6d4898f863d8df34c36b112dbb40df07932ef9b207d334c1bb4b64c1.
- HF model API verified revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 for
  sentence-transformers/all-MiniLM-L6-v2. Downloaded eleven files (about 87MiB)
  into ignored models/base-minilm/<revision>/. All actual file hashes verified.
- CPU, four Torch threads, batch32, float32, max sequence length256, raw profiles
  only, no prefix instructions, gender prefixes retained, unit embeddings.
  Model inputs never include labels/categories/subcategories/pair types.
- Symmetric label-free pair features [abs(u-v),u*v]; UMAP2D n_neighbors15,
  min_dist.1, euclidean, seed42, n_jobs1. External hdbscan HDBSCAN
  min_cluster_size15/min_samples5/euclidean/core_dist_n_jobs1.
  Maps fitted separately per split, labels used afterward for coloring/purity.
- Direct-feature sensitivity excluded as optional before outcomes. Clustered-only
  purity/noise/counts/sizes/observed and balanced-label majority baselines recorded.
- Cache under ignored .cache/embeddings/, keyed by ordered record IDs, raw text,
  model files, preprocessing, execution/software. Checksums/shape/norm checked.
- No truncation: maximum profile token length fit36/validation33/evaluation35.
- Reports: fit_baseline_metrics.json, validation_baseline_metrics.json,
  baseline_metrics.json (evaluation). All metrics, denominators, null reasons,
  exact category/pair-type support and breakdowns, both target tiers recorded.
- reports/baseline/: three row-aligned prediction CSVs and nine inspected PNGs
  (similarity, label-colored pair UMAP with noise, ROC for each split).
- reports/baseline_manifest.json seals seventeen tracked artifacts by SHA-256,
  inputs/config/quality/model/software/code provenance and split runtime.
  Repeat baseline verifies seal and returns without inference/evaluation rerun.
  Never use evaluation baseline for thresholds, selection, early stopping, or
  targeted augmentation. Development decisions use validation only.

## Measured results

| Split | N | Margin | Signed d | FPR at .5 | Accuracy | AUC | Coverage-adjusted purity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit | 4792 | -0.0843702154 | -0.3904995676 | 0.4956158664 | 0.3879382304 | 0.3992387840 | 0.4578464107 |
| Validation | 1199 | -0.0877095853 | -0.4048328864 | 0.5075125209 | 0.3753127606 | 0.4016193656 | 0.4962468724 |
| Evaluation | 1469 | -0.0787518547 | -0.3640484156 | 0.4802631579 | 0.4050374404 | 0.4007571821 | 0.5180394826 |

No baseline performance target is met in either target tier. This is baseline
measurement completion, not a claim of target attainment. Signed separation is
negative; do not invert scores or change threshold after looking at outcomes.
The model measures semantic similarity, not calibrated romantic compatibility.
S3 quality passing does not guarantee useful model separation.

Clusters: fit70/1394noise/largest378; validation4/17noise/largest1108;
evaluation7/18noise/largest1303. Clustered-only purity .645674/.503384/.524466,
respectively. Coverage-adjusted purity gives noise zero credit over all pairs.
Evaluation purity .518039 is close to observed majority baseline .517359.
Separate UMAP axes are arbitrary and not aligned. No bootstrap purity claims.
Paired uncertainty belongs to S7; S4 computes no bootstrap intervals.

## Actual validation and execution receipts

- BEFORE model loading: 20 hand-calculated metric/cache/seal tests passed.
  Tests check weighted ddof1 d and sign, FPR denominator, >= ties, continuous AUC,
  one-class/zero-variance/null cases, pair symmetry, noise/all-noise purity,
  record alignment, frozen inputs, cache corruption and seal tampering.
- Final full pytest: exit0, 96 passed in11.53s (75 prior +21 S4 cases).
  Added explicit gate-before-download check after initial95-pass full suite.
- First sandbox baseline command: exit1, Hub snapshot unavailable/network error.
  Retried with requested network approval and the same frozen config; actual
  baseline exit0, BASELINE PASS. No scoring/config change or discarded result.
- Split elapsed times: fit31.802s, validation6.709s, evaluation6.727s, including
  encoding/clustering/plots. Fit includes initial UMAP/JIT overhead; these are
  inference/analysis times, not training throughput or compute provisioning evidence.
- Repeated baseline command: exit0, BASELINE SEALED ARTIFACTS VERIFIED;
  no inference or evaluation repeated.
- Independent audit reproduced all metrics, category/pair-type breakdowns, and
  cluster purity from saved predictions, checked source IDs/order/labels and
  threshold predictions for all 4792/1199/1469 rows.
- Eleven model file hashes verified. Nine PNGs loaded and visually inspected:
  readable axes, split/sample counts, class legends, threshold, noise and ROC.
- No code/scientific repair after model outcomes; only tests/documentation added.
  No training, fine-tuned weights, API inference, remote or paid compute active.

## Preserved data and scientific limits

- Original pinned 6000-row training and1469-row evaluation sources/metadata
  unchanged. Approved nine-row quarantine =>5991 eligible, seed42 grouped split
  4792fit/1199validation. Eligible/evaluation pair overlap zero; profile/template
  overlap remains disclosed. Active immutable manifests under reports/quarantine-v1/.
- S3 frozen rubric/sample and completed actual human responses unchanged.
  Label agreement15 acceptable/3 ambiguous/2 unacceptable; other fields20 acceptable.
- Structural eligibility and source-bound quality guard required for future training.
- Synthetic labels, heuristic human diagnostics, residual profile/template overlap,
  fixed seeds and no real-world romantic ground truth limit claims.

## Provenance, environment and saving

- reports/baseline_config_manifest.json file SHA:
  12e9f737d1c711313d0f0ff38a49d7b1adae5a217785b7eec76677e77f801bd6.
- reports/base_model_manifest.json file SHA:
  7eba2f5aa611bda902405181576a6fda82a2c0d0009c5d95892e599e58e45097.
- reports/baseline_metrics.json evaluation file SHA:
  deac4e61b8d4ab14a7d3e25fb0d73acf9716cddd511f607f27dc7bcac2690bb1.
- reports/baseline_manifest.json file SHA:
  402dee34a2006b1fc89203a6e18031787880fcd51aae672a91ab13d2965a8236.
- Locked .venv Python3.12.13, local arm64 Mac14,2/macOS27.0/24GiB. Actual baseline
  library versions recorded per report; no dependencies or lock changed.
- Code commit at execution c645e05585d605c087b62dec5790a6999b5b345f plus exact
  uncommitted implementation SHA hashes in every baseline report. Final delivery
  commit is recorded on CLO-46; a commit cannot contain its own SHA.
- Assistant GPT-6 family per instructions, exact picker/reasoning not exposed;
  user preference Sol6.1. Context telemetry unavailable; exact75% boundary
  unmeasurable. Sequential work, no agents, no paid compute.
- Main, private https://github.com/Cloisterwood/mini-project-5. Starting commit
  c645e05585d605c087b62dec5790a6999b5b345f. Reviewed code/config/reports/CSVs/PNGs/
  docs committed and pushed, live remote SHA verified at closeout on CLO-46.
  Original data, model weights and caches excluded.
- S0–S4 Done; S5 not started. STOP: S4 complete.

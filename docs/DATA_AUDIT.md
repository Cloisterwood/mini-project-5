# Stage 2 data audit

Status: implementation verified; original-source leakage gate BLOCKED. No source rows were removed or changed.

Original counts are 6,000 train (2,997 positive / 3,003 negative) and 1,469 evaluation (709 positive / 760 negative). Both metadata files match every observed summary field.

Training has 40 normalized unordered duplicate extras (22 ordered repeats and 18 reversed repeats); evaluation has 2 (1 ordered, 1 reversed). Neither file has contradictory duplicate labels.

The training-only profile-component split is 4,799 fit / 1,201 validation, with zero shared normalized profiles or pairs. There are 4,796 profile components; the largest contains 21 pairs. Fine category/pair-type strata had insufficient distinct-group support, so stratification was coarsened to label/pair_type while preserving all profile groups. Seed is 42.

Fit and validation share 312 coarse template proxies. Original train/eval share 527 individual normalized profiles and 436 template proxies. Proxies are the first three words after a gender prefix, not confirmed generator templates. These findings limit claims about unseen people or writing templates.

## Exact source leakage

All nine overlapping pairs have label 0. Exact stable IDs and hashes are in reports/exploration.json. Line numbers below are 1-based.

| Training line | Evaluation line |
|---:|---:|
| 2776 | 566 |
| 360 | 958 |
| 2398 | 956 |
| 1704 | 1068 |
| 1690 | 369 |
| 511 | 1238 |
| 1290 | 579 |
| 4164 | 458 |
| 1256 | 1037 |

## Proposed remedy — not applied

Quarantine the 9 training rows whose normalized unordered pair appears in evaluation. Preserve all original files, retain all 1469 evaluation rows, and record every excluded ID plus hashes in a versioned derived-cohort manifest. Eligible training would be 5991 rows (2997 positive / 2994 negative). Pair overlap would be zero; 519 individual profiles would still be shared with evaluation, requiring disclosure.

The proposal is in reports/remediation_proposal.json. This changes the eligible training cohort from the frozen 6,000-row source, so a user decision is required before applying it. Alternatively keep S2 blocked and seek instructor/data-owner guidance. No claim of a clean final holdout is made for the current files.

## Checks and report integrity

The first persisted manifest exposed a JSON key-type checksum defect during reload: integer component-size histogram keys serialize as strings. The code now stores string keys before hashing, with a saved/reloaded-manifest regression test. Initial unverified audit artifacts are preserved under reports/s2-initial-audit/ and must not be used for training. The selected split IDs are unchanged; this was a coding repair, not a scientific experiment.

Original source/card hashes and license declaration are in reports/acquisition_manifest.json. The dataset card at the pinned revision declares Apache-2.0; raw files remain excluded from Git. No model, metric, human quality score, or training run is part of this stage.

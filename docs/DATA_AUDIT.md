# Stage 2 data audit

Status: Stage 2 complete. The original source has nine overlapping pairs; the user-approved derived training cohort passes the structural gate. No original source rows were removed or changed.

Original counts are 6,000 train (2,997 positive / 3,003 negative) and 1,469 evaluation (709 positive / 760 negative). Both metadata files match every observed summary field.

Training has 40 normalized unordered duplicate extras (22 ordered repeats and 18 reversed repeats); evaluation has 2 (1 ordered, 1 reversed). Neither file has contradictory duplicate labels.

The original-source audit split was 4,799 fit / 1,201 validation. The approved eligible split is **4,792 fit / 1,199 validation**, seed 42, with zero shared normalized profiles or pairs. Fine category/pair-type strata lacked sufficient distinct-group support, so stratification was coarsened to label/pair_type while preserving connected profile groups. Counts and component statistics are in the active manifest.

Active fit and validation share 344 coarse template proxies. Original train/eval share 527 individual normalized profiles; eligible train/eval share 519. Both original and eligible train/eval share 436 template proxies. Proxies are the first three words after a gender prefix, not confirmed generator templates. These findings limit claims about unseen people or writing templates.

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

## Approved remedy — applied October 4, 2026

The user authorized: “Use only the eligible rows and proceed with the quarantine.” The nine original training IDs listed in configs/quarantine-v1.json are excluded from both fit and validation. All original files and all 1,469 evaluation rows are preserved. Eligible training is 5,991 rows (2,997 positive / 2,994 negative). Pair overlap is zero; 519 profiles remain shared with evaluation and must be disclosed.

The historical proposal and original blocked audit remain at the report root. Active evidence is in reports/quarantine-v1/cohort_manifest.json, exploration.json, and split_manifest.json. This is an explicit protocol revision to eligibility, not an edit of the 6,000-row source or the evaluation set. It uses structural pair identity only, with no model outcomes. The active cohort/split manifests are immutable and revalidated from original source bytes by load_eligible_bundle(). S3 quality and human review are still required before training.

## Checks and report integrity

The first persisted manifest exposed a JSON key-type checksum defect during reload: integer component-size histogram keys serialize as strings. The code now stores string keys before hashing, with a saved/reloaded-manifest regression test. Initial unverified audit artifacts are preserved under reports/s2-initial-audit/ and must not be used for training. The selected split IDs are unchanged; this was a coding repair, not a scientific experiment.

Original source/card hashes and license declaration are in reports/acquisition_manifest.json. The dataset card at the pinned revision declares Apache-2.0; raw files remain excluded from Git. No model, metric, human quality score, or training run is part of this stage.

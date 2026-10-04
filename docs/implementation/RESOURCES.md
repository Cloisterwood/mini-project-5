# Project resources and tracking

Verified October 3, 2026.

- GitHub: https://github.com/Cloisterwood/mini-project-5 (private, owner Cloisterwood).
- Linear workspace: https://linear.app/cloisterwood; team Cloisterwood.
- Linear project: https://linear.app/cloisterwood/project/mini-project-5-dating-compatibility-1a9f2484672e
- Project UUID: `49861c5f-c59f-4a9c-a0ad-8802cf03ee5d`.
- Dataset: https://huggingface.co/datasets/dipenbhuva/dating-compatibility-pairs
- Dataset revision verified from Hugging Face API: `69b750dd024d5732ca19596d86bffc253abee92e`.
- Dataset card at the pinned revision declares Apache-2.0; public and ungated. S2 acquired and validated all 6,000/1,469 records and matching metadata. The approved nine-row training quarantine leaves 5,991 eligible rows and zero cross-file pair overlap. Original files are unchanged; active reports are under reports/quarantine-v1/. S3 human review is complete and the verified quality gate passes at 94.60/100.
- Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, pinned S4 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; file hashes in reports/base_model_manifest.json. Baseline complete and sealed; weights excluded from Git.

## Dataset acquisition

All four required files are present at the repository root. In Stage 2 download the files below from `https://huggingface.co/datasets/dipenbhuva/dating-compatibility-pairs/resolve/69b750dd024d5732ca19596d86bffc253abee92e/<filename>` into local `data/`, then hash and validate them. No extra dataset link is needed.

- `dating_pairs.jsonl`
- `eval_pairs.jsonl`
- `dating_pairs_metadata.json`
- `eval_pairs_metadata.json`

## Stage issues

| Stage | Linear issue | Setup status |
|---|---|---|
| S0 | [CLO-42](https://linear.app/cloisterwood/issue/CLO-42/mp5-s0-access-destinations-and-decisions) | Done |
| S1 | [CLO-43](https://linear.app/cloisterwood/issue/CLO-43/mp5-s1-reproducible-environment-and-package-skeleton) | Done |
| S2 | [CLO-44](https://linear.app/cloisterwood/issue/CLO-44/mp5-s2-dataset-validation-exploration-and-split) | Done |
| S3 | [CLO-45](https://linear.app/cloisterwood/issue/CLO-45/mp5-s3-quality-gate-and-human-review) | Done |
| S4 | [CLO-46](https://linear.app/cloisterwood/issue/CLO-46/mp5-s4-metrics-and-baseline) | Done |
| S5 | [CLO-47](https://linear.app/cloisterwood/issue/CLO-47/mp5-s5-training-smoke-test-and-compute-decision) | Todo |
| S6 | [CLO-48](https://linear.app/cloisterwood/issue/CLO-48/mp5-s6-three-bounded-training-experiments) | Todo |
| S7 | [CLO-49](https://linear.app/cloisterwood/issue/CLO-49/mp5-s7-frozen-final-comparison-and-uncertainty) | Todo |
| S8 | [CLO-50](https://linear.app/cloisterwood/issue/CLO-50/mp5-s8-all-figures-and-interpretation) | Todo |
| S9 | [CLO-51](https://linear.app/cloisterwood/issue/CLO-51/mp5-s9-reproducibility-github-delivery-and-closeout) | Todo |

Each issue contains the stage scope and acceptance checks; each successive stage is blocked by its predecessor. The user confirmed Sol 6.1 is available and selected as the default development model. S0 was marked Done on October 4 after Linear access in the active development session and the private GitHub destination were verified. Actual stage status lives in Linear and the handoff.

The existing Personal Agent Engine projects were found. A dedicated coursework project in the Cloisterwood team was chosen to match Mini Project 4 and keep the ten-stage plan together.

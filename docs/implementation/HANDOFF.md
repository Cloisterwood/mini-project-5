# Current handoff

Updated: 2026-10-04. Status: STAGE 2 COMPLETE — STAGE 3 NOT STARTED.

## Next single action

Read the shared operating rules, revised experiment contract, and S3 in 01-STAGES.md.
Implement the frozen quality rubric and training-only human review sheet using
`load_eligible_bundle()` from dating_compat.cohort. Do not use root historical split
IDs or quarantine IDs in quality samples, fit, or validation. Stop at S3 completion
or when human review is needed; an LLM cannot supply the user's human ratings.

Resume prompt: “Continue with Stage 3. Use the verified eligible cohort from the
handoff, implement and freeze the quality rubric, prepare the training-only human
review sheet, verify quality-gate protection, save Linear/GitHub evidence, then stop.”

## Approved protocol revision S2-Q1

The user authorized on October 4: “Use only the eligible rows and proceed with the
quarantine.” Policy is recorded in configs/quarantine-v1.json. Exactly nine original
training IDs whose normalized unordered pairs occur in evaluation are quarantined
from eligibility. All original source files and all 1,469 evaluation rows are unchanged.
This structural remedy uses no model outcomes. Source hashes and exact exclusion IDs
are frozen in the policy and active cohort manifest. The original blocked reports
and proposal remain historical evidence under reports/; initial unverified coding
artifacts remain under reports/s2-initial-audit/ and must not be used for training.

Active reports: reports/quarantine-v1/cohort_manifest.json, exploration.json,
and split_manifest.json. Original metadata still describes 6,000 original training
rows; eligible cohort statistics and quality denominators use 5,991 rows.

## Actual Stage 2 results

- Issue: S2 / CLO-44. Structural gate passes; S3 human quality gate has not run.
- Original source counts and matching metadata: 6,000 training / 1,469 evaluation.
- Eligible training: 5,991 (2,997 positive / 2,994 negative); excluded: nine label-0 rows.
- Seed-42 connected-profile split: 4,792 fit / 1,199 validation.
- Fit: 2,397 positive / 2,395 negative. Validation: 600 positive / 599 negative.
- Zero normalized unordered pair overlap between eligible training and evaluation.
- Zero shared normalized profiles or pairs between fit and validation.
- Evaluation is preserved in full, including its original two within-file duplicate extras.
- Remaining eligible/evaluation shared profiles: 519. Surface template proxies: 436.
- Fit/validation shared surface template proxies: 344. Proxies are only first-three-word
  scaffold diagnostics, not confirmed generator templates. Do not claim new unseen
  people/templates or statistically independent evaluation pairs without qualification.
- Original training duplicate extras: 40, evaluation: two; no contradictory duplicate labels.
- Grouping is never abandoned; label/category/pair_type strata were coarsened to
  label/pair_type only after reporting insufficient distinct-component support.

## Implementation and verified checks

- cohort.py verifies the policy's exact source hashes, exclusion IDs, original metadata
  gate, and actual source overlap. It preserves raw texts/evaluation, writes complete
  eligible/evaluation ID provenance, and recomputes saved cohort/split integrity.
- `load_eligible_bundle()` reloads original sources, checks pinned acquisition and policy,
  rederives eligibility, compares the saved cohort, then calls assert_data_ready on the
  active split. Later stages must use this entry point and add the separate S3 human gate.
- `python -m dating_compat explore` now defaults to the approved policy and writes
  active reports under reports/quarantine-v1/. Acquire remains rooted in reports/.
- `.venv/bin/python -m dating_compat explore`: exit 0, DATA EXPLORATION PASS.
- `.venv/bin/python -m pytest -q`: exit 0, 46 passed. New tests protect exact exclusions,
  source hashes, structural blockers, evaluation preservation, tampered eligibility,
  rehashed false cohort claims, saved cohort JSON roundtrip, and quarantine exclusion
  from both partitions. Prior loader/split/doctor tests continue to pass.
- Fresh `load_eligible_bundle()` plus saved-manifest validation: ELIGIBLE DATA PASS,
  5,991 train / 1,469 eval. Original SOURCE_HASHES verified unchanged for all five files.
- Identical explore rerun leaves frozen active reports unchanged and passes again.
- Active file SHA-256: cohort cbae2a16662a51b1e7d6e62bc8ced75d15d661ce08c25b99300d16ef678f2675;
  exploration 3e0f1cfaf562598aecf01fe91b401f3f29a2e34f3bdb68e210bae75570e0e6e2;
  split 8b0695476b179cfaedbbbdd51d7c3337136a04c9d17e0f1c4b65dceb63285851.
- git diff --check: exit 0. No dependencies changed; uv.lock and environment reports unchanged.

## Saving, host, and stop

- Local macOS 27.0 arm64 Mac14,2, 24 GiB RAM; project .venv Python 3.12.13.
- No model download, training process, paid compute, API inference, or subscription change.
  All audit/test sessions finished before closeout. CPU/MPS verification remains S1 evidence.
- Assistant GPT-6 family per session instructions; exact picker model/reasoning not exposed.
  Saved preference Sol 6.1. Context indicator unavailable; exact 75% boundary cannot be measured.
- Branch main; confirmed private GitHub https://github.com/Cloisterwood/mini-project-5.
  Starting checkpoint 0d878ea9dd30d7031f61787e2aff2edbe9f07e1f. Final reviewed commit and
  verified remote SHA are recorded on CLO-44; this commit cannot contain its own SHA.
- Linear moved back In Progress when quarantine was approved; completion and verified
  publishing receipt recorded at closeout. S2 Done only after these actual checks.
- Stop reason: Stage 2 complete. No unresolved data-policy decision. Do not start S3 until
  the user says continue. Preserve original-source audit evidence and the historical July plan.

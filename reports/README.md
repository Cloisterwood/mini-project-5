# Reports

`environment.json` records the Stage 1 MPS doctor result outside the sandbox;
`environment-cpu.json` records the CPU check. Later stages add measured data,
quality, baseline, training, and evaluation reports. No model performance is measured in S1.

S2 artifacts: `acquisition_manifest.json`, `exploration.json`, `split_manifest.json`,
and `remediation_proposal.json`. These are historical original-source evidence;
the original data has nine overlapping pairs. The approved eligible cohort and
active split/exploration artifacts are under `quarantine-v1/`. Its structural gate
passes; S3 human quality review is still required. Initial invalid
checksum artifacts are preserved under `s2-initial-audit/` with an explanatory README.

S3: `quality_rubric_manifest.json` freezes the YAML hash before scoring;
`human_review_sample.json` freezes 20 eligible fit IDs and sampling provenance;
`quality_report.json` is the current quality status and updates when actual human
responses arrive. It is pending, with human scores/overall null. Ratings remain
blank in docs/human_review_responses.csv. No passing human review is claimed.

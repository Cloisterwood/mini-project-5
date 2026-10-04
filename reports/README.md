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

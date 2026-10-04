# Reports

`environment.json` records the Stage 1 MPS doctor result outside the sandbox;
`environment-cpu.json` records the CPU check. Later stages add measured data,
quality, baseline, training, and evaluation reports. No model performance is measured in S1.

S2 artifacts: `acquisition_manifest.json`, `exploration.json`, `split_manifest.json`,
and `remediation_proposal.json`. The original source data gate is blocked by nine
train/eval overlapping pairs; the proposed quarantine is not applied. The split
manifest is a reproducible audit artifact, not permission to train. Initial invalid
checksum artifacts are preserved under `s2-initial-audit/` with an explanatory README.

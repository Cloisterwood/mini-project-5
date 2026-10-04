# Reports

`environment.json` records the Stage 1 MPS doctor result outside the sandbox;
`environment-cpu.json` records the CPU check. Later stages add measured data,
quality, baseline, training, and evaluation reports. No model performance is measured in S1.

S2 artifacts: `acquisition_manifest.json`, `exploration.json`, `split_manifest.json`,
and `remediation_proposal.json`. These are historical original-source evidence;
the original data has nine overlapping pairs. The approved eligible cohort and
active split/exploration artifacts are under `quarantine-v1/`. Its structural gate
passes; S3 human quality review is complete and its gate passes. Initial invalid
checksum artifacts are preserved under `s2-initial-audit/` with an explanatory README.

S3: `quality_rubric_manifest.json` freezes the YAML hash before scoring;
`human_review_sample.json` freezes 20 eligible fit IDs and sampling provenance;
`quality_report.json` is the current quality status and updates when actual human
responses arrive. All 20 actual human reviews are recorded, with quality 94.60/100.

S4: baseline_config_manifest.json freezes settings before outcomes, and
base_model_manifest.json pins the exact revision and eleven file hashes.
fit_baseline_metrics.json, validation_baseline_metrics.json, and
baseline_metrics.json (evaluation) store measured metrics/targets/breakdowns.
baseline/ contains row-aligned predictions and nine inspected PNGs. All seventeen
artifacts are sealed by baseline_manifest.json. The baseline command verifies
them on repeat invocation without inference. Do not overwrite evaluation evidence
or use it for development decisions. Baseline performance targets are unmet.

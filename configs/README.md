# Configuration

Stage 3 froze quality-rubric.yaml before scoring, with its SHA-256 in
reports/quality_rubric_manifest.json. Do not change thresholds to pass the gate.
quarantine-v1.json records approved eligibility. Stage 5 adds training configuration.
The current experiment contract is in `docs/implementation/02-EXPERIMENT-CONTRACT.md`.

S4 baseline.json pins model revision, CPU/float32 execution, max sequence length
256, raw texts/gender prefixes, threshold .5, and the contract's UMAP/HDBSCAN
settings. Frozen before outcomes in reports/baseline_config_manifest.json. Do not
change it after viewing evaluation to improve results. Training must preserve
the same preprocessing and primary scoring policy.

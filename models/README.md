# Local model artifacts

Model weights and checkpoints go here in later stages and are excluded from Git.
Durable storage and checksums must be recorded before delivery. S1 creates no model.

S4 downloaded the public base MiniLM revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41 into base-minilm/<revision>/ (about 87MiB).
The eleven files are hashed in reports/base_model_manifest.json. Reproduce the
download with the pinned baseline configuration; this folder is not pushed to
GitHub. S5 smoke checkpoints exist under ignored checkpoints/s5-smoke/ (about 1 GiB),
with optimizer/scheduler/RNG and file hashes in reports/compute_benchmark.json.
These are local non-scored diagnostic weights. No scored experiment exists yet.

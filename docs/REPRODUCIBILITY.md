# Reproduce and retrieve the delivery

GitHub source/report repository: https://github.com/Cloisterwood/mini-project-5
(private; collaborator access required). Selected inference model release:
https://github.com/Cloisterwood/mini-project-5/releases/tag/v0.1.0 . Sign into gh
using your own account; do not paste credentials into project files or chat.

## Retrieve on a fresh checkout

Run in Terminal. Python3.12.13 and uv>=.10.5 required; gh authenticates private access.

```bash
gh repo clone Cloisterwood/mini-project-5
cd mini-project-5
uv sync --locked --python 3.12.13 --cache-dir .cache/uv
.venv/bin/python -m dating_compat acquire
.venv/bin/python -m dating_compat prepare-training
mkdir -p .cache/release-download checkpoints/run-002
gh release download v0.1.0 --repo Cloisterwood/mini-project-5 --dir .cache/release-download --pattern run-002-selected.tar.gz --pattern model_release_manifest.json
.venv/bin/python -c "from pathlib import Path; from dating_compat.data import file_hash; import json; m=json.loads(Path('reports/model_release_manifest.json').read_text()); assert file_hash(Path('.cache/release-download/run-002-selected.tar.gz'))==m['archive_sha256']; print('ARCHIVE SHA PASS')"
tar -xzf .cache/release-download/run-002-selected.tar.gz -C checkpoints/run-002
.venv/bin/python -m dating_compat visualize
```

Extraction above is for a fresh checkout with no selected directory. Preserve an
existing model; extract into a separate directory to inspect rather than overwriting.
Archive83,517,969bytes, SHA e46e73e4d41445d794b3672eafddebea42f1072473122eed532d7760992be3df.
Inner selected file hashes/state in reports/model_release_manifest.json. Selected
state8c79ccf853ccd4e24e64e55b637b68f2d7847c6208ca2911cc7e69ea9e2a59f4.
Release has inference files only, including its generated model card with a few
synthetic fit examples; no optimizer/full resume or full dataset. Data source's
Apache-2.0 declaration and source/hash receipts are preserved by acquisition.
Base MiniLM revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41 attribution is in its
model card/base manifest. Coursework is not a production matchmaking model.

The visualize command verifies saved report seals without new inference or
resampling. Local final-evaluation verification additionally requires the selected
model directories for all three historical runs; only run-002 is released. Use the
delivery verification command below on a fresh checkout with the released model.

## Verify regeneration and saved-model loading

Choose an unused output directory; never delete/replace original frozen reports.

```bash
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m dating_compat visualize --output .cache/regenerated-figures
.venv/bin/python scripts/verify_delivery.py --model-dir checkpoints/run-002/selected --regenerated-figures .cache/regenerated-figures --output .cache/delivery-verification.json
.venv/bin/python -m pytest -q
```

Expected marker DELIVERY VERIFICATION PASS. Script verifies model file/state hashes,
loads it twice and checks identical normalized embeddings on eight fit-only probes,
rebuilds baseline/final/comparison aggregates from stored prediction CSVs, and
requires byte-identical JSON/PNG files. Stored uncertainty is reused, not re-estimated.
Reports regenerate into .cache/reproduced-reports/. The clean-environment experiment
used an archive extracted to a separate directory, independent of the original model.

## Rebuilding the training experiments

The archive is the exact selected artifact; rebuilding need not produce identical
weights on another device/platform. Full historical commands are:

```bash
.venv/bin/python -m dating_compat acquire
.venv/bin/python -m dating_compat explore
.venv/bin/python -m dating_compat quality
.venv/bin/python -m dating_compat baseline
.venv/bin/python -m dating_compat prepare-training
.venv/bin/python -m dating_compat train-smoke --device cpu
.venv/bin/python -m dating_compat train --run-id run-001 --device cpu
.venv/bin/python -m dating_compat validate-run --run-id run-001
.venv/bin/python -m dating_compat train --run-id run-002 --config configs/train-run-002.yaml --device cpu
.venv/bin/python -m dating_compat validate-run --run-id run-002
.venv/bin/python -m dating_compat train --run-id run-003 --config configs/train-run-003.yaml --device cpu
.venv/bin/python -m dating_compat validate-run --run-id run-003
.venv/bin/python -m dating_compat select-run
.venv/bin/python -m dating_compat final-evaluation
.venv/bin/python -m dating_compat visualize
```

These are recorded experiment commands, not permission to overwrite delivered runs.
For independent training create a separate workspace containing source, lock/configs,
approved human review and original acquired data, with initially empty report,
figure, model and checkpoint directories. Keep the frozen configs and review CSV;
regenerate governance reports in that separate workspace. Baseline downloads the
pinned base model. Retain all baseline/quality/cohort gates. Run fresh preflight before
each experiment. Review/select only validation results; final evaluation once after
selection. Existing runs require explicit same-run resume with all model/optimizer/
scheduler/RNG state; the inference release cannot resume training.

CPU4threads/float32 was measured (actual runtimes274.03/463.36/492.73s); MPS/CUDA
training not verified. No cloud resource is provisioned. Independent seed/device
comparisons need their own budget and an untouched test set for new tuning claims.

## Actual clean-environment evidence

Fresh /private/tmp/mp5-s9-venv installed the unchanged uv.lock offline from existing
cached wheels, on the same ARM64 macOS host. ENVIRONMENT PASS,121 tests in37.53s.
Original environment121 tests in11.81s. Restored selected model files/state matched,
eight fit-probe embeddings differed by0; three regenerated JSON reports and all six
submission PNGs matched original bytes. Downloaded private release asset bytes and
GitHub asset digest match the archive SHA. Receipts: reports/reproducibility/.
This tests environment isolation and artifact restoration, not a fresh internet
install, another operating system, or repeat full training. See the unchecked
scientific/assignment criteria in docs/DELIVERABLE_CHECKLIST.md.

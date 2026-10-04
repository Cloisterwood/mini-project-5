# Current handoff

Updated: 2026-10-04. Status: S2 IMPLEMENTATION VERIFIED — DATA GATE BLOCKED.

## Next single action

Resolve the pending user choice: quarantine the nine overlapping source-training
rows from the derived training cohort (recommended), or keep S2 blocked and seek
instructor/data-owner guidance. The concrete proposal and excluded IDs are in
`docs/DATA_AUDIT.md` and `reports/remediation_proposal.json`. It has NOT been applied.
Do not start S3 or training while this decision is pending.

If approved, record a protocol revision that keeps original files immutable,
retains all 1,469 evaluation rows, excludes the nine specified training IDs from
fit/validation eligibility, and records all hashes. Eligible training becomes
5,991 rows, 2,997 positive / 2,994 negative. Regenerate a NEW versioned manifest,
update the structural readiness check to verify that derivation against original
hashes and the exact exclusion list, and test zero cross-file pair leakage. Do not
silently modify, overwrite, or delete the original acquisition/audit evidence.
Shared individual profiles and template proxies still need disclosure; removing
pair overlap does not prove independence or realistic compatibility.

Resume prompt: “Resume S2. Read the handoff and the user's leakage-remedy decision.
Apply only the approved source-preserving derivation, verify its provenance and
leakage checks, save Linear/GitHub receipts, then stop before S3.”

## S2 checkpoint and actual findings

- Issue: [CLO-44](https://linear.app/cloisterwood/issue/CLO-44/mp5-s2-dataset-validation-exploration-and-split). Implementation verified; not Done because the original data gate is blocked.
- Stop reason: pending human decision on source leakage. Context indicator unavailable; exact 75% boundary cannot be measured. Assistant GPT-6 family per session instructions; picker model/reasoning not exposed. Saved preference Sol 6.1.
- Host/interpreter: macOS 27.0 arm64 Mac14,2, 24 GiB RAM; project .venv Python 3.12.13. No model/device workload in S2; S1 verified CPU/MPS. uv.lock unchanged.
- Pinned revision: 69b750dd024d5732ca19596d86bffc253abee92e. Five files acquired: four data/metadata files and dataset card. Raw bytes in ignored data/; all original checksums verified. Pinned card declares Apache-2.0.
- Observed/expected counts: 6,000 train and 1,469 evaluation. Metadata matches all observed fields, including categories and both-class counts. No schema-invalid rows; none discarded.
- Train labels: 2,997 positive / 3,003 negative. Eval labels: 709 positive / 760 negative. Pair type is metadata and never supplies the target.
- Normalized unordered duplicate extras: train 40 (22 ordered + 18 reversed), eval 2 (1 ordered + 1 reversed). No contradictory duplicate groups.
- Cross-file normalized unordered overlap: 9 pairs, all label 0. Exact original line numbers and stable IDs are recorded in DATA_AUDIT.md and exploration.json.
- Training-only grouped split: 4,799 fit / 1,201 validation, seed 42. Zero shared normalized profiles or unordered pairs across fit/validation. Fit labels 2,397/2,402 positive/negative; validation 600/601.
- Profile components: 4,796; largest 21 pairs. Rare label/category/pair_type strata lacked five distinct groups, so label/pair_type stratification was used without relaxing grouping. Distribution deviations and rare supports are reported.
- Shared normalized profiles train/eval: 527. Shared coarse template proxies: fit/validation 312; train/eval 436. Templates are only first-three-word surface proxies, not confirmed generator identity.
- Proposed training quarantine would leave zero pair overlap and 519 shared normalized profiles with evaluation. This is a proposal, not current eligibility.

## Code and checks

- Added data.py: strict frozen Pydantic schemas, filename/line errors, duplicate JSON-key/nonstandard-number rejection, immutable acquisition with known source checksums, metadata comparison, stable row IDs, normalized pair keys/audits, and model text columns with separate float labels.
- Added split.py: connected-profile groups, seeded stratification with disclosed coarsening, frozen IDs/checksum, persisted-manifest verification, and `assert_data_ready` that recomputes source leakage/contradictions rather than trusting an editable ready flag. S3 must separately enforce its human quality gate.
- Added explore.py and acquire/explore CLI commands, loader/split tests, README commands, audit/proposal docs, and JSON artifacts. Source text is never normalized before model input.
- `.venv/bin/python -m dating_compat acquire`: exit 0, ACQUISITION PASS. Initial curl inside sandbox failed DNS; approved pinned downloads succeeded. No remaining access blocker.
- `.venv/bin/python -m dating_compat explore`: reports produced; exit 1, DATA GATE BLOCKED: Train/eval leakage: 9 normalized unordered pairs overlap. Same failure is expected on identical offline reruns until source-preserving remediation is approved and implemented.
- `.venv/bin/python -m pytest -q`: exit 0, 38 passed. Includes strict labels/text, malformed lines, missing metadata, metadata mismatch, reversed/contradictory pairs, connected-profile grouping, no unsafe fallback, gate rejection, source/manifest tampering, immutable writes, model input fields, and JSON roundtrip.
- Saved JSON reloaded and validate_manifest(require_unblocked=False): SAVED MANIFEST PASS. assert_data_ready rejected the real cross-file leakage as expected. Corrected manifest preserves ALL original fit/validation IDs.
- Coding repair: first reload failed checksum because integer histogram keys changed to strings in JSON. Corrected before hashing, added a roundtrip test including sizes 1/2/11. Initial unverified reports preserved in reports/s2-initial-audit/; do NOT use those for training. This is a coding repair, not a training experiment or changed split policy.
- Current logical manifest digest: 9844c54bf0d8324c1957d5eed86f21b00f074dfd12cbc0c8efe9e22539bdd39a. File SHA differs by design (pretty JSON vs canonical payload digest).
- Artifact file SHA-256: acquisition 7ad987c67de8d66fee4a77be863a6d282dcd75c53bfcb9d435ec940db9ac1913; exploration 5a6b2ae6e0f7c6ea24f3e84bc974f3e0daca62494a20c58553e3c439e89f2761; split 2a8ea5ae8f21329c54c31f089e740f5b9c4cdb04eddc2dddd6e12e97fe842e7f; proposal ff860db95ca0fdb07098192fda5f142e5b30ef16e6b3a8c0aae3b8b80761100a.
- No training, model download, API inference, paid compute, or subscription changes. Acquisition/audit/test processes completed. Originals preserved; no learned metrics claimed.

## Saving and routing

Branch main; confirmed private GitHub https://github.com/Cloisterwood/mini-project-5.
Starting checkpoint b8f5f7be647b11de725c95606730a1f48223d7ec. Final reviewed WIP
checkpoint and remote SHA verification are recorded on CLO-44; this commit cannot
contain its own SHA. Linear moves to Agent Needs Input for the actual pending user
decision; it must not be marked Done until resolution is verified. S0 and S1 remain
complete. Preserve the frozen experiment contract and historical July plan.

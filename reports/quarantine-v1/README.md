# Active Stage 2 reports

User-approved S2-Q1 training quarantine. Exactly nine source-training IDs are excluded
from eligibility; all original files and all 1,469 evaluation rows remain unchanged.

- `cohort_manifest.json`: approval, source/policy hashes, excluded IDs, eligible IDs,
  unchanged evaluation IDs, counts, and post-quarantine overlap audit.
- `split_manifest.json`: eligible fit/validation IDs and grouped-split provenance.
- `exploration.json`: eligible training and original evaluation statistics, original
  metadata comparisons, and both original/post-quarantine pair audits.

Use `load_eligible_bundle()` from dating_compat.cohort for all later quality/training
work. Root reports remain historical original-source evidence. This structural pass
does not replace the S3 human quality gate.

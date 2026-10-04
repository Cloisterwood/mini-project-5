# Initial unverified S2 audit — do not use for training

Preserved before the persisted-manifest checksum repair. Integer histogram keys
became strings when saved to JSON, so the original manifest failed checksum
verification after reload. Current root reports use string keys before hashing.
The fit/validation IDs and data findings did not change. See docs/DATA_AUDIT.md.

"""Immutable acquisition, strict validation, IDs, and label-independent pair audit."""

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_validator

REVISION = "69b750dd024d5732ca19596d86bffc253abee92e"
SOURCE = "https://huggingface.co/datasets/dipenbhuva/dating-compatibility-pairs"
# Observed checksums of the original files acquired at the fixed revision in S2.
SOURCE_HASHES = {
    "dating_pairs.jsonl": "ccdffa2e004f0b36ff7102b23023f3a65bba2c7c02afb1d7029d03490e319302",
    "eval_pairs.jsonl": "0ef015ff913bdc21c2ac9897a0e67ebfee788d64b63fca98b195ce262401a8d2",
    "dating_pairs_metadata.json": "5e0486fc6068de27dd133dc4363dbdbc5501a635760f7a873783893a2c145d24",
    "eval_pairs_metadata.json": "66f963049c974ed8cc685c33ea5b7e191598fdc71501d758c990c4a756767013",
    "README.md": "d826049a1ff605cbb36f98cfc5aef368f0ee6b0a31eddfb0eadb47131fdf3ddb",
}
EXPECTED_COUNTS = {"train": 6000, "eval": 1469}


class DataError(ValueError):
    """Validation failure with source location or artifact context."""


class Pair(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    text_1: str
    text_2: str
    label: Annotated[StrictInt, Field(ge=0, le=1)]
    category: str
    subcategory: str
    pair_type: str

    @field_validator("text_1", "text_2", "category", "subcategory", "pair_type")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("required text must not be blank")
        return value  # Audit normalization must never change model input.


class Metadata(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    total_pairs: Annotated[StrictInt, Field(ge=0)]
    compatible_pairs: Annotated[StrictInt, Field(ge=0)]
    incompatible_pairs: Annotated[StrictInt, Field(ge=0)]
    dealbreaker_pairs: Annotated[StrictInt, Field(ge=0)]
    complex_pairs: Annotated[StrictInt, Field(ge=0)]
    llm_judged_pairs: Annotated[StrictInt, Field(ge=0)]
    realistic_pairs: Annotated[StrictInt, Field(ge=0)]
    subtle_mismatch_pairs: Annotated[StrictInt, Field(ge=0)]
    categories_used: list[str]

    @field_validator("categories_used")
    @classmethod
    def categories(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("categories must be nonblank and unique")
        return values


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_immutable(path: Path, value: dict) -> None:
    """Allow identical reruns, refuse replacement of a frozen artifact."""
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != rendered:
            raise DataError(f"{path}: immutable artifact differs; use a new report directory for a protocol revision")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json(text: str):
    def invalid(value):
        raise DataError(f"nonstandard JSON constant {value}")
    return json.loads(text, object_pairs_hook=_object, parse_constant=invalid)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def template_key(text: str) -> str:
    """Coarse surface-form proxy: first three words after a gender prefix."""
    text = re.sub(r"^(boy|girl)\s*:\s*", "", normalize(text))
    return " ".join(re.findall(r"\b[\w']+\b", text)[:3])


@dataclass(frozen=True)
class Record:
    id: str
    filename: str
    line: int
    pair: Pair

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted((normalize(self.pair.text_1), normalize(self.pair.text_2))))


def load_records(path: Path) -> list[Record]:
    records = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line, text in enumerate(stream, 1):
                try:
                    pair = Pair.model_validate(parse_json(text))
                except (ValueError, ValidationError) as exc:
                    raise DataError(f"{path.name}:{line}: {exc}") from exc
                identifier = digest([path.name, line, pair.model_dump()])
                records.append(Record(identifier, path.name, line, pair))
    except (OSError, UnicodeError) as exc:
        raise DataError(f"{path}: {exc}") from exc
    if not records:
        raise DataError(f"{path.name}:1: empty JSONL file")
    return records


def metadata_observed(records: list[Record]) -> dict:
    labels = Counter(record.pair.label for record in records)
    types = Counter(record.pair.pair_type for record in records)
    return {
        "total_pairs": len(records), "compatible_pairs": labels[1], "incompatible_pairs": labels[0],
        "dealbreaker_pairs": types["dealbreaker"],
        "complex_pairs": types["complex_compatible"] + types["complex_incompatible"],
        "llm_judged_pairs": types["llm_judged_compatible"] + types["llm_judged_incompatible"],
        "realistic_pairs": types["curated_realistic"], "subtle_mismatch_pairs": types["subtle_mismatch"],
        "categories_used": sorted({record.pair.category for record in records}),
    }


def model_columns(records: list[Record]) -> dict:
    """Only raw text features and a separate float target; never embed metadata."""
    return {"text_1": [r.pair.text_1 for r in records], "text_2": [r.pair.text_2 for r in records],
            "label": [float(r.pair.label) for r in records]}


def load_bundle(directory: Path) -> dict:
    result = {"records": {}, "metadata": {}, "hashes": {}, "metadata_comparison": {}, "blockers": []}
    for split, stem in (("train", "dating_pairs"), ("eval", "eval_pairs")):
        data_path, meta_path = directory / f"{stem}.jsonl", directory / f"{stem}_metadata.json"
        records = load_records(data_path)
        try:
            metadata = Metadata.model_validate(parse_json(meta_path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise DataError(f"{meta_path.name}: {exc}") from exc
        declared = metadata.model_dump()
        declared["categories_used"] = sorted(declared["categories_used"])
        observed = metadata_observed(records)
        differences = {key: {"declared": declared[key], "observed": observed[key]}
                       for key in observed if declared[key] != observed[key]}
        result["records"][split], result["metadata"][split] = records, metadata
        result["metadata_comparison"][split] = {"expected_count": EXPECTED_COUNTS[split],
            "observed_count": len(records), "differences": differences, "observed": observed}
        if differences:
            result["blockers"].append(f"{split}: metadata mismatch")
        if len(records) != EXPECTED_COUNTS[split]:
            result["blockers"].append(f"{split}: expected {EXPECTED_COUNTS[split]} rows, observed {len(records)}")
        for path in (data_path, meta_path):
            result["hashes"][path.name] = file_hash(path)
    return result


def pair_audit(records: list[Record]) -> dict:
    exact, unordered, normalized = defaultdict(list), defaultdict(list), defaultdict(list)
    for record in records:
        a, b = record.pair.text_1, record.pair.text_2
        exact[(a, b)].append(record)
        unordered[tuple(sorted((a, b)))].append(record)
        normalized[record.key].append(record)
    repeated = lambda groups: sum(len(group) - 1 for group in groups.values())
    groups = [{"pair_key_sha256": digest(key), "ids": [r.id for r in group],
               "lines": [r.line for r in group], "labels": sorted({r.pair.label for r in group})}
              for key, group in sorted(normalized.items()) if len(group) > 1]
    return {"exact_ordered_duplicate_extras": repeated(exact),
        "raw_unordered_duplicate_extras": repeated(unordered),
        "reversed_duplicate_extras": repeated(unordered) - repeated(exact),
        "normalized_unordered_duplicate_extras": repeated(normalized),
        "normalization_only_duplicate_extras": repeated(normalized) - repeated(unordered),
        "duplicate_groups": groups, "contradictory_groups": [g for g in groups if len(g["labels"]) > 1]}


def cross_audit(left: list[Record], right: list[Record]) -> dict:
    a, b = defaultdict(list), defaultdict(list)
    for record in left:
        a[record.key].append(record)
    for record in right:
        b[record.key].append(record)
    overlaps = [{"pair_key_sha256": digest(key), "left_ids": [r.id for r in a[key]],
        "right_ids": [r.id for r in b[key]], "left_lines": [r.line for r in a[key]],
        "right_lines": [r.line for r in b[key]],
        "labels": sorted({r.pair.label for r in a[key] + b[key]})} for key in sorted(a.keys() & b.keys())]
    profiles = lambda rows: {normalize(text) for r in rows for text in (r.pair.text_1, r.pair.text_2)}
    templates = lambda rows: {template_key(text) for r in rows for text in (r.pair.text_1, r.pair.text_2)}
    return {"overlapping_pair_count": len(overlaps), "overlapping_pairs": overlaps,
        "shared_normalized_profiles": len(profiles(left) & profiles(right)),
        "shared_template_proxies": len(templates(left) & templates(right)),
        "template_proxy_definition": "First three word tokens after removing boy:/girl:; coarse surface-form proxy, not proven template identity"}


def acquire(directory: Path, reports: Path) -> dict:
    """Download only the frozen revision; never replace an existing source file."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, expected in SOURCE_HASHES.items():
        path = directory / name
        url = f"{SOURCE}/resolve/{REVISION}/{name}"
        if not path.exists():
            with urlopen(url, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise DataError(f"{name}: downloaded checksum differs from pinned acquisition")
            with path.open("xb") as stream:
                stream.write(content)
        if file_hash(path) != expected:
            raise DataError(f"{name}: existing checksum differs; original left unchanged")
        files[name] = {"source_url": url, "sha256": expected, "bytes": path.stat().st_size}
    card = (directory / "README.md").read_text(encoding="utf-8")
    if not re.search(r"(?m)^license: apache-2\.0$", card):
        raise DataError("Pinned dataset card does not declare the expected Apache-2.0 license")
    path = reports / "acquisition_manifest.json"
    acquired = parse_json(path.read_text())["acquired_at_utc"] if path.exists() else datetime.now(timezone.utc).isoformat()
    manifest = {"source": SOURCE, "revision": REVISION, "acquired_at_utc": acquired,
        "license": {"declared": "Apache-2.0", "evidence": "Pinned dataset card YAML", "card_sha256": SOURCE_HASHES["README.md"]},
        "checksum_origin": "Observed original bytes acquired at pinned revision during S2; not a separate publisher signature",
        "files": files}
    write_immutable(path, manifest)
    return manifest


def verify_acquisition(directory: Path, reports: Path) -> str:
    path = reports / "acquisition_manifest.json"
    try:
        manifest = parse_json(path.read_text(encoding="utf-8"))
        if manifest["revision"] != REVISION or manifest["source"] != SOURCE:
            raise DataError("Acquisition source/revision mismatch")
        for name, expected in SOURCE_HASHES.items():
            if manifest["files"][name]["sha256"] != expected or file_hash(directory / name) != expected:
                raise DataError(f"{name}: acquisition/source checksum mismatch")
        if manifest["license"]["declared"] != "Apache-2.0":
            raise DataError("Acquisition license declaration differs")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError(f"{path}: invalid or missing acquisition provenance: {exc}") from exc
    return file_hash(path)

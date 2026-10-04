"""Resumable terminal entry of actual human review responses."""

import csv
import os
import tempfile
from pathlib import Path

from dating_compat.data import DataError, file_hash
from dating_compat.review import COLUMNS, RATING_FIELDS, RATINGS, read_review

QUESTIONS = {
    "label_agreement": "Does the supplied compatible/incompatible label follow the texts? Different hobbies alone need not mean incompatibility.",
    "naturalness": "Would a person plausibly say these sentences, with fluent wording?",
    "coherence": "Is each profile understandable and internally consistent? Different preferences can coexist.",
    "preference_realism": "Are the preferences and combinations plausible for real people?",
    "hierarchy_consistency": "Does the label respect priorities: dealbreakers, core values, lifestyle, interests, then activities? Use ambiguous when priorities or conflict are unclear.",
    "clarity": "Is there enough information to judge without unresolved ambiguity or contradictory cues?",
}
CHOICES = {"1": "acceptable", "2": "ambiguous", "3": "unacceptable", **{r: r for r in RATINGS}}


class StopReview(Exception):
    """The human chose to leave without saving the current pair."""


def complete(row: dict) -> bool:
    return (all(row[field] in RATINGS for field in RATING_FIELDS)
            and bool(row["reviewer"].strip()) and row["human_reviewed"] == "yes")


def save_rows(path: Path, rows: list[dict], previous_hash: str) -> str:
    """Replace atomically; refuse to overwrite edits made since loading."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, prefix=".human-review-",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        if file_hash(path) != previous_hash:
            raise DataError("CSV changed during this session. Restart to load those changes; nothing was overwritten.")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return file_hash(path)


def run_review(path: Path, rows: list[dict], previous_hash: str, pair_number: int | None = None,
               input_fn=input, output_fn=print) -> int:
    """Prompt one pair at a time, committing only explicitly confirmed human input."""
    def ask(prompt):
        value = input_fn(prompt).strip()
        if value.lower() in {"q", "quit"}:
            raise StopReview
        return value

    def required(prompt):
        while True:
            value = ask(prompt)
            if value:
                return value
            output_fn("Please enter a value, or q to quit.")

    if pair_number is not None and not 1 <= pair_number <= len(rows):
        raise DataError(f"Pair number must be between 1 and {len(rows)}")
    indices = ([pair_number - 1] if pair_number is not None else
               [i for i, row in enumerate(rows) if not complete(row)])
    if not indices:
        output_fn("All pairs already have human responses. Use --pair NUMBER to revisit one.")
        return 0
    output_fn(f"Human review: {sum(complete(row) for row in rows)}/{len(rows)} pairs complete.")
    output_fn("1 = acceptable; 2 = ambiguous; 3 = unacceptable. Enter q at any prompt to quit.")
    output_fn("Acceptable = defensible; ambiguous = insufficient/conflicting evidence; unacceptable = clear disagreement or quality problem.")
    output_fn("Each confirmed pair is saved. An unfinished pair is discarded when you quit.")
    output_fn("Assignment hierarchy weights: dealbreakers 10; core values 8–10; lifestyle 5–7; interests 3–5; activities 1–3. These are assignment assumptions.")
    try:
        reviewer = required("Your reviewer name or alias: ")
        for index in indices:
            row = rows[index].copy()
            output_fn(f"\nPair {row['sample_number']} of {len(rows)}")
            output_fn(f"Profile 1: {row['text_1']}\nProfile 2: {row['text_2']}")
            label = "compatible" if row["label"] == "1" else "incompatible"
            output_fn(f"Supplied label: {label} ({row['label']})")
            output_fn(f"Metadata: {row['category']}; {row['subcategory']}; {row['pair_type']}. Judge the supplied label, not pair type.")
            for field in RATING_FIELDS:
                output_fn(f"\n{field.replace('_', ' ').capitalize()}: {QUESTIONS[field]}")
                current = row[field]
                while True:
                    value = ask(f"Rating 1/2/3{f' [Enter keeps {current}]' if current else ''}: ").lower()
                    if not value and current:
                        value = current
                    if value in CHOICES:
                        row[field] = CHOICES[value]
                        break
                    output_fn("Enter 1, 2, 3, or the full rating word.")
            needs_note = any(row[field] != "acceptable" for field in RATING_FIELDS)
            if row["notes"]:
                output_fn(f"Existing notes: {row['notes']}")
            while True:
                note = ask("Notes (required for ambiguous/unacceptable; Enter keeps existing notes): ")
                row["notes"] = note or row["notes"]
                if not needs_note or row["notes"].strip():
                    break
                output_fn("Explain your ambiguous or unacceptable judgment before saving.")
            output_fn("\nYour responses:")
            for field in RATING_FIELDS:
                output_fn(f"  {field}: {row[field]}")
            output_fn(f"  Notes: {row['notes'] or '(none)'}\n  Reviewer: {reviewer}")
            while True:
                decision = ask("Type yes to confirm these are your human judgments and save, or no to discard this pair: ").lower()
                if decision in {"yes", "no"}:
                    break
                output_fn("Enter yes or no (or q to quit).")
            if decision == "no":
                output_fn("Pair discarded; its existing CSV responses are unchanged.")
                continue
            row.update(reviewer=reviewer, human_reviewed="yes")
            updated = rows.copy()
            updated[index] = row
            previous_hash = save_rows(path, updated, previous_hash)
            rows = updated
            output_fn(f"SAVED pair {row['sample_number']} — {sum(complete(r) for r in rows)}/{len(rows)} complete.")
    except (StopReview, EOFError, KeyboardInterrupt):
        output_fn("\nReview paused. Previously saved pairs are preserved; rerun the command to resume.")
        return 0
    if all(complete(row) for row in rows):
        output_fn("HUMAN REVIEW COMPLETE. Next run: .venv/bin/python -m dating_compat quality")
    else:
        output_fn("Review remains incomplete. Rerun the command to fill unfinished pairs.")
    return 0


def guided_review(data_dir: Path, response_path: Path, pair_number: int | None = None) -> int:
    from dating_compat.cohort import load_eligible_bundle
    from dating_compat.data import digest, parse_json
    from dating_compat.quality import load_rubric
    from dating_compat.review import select_review

    bundle = load_eligible_bundle(data_dir)
    rubric, rubric_hash = load_rubric()
    sample = parse_json(Path("reports/human_review_sample.json").read_text(encoding="utf-8"))
    if digest(sample) != digest(select_review(bundle, rubric, rubric_hash)):
        raise DataError("Frozen human sample differs from the eligible fit sample")
    review = read_review(bundle, sample, response_path)
    if not review["rows"]:
        raise DataError("; ".join(review["errors"]))
    # read_review emits this sole status message for a valid, partially filled sheet.
    if review["errors"] and review["errors"] != [
        "Human review pending: all six ratings, reviewer, and human_reviewed=yes required on every row"
    ]:
        raise DataError("; ".join(review["errors"]))
    return run_review(response_path, review["rows"], review["sha256"], pair_number)

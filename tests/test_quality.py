import csv
import json
from pathlib import Path

import pytest

from dating_compat import quality
from dating_compat.data import DataError, Pair, Record, digest
from dating_compat.review import COLUMNS, RATING_FIELDS, prepare_review, read_review, select_review
from dating_compat.review_guide import run_review


def guide_inputs(values):
    answers = iter(values)
    return lambda prompt: next(answers)


def test_guide_saves_real_inputs_requires_notes_and_resumes(setup_review):
    bundle, _, _, sample, path = setup_review
    before = read_review(bundle, sample, path)
    output = []
    run_review(path, before["rows"], before["sha256"],
               input_fn=guide_inputs(["TEST HUMAN", "bad", "2", *["1"] * 5,
                                      "", "Priority unclear", "yes", "q"]), output_fn=output.append)
    saved = read_review(bundle, sample, path)
    assert saved["rated_rows"] == 1
    assert saved["rows"][0]["label_agreement"] == "ambiguous"
    assert saved["rows"][0]["notes"] == "Priority unclear"
    assert saved["rows"][0]["human_reviewed"] == "yes"
    assert all(saved["rows"][i][key] == before["rows"][i][key]
               for i in range(20) for key in COLUMNS if key not in (*RATING_FIELDS, "notes", "reviewer", "human_reviewed"))
    assert any("Explain your" in line for line in output)
    output.clear()
    run_review(path, saved["rows"], saved["sha256"], input_fn=guide_inputs(["TEST HUMAN", "q"]), output_fn=output.append)
    assert "\nPair 2 of 20" in output
    assert "\nPair 1 of 20" not in output


@pytest.mark.parametrize("ending", ["q", "no"])
def test_guide_does_not_save_without_confirmation(setup_review, ending):
    bundle, _, _, sample, path = setup_review
    before = read_review(bundle, sample, path)
    run_review(path, before["rows"], before["sha256"], pair_number=1,
               input_fn=guide_inputs(["TEST HUMAN", *["1"] * 6, "", ending]), output_fn=lambda line: None)
    assert read_review(bundle, sample, path)["sha256"] == before["sha256"]


def test_guide_edit_preserves_existing_answers_and_other_pairs(setup_review):
    bundle, _, _, sample, path = setup_review
    edit_csv(path, complete_fixture)
    before = read_review(bundle, sample, path)
    run_review(path, before["rows"], before["sha256"], pair_number=7,
               input_fn=guide_inputs(["SECOND TEST HUMAN", *[""] * 6, "", "yes"]), output_fn=lambda line: None)
    after = read_review(bundle, sample, path)
    assert after["complete"]
    assert after["rows"][6]["reviewer"] == "SECOND TEST HUMAN"
    assert after["rows"][:6] == before["rows"][:6]
    assert after["rows"][7:] == before["rows"][7:]


def test_guide_preserves_external_changes(setup_review):
    bundle, _, _, sample, path = setup_review
    before = read_review(bundle, sample, path)
    edit_csv(path, lambda rows: rows[1].update(notes="External human edit"))
    changed = path.read_bytes()
    with pytest.raises(DataError, match="CSV changed"):
        run_review(path, before["rows"], before["sha256"], pair_number=1,
                   input_fn=guide_inputs(["TEST HUMAN", *["1"] * 6, "", "yes"]), output_fn=lambda line: None)
    assert path.read_bytes() == changed
    assert not list(path.parent.glob(".human-review-*"))


def test_guide_eof_preserves_csv(setup_review):
    bundle, _, _, sample, path = setup_review
    before = read_review(bundle, sample, path)
    def eof(prompt):
        raise EOFError
    run_review(path, before["rows"], before["sha256"], input_fn=eof, output_fn=lambda line: None)
    assert read_review(bundle, sample, path)["sha256"] == before["sha256"]


def test_guided_review_rejects_tampered_sample_and_csv(setup_review, monkeypatch):
    from dating_compat import cohort, review_guide
    bundle, _, _, sample, path = setup_review
    monkeypatch.setattr(cohort, "load_eligible_bundle", lambda data_dir: bundle)
    original_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs:
                        json.dumps(sample) if self == Path("reports/human_review_sample.json")
                        else original_read(self, **kwargs))
    edit_csv(path, lambda rows: rows[0].update(text_1="Changed profile"))
    with pytest.raises(DataError, match="metadata changed"):
        review_guide.guided_review(Path("unused"), path)
    sample["selected_ids"].reverse()
    with pytest.raises(DataError, match="Frozen human sample"):
        review_guide.guided_review(Path("unused"), path)


@pytest.fixture
def setup_review(tmp_path):
    rubric, rubric_hash = quality.load_rubric()
    rows = []
    for i in range(180):
        first, second = ("girl", "boy") if (i // 2) % 2 else ("boy", "girl")
        pair = Pair(text_1=f"{first}: I enjoy quiet walks profile{i}" + " very" * ((i // 2) % 4),
                    text_2=f"{second}: I like live music partner{i}", label=i % 2,
                    category=rubric["required_categories"][(i // 2) % 5], subcategory="example",
                    pair_type=rubric["required_pair_types"][(i // 2) % 9])
        rows.append(Record(digest([i, pair.model_dump()]), "dating_pairs.jsonl", i + 1, pair))
    bundle = {"records": {"train": rows}, "hashes": {"train": "source"},
        "cohort_manifest": {"manifest_sha256": "cohort", "eligible_label_counts": {"0": 90, "1": 90}},
        "split_manifest": {"manifest_sha256": "split", "fit_ids": [r.id for r in rows]}}
    response = tmp_path / "responses.csv"
    sample = prepare_review(bundle, rubric, rubric_hash, tmp_path, response)
    return bundle, rubric, rubric_hash, sample, response


def edit_csv(path, edit):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    edit(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def complete_fixture(rows, rating="acceptable"):
    for row in rows:
        row.update({field: rating for field in RATING_FIELDS})
        row.update(reviewer="UNIT TEST FIXTURE", human_reviewed="yes", notes="Synthetic fixture only" if rating != "acceptable" else "")


@pytest.mark.parametrize("value, expected", [(0, 5), (.025, 2.5), (.05, 0), (.2, 0)])
def test_duplicate_linear_scoring(value, expected):
    assert quality.score_value(value, {"method": "lower_better", "good": 0, "bad": .05}) == pytest.approx(expected)


@pytest.mark.parametrize("value, expected", [(0, 0), (.2, 2.5), (.4, 5), (.5, 5), (.6, 5), (.8, 2.5), (1, 0)])
def test_label_balance_scoring(value, expected):
    assert quality.score_value(value, {"method": "label_balance"}) == pytest.approx(expected)


def test_undefined_values_and_phi():
    assert quality.score_value(None, {"method": "acceptable_fraction"}) is None
    with pytest.raises(DataError, match="finite"):
        quality.score_value(float("nan"), {"method": "acceptable_fraction"})
    assert quality.phi(10, 0, 0, 10) == 1
    assert quality.phi(5, 5, 5, 5) == 0
    assert quality.phi(0, 0, 5, 5) is None


def test_sample_is_seeded_fit_only_diverse_and_unfilled(setup_review):
    bundle, rubric, rubric_hash, sample, response = setup_review
    assert sample == select_review(bundle, rubric, rubric_hash)
    assert len(sample["selected_ids"]) == 20 and len(set(sample["selected_ids"])) == 20
    assert set(sample["selected_ids"]) <= set(bundle["split_manifest"]["fit_ids"])
    assert len(sample["pair_type_counts"]) == 9 and min(sample["pair_type_counts"].values()) >= 2
    review = read_review(bundle, sample, response)
    assert not review["complete"] and review["rated_rows"] == 0
    assert all(not row[field] for row in review["rows"] for field in RATING_FIELDS)


def test_missing_review_never_passes(setup_review):
    bundle, rubric, rubric_hash, sample, response = setup_review
    response.unlink()
    report = quality.make_quality_report(bundle, rubric, rubric_hash, sample, read_review(bundle, sample, response))
    assert not report["gate"]["passed"]
    assert report["overall_score_out_of_100"] is None
    assert report["dimensions"]["real_life_matching"]["score_out_of_20"] is None


def test_ambiguous_is_not_acceptable_and_human_failure_blocks(setup_review):
    bundle, rubric, rubric_hash, sample, response = setup_review
    edit_csv(response, lambda rows: complete_fixture(rows, "ambiguous"))
    review = read_review(bundle, sample, response)
    assert review["complete"]
    report = quality.make_quality_report(bundle, rubric, rubric_hash, sample, review)
    assert report["gate"]["status"] == "failed"
    assert report["dimensions"]["real_life_matching"]["score_out_of_20"] == 0


@pytest.mark.parametrize("change", ["text", "unknown_rating", "no_notes", "no_reviewer", "no_attestation"])
def test_review_cannot_change_sample_or_fake_completion(setup_review, change):
    bundle, _, _, sample, response = setup_review
    def alter(rows):
        complete_fixture(rows)
        if change == "text": rows[0]["text_1"] = "Replaced profile"
        elif change == "unknown_rating": rows[0]["naturalness"] = "true"
        elif change == "no_notes": rows[0]["clarity"] = "ambiguous"
        elif change == "no_reviewer": rows[0]["reviewer"] = ""
        else: rows[0]["human_reviewed"] = ""
    edit_csv(response, alter)
    assert not read_review(bundle, sample, response)["complete"]


def test_rubric_change_cannot_weaken_gate(tmp_path):
    rubric = tmp_path / "rubric.yaml"
    rubric.write_bytes(Path("configs/quality-rubric.yaml").read_bytes())
    frozen = tmp_path / "freeze.json"
    frozen.write_bytes(Path("reports/quality_rubric_manifest.json").read_bytes())
    quality.load_rubric(rubric, frozen)
    rubric.write_text(rubric.read_text().replace("overall_minimum: 60", "overall_minimum: 1"))
    with pytest.raises(DataError, match="Frozen quality rubric"):
        quality.load_rubric(rubric, frozen)


def test_gate_recomputes_review_and_rejects_forged_or_stale_pass(setup_review, tmp_path, monkeypatch):
    bundle, rubric, rubric_hash, sample, response = setup_review
    monkeypatch.setattr(quality, "load_eligible_bundle", lambda _: bundle)
    path = tmp_path / "quality.json"
    def save():
        report = quality.make_quality_report(bundle, rubric, rubric_hash, sample, read_review(bundle, sample, response))
        path.write_text(json.dumps(report))
        return report
    save()
    with pytest.raises(quality.QualityGateError, match="pending_human_review"):
        quality.require_quality_gate(report_path=path, response_path=response, sample_path=tmp_path / "human_review_sample.json")
    edit_csv(response, lambda rows: complete_fixture(rows))
    save()
    assert quality.require_quality_gate(report_path=path, response_path=response, sample_path=tmp_path / "human_review_sample.json")["gate"]["passed"]
    edit_csv(response, lambda rows: complete_fixture(rows, "unacceptable"))
    with pytest.raises(quality.QualityGateError, match="stale or modified"):
        quality.require_quality_gate(report_path=path, response_path=response, sample_path=tmp_path / "human_review_sample.json")
    failed = save()
    with pytest.raises(quality.QualityGateError, match="failed"):
        quality.require_quality_gate(report_path=path, response_path=response, sample_path=tmp_path / "human_review_sample.json")
    failed["gate"]["passed"] = True
    failed["gate"]["status"] = "passed"
    path.write_text(json.dumps(failed))
    with pytest.raises(quality.QualityGateError, match="stale or modified"):
        quality.require_quality_gate(report_path=path, response_path=response, sample_path=tmp_path / "human_review_sample.json")

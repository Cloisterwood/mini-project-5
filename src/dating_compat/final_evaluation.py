"""One frozen final evaluation, sealed predictions and paired comparison."""

import csv
import io
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from dating_compat.cluster import cluster_purity, pair_scores, project_and_cluster
from dating_compat.data import DataError, SOURCE_HASHES, digest, file_hash, load_records, parse_json, write_immutable
from dating_compat.evaluate import checked_predictions, embedding_cache, verify_seal, write_bytes_immutable
from dating_compat.metrics import breakdown, pair_metrics, target_results
from dating_compat.selection import freeze_selection
from dating_compat.train import _model, state_hash
from dating_compat.uncertainty import paired_bootstrap
from dating_compat.validation import signed_deltas


def error_analysis(records, scores, baseline, config):
    rows = []
    support = {}
    for kind in sorted({r.pair.pair_type for r in records}):
        group = [(r,float(s),float(b)) for r,s,b in zip(records,scores,baseline) if r.pair.pair_type == kind]
        support[kind] = {"n": len(group), "positive": sum(r.pair.label for r,_,_ in group),
            "negative": sum(1-r.pair.label for r,_,_ in group),
            "fp": sum(r.pair.label==0 and s>=.5 for r,s,_ in group), "fn": sum(r.pair.label==1 and s<.5 for r,s,_ in group)}
        for error in ("false_positive", "false_negative"):
            candidates = [(r,s,b) for r,s,b in group if (r.pair.label==0 and s>=.5 if error=="false_positive" else r.pair.label==1 and s<.5)]
            for r,s,b in candidates[:config["error_examples_per_type_and_error"]]:
                rows.append({"record_id":r.id,"source_line":r.line,"error":error,"label":r.pair.label,"cosine":s,"baseline_cosine":b,
                    "category":r.pair.category,"subcategory":r.pair.subcategory,"pair_type":kind,
                    "text_1_excerpt":r.pair.text_1[:500],"text_2_excerpt":r.pair.text_2[:500],
                    "explanation":"Cosine crossed the fixed .5 threshold despite incompatible source label" if error=="false_positive" else "Cosine stayed below the fixed .5 threshold despite compatible source label"})
    return {"status":"post_selection_descriptive_audit","selection":"First source-order examples, up to three per pair_type/error; not an exhaustive or random error sample",
        "support":support,"examples":rows,"limitations":["Source labels are synthetic; discrepancies do not establish real-world compatibility", "Threshold explanations are observable, not causal attribution", "No training or threshold changes follow this analysis"]}


def final_evaluation():
    if not Path("reports/selected_run_manifest.json").exists():
        raise DataError("Existing frozen selection required before final evaluation")
    selected = freeze_selection()  # guarded validation only; cannot create a new choice here
    config = parse_json(Path("configs/final-evaluation.json").read_text())
    baseline_config = parse_json(Path("configs/baseline.json").read_text())
    receipt = parse_json((Path("reports/runs")/selected["selected_run"]/"selected_checkpoint_manifest.json").read_text())
    model_path = Path(selected["selected_path"])
    files = {str(p.relative_to(model_path)):file_hash(p) for p in sorted(model_path.rglob("*")) if p.is_file()}
    if files != receipt["selected_files"]:
        raise DataError("Selected model files changed")
    binding = {"selection_file_sha256":file_hash(Path("reports/selected_run_manifest.json")),
        "selected_weights_sha256":selected["selected_weights_sha256"],"selected_files":files,
        "final_config_sha256":file_hash(Path("configs/final-evaluation.json")),"baseline_config_sha256":file_hash(Path("configs/baseline.json"))}
    write_immutable(Path("reports/final_pre_evaluation_manifest.json"),binding)
    # Only after this selection/model receipt is saved do we open evaluation files.
    source = Path("data/eval_pairs.jsonl")
    if file_hash(source) != SOURCE_HASHES[source.name]:
        raise DataError("Final evaluation source differs from pinned original")
    seal = parse_json(Path("reports/baseline_manifest.json").read_text())
    verify_seal(Path("reports/baseline_manifest.json"),seal["inputs"])
    binding["evaluation_source_sha256"] = file_hash(source)
    binding["baseline_seal_sha256"] = file_hash(Path("reports/baseline_manifest.json"))
    output = Path("reports/final_evaluation_manifest.json")
    if output.exists():
        saved = parse_json(output.read_text())
        if saved["binding"] != binding or digest({k:v for k,v in saved.items() if k!="manifest_sha256"}) != saved["manifest_sha256"]:
            raise DataError("Final evaluation seal differs")
        for name,sha in saved["artifacts"].items():
            if file_hash(Path(name)) != sha:
                raise DataError("Final evaluation artifact changed")
        return saved
    started = time.monotonic()
    records = load_records(source)
    base = parse_json(Path("reports/baseline_metrics.json").read_text())
    with Path("reports/baseline/baseline_eval_predictions.csv").open(newline="") as stream:
        base_rows = list(csv.DictReader(stream))
    ids = [r.id for r in records]
    if base["ordered_record_ids_sha256"] != digest(ids) or [r["record_id"] for r in base_rows] != ids or [int(r["label"]) for r in base_rows] != [r.pair.label for r in records]:
        raise DataError("Final evaluation and baseline IDs/order/labels differ")
    labels = [r.pair.label for r in records]
    base_scores = np.array([float(r["cosine"]) for r in base_rows])
    if pair_metrics(labels,base_scores) != base["metrics"] or base["config"] != baseline_config:
        raise DataError("Baseline final metrics/config mismatch")
    software = {n:version(n) for n in base["execution"]["software"]}
    if software != base["execution"]["software"]:
        raise DataError("Final software differs from baseline")
    model = _model(selected["selected_config"],"cpu",model_path)
    if state_hash(model) != selected["selected_weights_sha256"]:
        raise DataError("Loaded selected weights differ")
    first,second,cache = embedding_cache(model,records,baseline_config,{"manifest_sha256":digest(files)},Path(".cache/embeddings"),software)
    scores = pair_scores(first,second)
    xy,clusters = project_and_cluster(first,second,baseline_config)
    predictions = checked_predictions(records,scores,xy,clusters)
    metrics = pair_metrics(labels,scores); clustering = cluster_purity(labels,clusters)
    provenance = {"evaluation_source_sha256":file_hash(source),"ordered_record_ids_sha256":digest(ids),"selected_run":selected["selected_run"],"selected_weights_sha256":selected["selected_weights_sha256"],"selection_manifest_sha256":selected["manifest_sha256"]}
    fine = {"version":"s7-final-v1","split":"eval","provenance":provenance,"config":baseline_config,"embedding_cache":cache,"metrics":metrics,"clustering":clustering,"breakdowns":breakdown(records,scores),"targets":target_results(metrics,clustering,baseline_config["target_tiers"]),"limitations":base["limitations"]+["Selected using validation only; final evaluation never triggers tuning"]}
    uncertainty = paired_bootstrap(records,base_scores,scores,config)
    comparison = {"version":"s7-comparison-v1","provenance":provenance,"baseline_provenance":{"evaluation_source_sha256":base["inputs"]["source_hashes"][source.name],"ordered_record_ids_sha256":base["ordered_record_ids_sha256"]},"classification_deltas":signed_deltas(base["metrics"],metrics),"purity_comparison":{"before":base["clustering"]["coverage_adjusted_purity"],"after":clustering["coverage_adjusted_purity"],"delta":clustering["coverage_adjusted_purity"]-base["clustering"]["coverage_adjusted_purity"]},"targets":{"baseline":base["targets"],"finetuned":fine["targets"]},"breakdowns":{"baseline":base["breakdowns"],"finetuned":fine["breakdowns"]},"uncertainty":uncertainty,"policy":"Final post-selection results; no changes to training, threshold or clustering"}
    artifacts = {Path("reports/finetuned_metrics.json"):fine,Path("reports/comparison.json"):comparison,Path("reports/final_error_analysis.json"):error_analysis(records,scores,base_scores,config)}
    for path,value in artifacts.items():write_immutable(path,value)
    buffer=io.StringIO(newline=""); writer=csv.DictWriter(buffer,fieldnames=list(predictions[0]),lineterminator="\n");writer.writeheader();writer.writerows(predictions)
    csv_path=Path("reports/final/finetuned_eval_predictions.csv");write_bytes_immutable(csv_path,buffer.getvalue().encode())
    result={"binding":binding,"provenance":provenance,"artifacts":{str(p):file_hash(p) for p in [*artifacts,csv_path]},"elapsed_seconds":time.monotonic()-started,"software":software,"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"code_sha256":{n:file_hash(Path(__file__).with_name(n)) for n in ("final_evaluation.py","uncertainty.py","metrics.py","cluster.py")}}
    result["manifest_sha256"]=digest(result);write_immutable(output,result)
    return result

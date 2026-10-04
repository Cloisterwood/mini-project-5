"""Offline delivery verification; no training, evaluation inference or resampling."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dating_compat.cluster import cluster_purity
from dating_compat.data import digest, file_hash, write_immutable
from dating_compat.metrics import breakdown, pair_metrics, target_results
from dating_compat.train import _model, state_hash
from dating_compat.training_data import load_training_inputs
from dating_compat.validation import signed_deltas


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--regenerated-figures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    selected=json.loads(Path('reports/selected_run_manifest.json').read_text())
    release=json.loads(Path('reports/model_release_manifest.json').read_text())
    files={str(p.relative_to(args.model_dir)):file_hash(p) for p in args.model_dir.rglob('*') if p.is_file()}
    assert files==release['selected_files'], 'Restored model file hashes differ'
    inputs,_=load_training_inputs()
    texts=[r['text_1'] for r in inputs['fit'][:8]]
    original=_model(selected['selected_config'],'cpu',Path(selected['selected_path']))
    restored=_model(selected['selected_config'],'cpu',args.model_dir)
    assert state_hash(original)==state_hash(restored)==selected['selected_weights_sha256']
    a=original.encode(texts,normalize_embeddings=True,convert_to_numpy=True)
    b=restored.encode(texts,normalize_embeddings=True,convert_to_numpy=True)
    maxdiff=float(np.max(np.abs(a-b)))
    assert maxdiff==0 and np.isfinite(b).all()
    reports=[]; metrics=[]
    root=Path('.cache/reproduced-reports')
    for report_name,csv_name in [('baseline_metrics.json','reports/baseline/baseline_eval_predictions.csv'),('finetuned_metrics.json','reports/final/finetuned_eval_predictions.csv')]:
        saved=json.loads((Path('reports')/report_name).read_text())
        with Path(csv_name).open(newline='') as stream: rows=list(csv.DictReader(stream))
        records=[SimpleNamespace(pair=SimpleNamespace(label=int(r['label']),category=r['category'],pair_type=r['pair_type'])) for r in rows]
        y=[int(r['label']) for r in rows]; scores=[float(r['cosine']) for r in rows]
        rebuilt={**saved,'metrics':pair_metrics(y,scores),'breakdowns':breakdown(records,scores),'clustering':cluster_purity(y,[int(r['cluster_id']) for r in rows])}
        rebuilt['targets']=target_results(rebuilt['metrics'],rebuilt['clustering'],rebuilt['config']['target_tiers'])
        assert rebuilt==saved, 'Regenerated metrics differ'
        write_immutable(root/report_name,rebuilt);reports.append(report_name);metrics.append(rebuilt)
    saved=json.loads(Path('reports/comparison.json').read_text())
    rebuilt={**saved,'classification_deltas':signed_deltas(metrics[0]['metrics'],metrics[1]['metrics']),
        'breakdowns':{'baseline':metrics[0]['breakdowns'],'finetuned':metrics[1]['breakdowns']},
        'purity_comparison':{'before':metrics[0]['clustering']['coverage_adjusted_purity'],'after':metrics[1]['clustering']['coverage_adjusted_purity'],'delta':metrics[1]['clustering']['coverage_adjusted_purity']-metrics[0]['clustering']['coverage_adjusted_purity']}}
    assert rebuilt==saved;write_immutable(root/'comparison.json',rebuilt);reports.append('comparison.json')
    report_hashes={n:file_hash(root/n) for n in reports}
    assert all(sha==file_hash(Path('reports')/n) for n,sha in report_hashes.items())
    figure_hashes={}
    for p in Path('figures/submission').glob('*.png'):
        sha=file_hash(args.regenerated_figures/p.name); assert sha==file_hash(p),'Regenerated PNG differs';figure_hashes[p.name]=sha
    assert len(figure_hashes)==6
    receipt={'status':'pass','model_dir':str(args.model_dir),'selected_weights_sha256':state_hash(restored),
        'fit_probe_count':len(texts),'fit_probe_texts_sha256':digest(texts),'maximum_absolute_embedding_difference':maxdiff,
        'regenerated_report_hashes':report_hashes,'regenerated_figure_hashes':figure_hashes,
        'uncertainty':'Saved paired bootstrap intervals reused, not recomputed','evaluation_inference':False,
        'script_sha256':file_hash(Path(__file__))}
    write_immutable(args.output,receipt)
    print('DELIVERY VERIFICATION PASS: restored model, three reports, six PNGs')


if __name__=='__main__':main()

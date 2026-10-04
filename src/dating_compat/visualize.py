"""Six assignment figures drawn exclusively from sealed predictions and reports."""

import csv
import io
from pathlib import Path

import numpy as np

from dating_compat.data import DataError, digest, file_hash, parse_json, write_immutable
from dating_compat.evaluate import write_bytes_immutable
from dating_compat.metrics import pair_metrics

COLORS = ("#0072B2", "#D55E00")
NAMES = ("similarity_distribution_before_after", "umap_before_after", "category_effect_size",
         "roc_curves", "training_loss_curve", "improvement_delta")


def verified_inputs():
    sources = set()
    for name in ("reports/baseline_manifest.json", "reports/final_evaluation_manifest.json"):
        path = Path(name); seal = parse_json(path.read_text())
        if digest({k:v for k,v in seal.items() if k!="manifest_sha256"}) != seal["manifest_sha256"]:
            raise DataError("Figure source seal digest differs")
        for artifact, sha in seal["artifacts"].items():
            if file_hash(Path(artifact)) != sha:
                raise DataError("Figure source artifact changed")
            sources.add(artifact)
        sources.add(name)
    fine = parse_json(Path("reports/finetuned_metrics.json").read_text())
    base = parse_json(Path("reports/baseline_metrics.json").read_text())
    comparison = parse_json(Path("reports/comparison.json").read_text())
    run = Path("reports/runs") / fine["provenance"]["selected_run"]
    seal = parse_json((run/"validation_manifest.json").read_text())
    if digest({k:v for k,v in seal.items() if k!="manifest_sha256"}) != seal["manifest_sha256"]:
        raise DataError("Selected run validation seal differs")
    for name,sha in seal["artifacts"].items():
        if file_hash(Path(name)) != sha: raise DataError("Selected loss/validation artifact changed")
        sources.add(name)
    sources.add(str(run/"validation_manifest.json"))
    history = parse_json((run/"loss_history.json").read_text())["history"]
    paths = (Path("reports/baseline/baseline_eval_predictions.csv"),Path("reports/final/finetuned_eval_predictions.csv"))
    rows = []
    for path in paths:
        with path.open(newline="") as stream: rows.append(list(csv.DictReader(stream)))
    ids = [r["record_id"] for r in rows[0]]
    if ids != [r["record_id"] for r in rows[1]] or digest(ids) != fine["provenance"]["ordered_record_ids_sha256"] or digest(ids) != base["ordered_record_ids_sha256"]:
        raise DataError("Figure evaluation IDs/order differ")
    labels = np.array([int(r["label"]) for r in rows[0]])
    scores = [np.array([float(r["cosine"]) for r in group]) for group in rows]
    if list(labels) != [int(r["label"]) for r in rows[1]]:
        raise DataError("Figure evaluation labels differ")
    for report, values in zip((base,fine),scores):
        if pair_metrics(labels,values) != report["metrics"]: raise DataError("Figure metrics differ from predictions")
    return base,fine,comparison,rows,labels,scores,history,{n:file_hash(Path(n)) for n in sorted(sources)}


def visualize(directory: Path = Path("figures/submission")):
    base,fine,comparison,rows,labels,scores,history,inputs = verified_inputs()
    manifest = directory/"manifest.json"
    if manifest.exists():
        saved = parse_json(manifest.read_text())
        if saved["inputs"] != inputs or digest({k:v for k,v in saved.items() if k!="manifest_sha256"}) != saved["manifest_sha256"]:
            raise DataError("Figure manifest provenance differs")
        for name,sha in saved["artifacts"].items():
            if file_hash(Path(name)) != sha: raise DataError("Saved figure changed")
        return saved
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    plt.rcParams.update({"font.size":11,"axes.titlesize":12,"legend.fontsize":10})
    figures = []
    titles = ("Base MiniLM", "Final run-002")
    n = len(labels)
    fig,axes = plt.subplots(1,2,figsize=(13,5),sharex=True,sharey=True)
    for ax,values,title in zip(axes,scores,titles):
        for label in (0,1):
            ax.hist(values[labels==label],bins=np.linspace(-1,1,41),alpha=.6,color=COLORS[label],label=f"{'Compatible' if label else 'Incompatible'} (n={(labels==label).sum()})")
        ax.axvline(.5,color="black",linestyle="--",label="Threshold .5")
        ax.set(title=f"{title} — evaluation (n={n})",xlabel="Cosine similarity",ylabel="Pair count");ax.legend();ax.grid(alpha=.2)
    figures.append(fig)
    fig,axes = plt.subplots(1,2,figsize=(13,5))
    for ax,group,title,report in zip(axes,rows,titles,(base,fine)):
        xy=np.array([[float(r['umap_x']),float(r['umap_y'])] for r in group]); noise=np.array([int(r['cluster_id'])==-1 for r in group])
        for label in (0,1):
            mask=labels==label;ax.scatter(xy[mask,0],xy[mask,1],s=9,alpha=.6,color=COLORS[label],label='Compatible' if label else 'Incompatible')
        ax.scatter(xy[noise,0],xy[noise,1],s=18,facecolors='none',edgecolors='gray',label=f'Noise (n={noise.sum()})')
        ax.set(title=f"{title}: pairs n={n}, purity={report['clustering']['coverage_adjusted_purity']:.3f}",xlabel='UMAP 1 (arbitrary axis)',ylabel='UMAP 2 (arbitrary axis)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Evaluation pairs: separately fitted label-free maps; axes are unaligned',fontsize=12)
    figures.append(fig)
    categories=sorted(fine['breakdowns']['category']); y=np.arange(len(categories))
    fig,ax=plt.subplots(figsize=(13,8))
    for shift,report,color,title in zip((-.16,.16),(base,fine),COLORS,titles):
        for i,category in enumerate(categories):
            d=report['breakdowns']['category'][category]['cohens_d']
            if d is not None:
                ax.barh(i+shift,d,height=.28,color=color,alpha=.85,label=title if i==1 else None)
                ax.annotate(f'{d:.2f}',(d,i+shift),xytext=(5,0),textcoords='offset points',fontsize=10,va='center')
    ticklabels=[]
    for category in categories:
        m=fine['breakdowns']['category'][category]; extra=' — d unavailable (one class)' if m['cohens_d'] is None else ''
        ticklabels.append(f"{category.replace('_',' ')} (+: {m['positive']}, −: {m['negative']}){extra}")
    ax.set_yticks(y,ticklabels);ax.invert_yaxis();ax.axvline(0,color='gray',linestyle='--');ax.set(xlabel="Signed Cohen's d (pooled sample SD)",title='Evaluation category effect sizes — small subgroups are exploratory');ax.legend();ax.grid(axis='x',alpha=.2);ax.margins(x=.15)
    figures.append(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for values,title,color,report in zip(scores,titles,COLORS,(base,fine)):
        fpr,tpr,_=roc_curve(labels,values);ax.plot(fpr,tpr,color=color,label=f"{title} AUC={report['metrics']['auc']:.3f}")
    ax.plot([0,1],[0,1],color='gray',linestyle='--');ax.set(title=f'Evaluation ROC (n={n})',xlabel='False positive rate',ylabel='True positive rate');ax.legend();ax.grid(alpha=.2)
    figures.append(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for key,label,color in (('loss','Fit loss: 10-step windows',COLORS[0]),('eval_loss','Validation loss: full set',COLORS[1])):
        logs=[x for x in history if key in x];ax.plot([x['epoch'] for x in logs],[x[key] for x in logs],color=color,label=label,marker='o' if key=='eval_loss' else None,markersize=4)
    fit_logs=[x for x in history if 'loss' in x]
    epochs=sorted({int(np.ceil(x['epoch']-1e-9)) for x in fit_logs})
    epoch_means=[float(np.mean([x['loss'] for x in fit_logs if int(np.ceil(x['epoch']-1e-9))==epoch])) for epoch in epochs]
    ax.plot(epochs,epoch_means,color='black',marker='s',label='Fit: mean logged window loss per epoch')
    ax.set(title='Selected run-002: fit n=4792 / validation n=1199',xlabel='Epoch',ylabel='CosineSimilarityLoss (mean squared error)');ax.legend();ax.grid(alpha=.2)
    figures.append(fig)
    fig,axes=plt.subplots(3,3,figsize=(13,10))
    keys=('margin','cohens_d','fpr','auc','accuracy','precision','recall','f1','cluster_purity')
    for ax,key in zip(axes.flat,keys):
        delta=comparison['purity_comparison']['delta'] if key=='cluster_purity' else comparison['classification_deltas'][key]['delta']
        interval=comparison['uncertainty']['intervals'].get(key,{}).get('delta',{}).get('interval')
        ax.axvline(0,color='gray',linestyle='--');ax.barh(0,delta,height=.3,color=COLORS[1],alpha=.7)
        if interval is not None:ax.hlines(0,*interval,color=COLORS[1],linewidth=3)
        ax.set(title=f"{key.replace('_',' ')}: {delta:+.3f}",xlabel='Final − baseline (lower is better)' if key=='fpr' else 'Final − baseline (higher is better)',yticks=[]);ax.grid(axis='x',alpha=.2);ax.margins(x=.25)
        ax.text(.5,.12,'Paired component 95% interval' if interval else 'Point estimate; no interval',transform=ax.transAxes,ha='center',fontsize=10)
    fig.suptitle('Evaluation deltas — 2000 paired draws; template/seed/selection uncertainty excluded',fontsize=12)
    figures.append(fig)
    paths=[]
    for name,fig in zip(NAMES,figures):
        fig.tight_layout();buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=150);plt.close(fig)
        path=directory/f'{name}.png';write_bytes_immutable(path,buffer.getvalue());paths.append(path)
    result={'version':'s8-figures-v1','inputs':inputs,'artifacts':{str(p):file_hash(p) for p in paths},'evaluation_n':n,'positive':int((labels==1).sum()),'negative':int((labels==0).sum()),'selected_run':'run-002','code_sha256':file_hash(Path(__file__)),'minimum_font_points':10,'policy':'Saved predictions only; no inference, fitting or bootstrap'}
    result['manifest_sha256']=digest(result);write_immutable(manifest,result);return result

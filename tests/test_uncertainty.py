from types import SimpleNamespace
import pytest
from dating_compat.data import DataError
from dating_compat.uncertainty import paired_bootstrap


def rows():
    return [SimpleNamespace(pair=SimpleNamespace(label=i%2,text_1=f'a{i}',text_2=f'b{i}')) for i in range(40)]


def config():
    return {"component_minimum":30,"largest_component_maximum_fraction":.25,"bootstrap_resamples":30,"bootstrap_seed":42,"confidence_level":.95,"bootstrap_metrics":["margin","cohens_d","fpr","auc"]}


def test_same_predictions_have_exact_zero_paired_delta():
    records=rows(); scores=[.1+(i%2)*.6+i*.001 for i in range(40)]
    a=paired_bootstrap(records,scores,scores,config()); b=paired_bootstrap(records,scores,scores,config())
    assert a==b and a['method']=='paired_profile_component_percentile'
    assert all(v['delta']['interval']==[0.,0.] for v in a['intervals'].values())


def test_dominant_profile_component_has_explicit_fallback():
    records=rows()
    for r in records:r.pair.text_1='shared'
    scores=[.1+(i%2)*.6+i*.001 for i in range(40)]
    result=paired_bootstrap(records,scores,scores,config())
    assert result['profile_components']['count']==1
    assert result['method']=='paired_pair_percentile_conditional_approximate'


def test_missing_class_counts_undefined_resamples():
    records=rows()
    for r in records:r.pair.label=0
    result=paired_bootstrap(records,[.1]*40,[.2]*40,config())
    assert result['intervals']['auc']['delta']['interval'] is None
    assert result['intervals']['auc']['delta']['undefined_resamples']==30
    with pytest.raises(DataError):paired_bootstrap(records,[.1],[.2]*40,config())

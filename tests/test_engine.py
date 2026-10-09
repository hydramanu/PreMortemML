import json
import numpy as np
import pandas as pd
import pytest
from premortem.engine import (Scenario,demo_data,stress,split_data,validate,weights,metrics,run)
from premortem.report import write_report

def test_split_disjoint():
    x,y=demo_data(n=300);a,b,c,*_=split_data(x,y,42)
    assert not(set(a.index)&set(b.index) or set(a.index)&set(c.index) or set(b.index)&set(c.index))
    assert len(a)+len(b)+len(c)==len(x)

def test_zero_corruption_and_no_mutation():
    x,_=demo_data(n=300);before=x.copy(deep=True)
    pd.testing.assert_frame_equal(stress(x,Scenario('signal_1'),x.std(),5),x)
    stress(x,Scenario('signal_1',missing=.4,noise=1),x.std(),5)
    pd.testing.assert_frame_equal(x,before)

def test_determinism_and_only_target_feature():
    x,_=demo_data(n=300);s=Scenario('signal_1',noise=.6,missing=.2)
    a=stress(x,s,x.std(),5);b=stress(x,s,x.std(),5)
    pd.testing.assert_frame_equal(a,b)
    pd.testing.assert_frame_equal(a.iloc[:,1:],x.iloc[:,1:])
    assert a.signal_1.isna().any()

@pytest.mark.parametrize('kwargs',[{'missing':.5},{'noise':-1},{'offset':2},{'prevalence':0},{'noise':float('nan')}])
def test_invalid_scenario(kwargs):
    with pytest.raises(ValueError):Scenario('x',**kwargs)

def test_prevalence_weighting():
    y=np.array([0,0,0,1]);w=weights(y,.7)
    assert np.average(y,weights=w)==pytest.approx(.7)
    p=np.array([.1,.8,.1,.8])
    assert metrics(y,p,w)['balanced_accuracy']==pytest.approx(metrics(y,p)['balanced_accuracy'])

def test_bad_data():
    x,y=demo_data(n=300);x['bad']='string'
    with pytest.raises(ValueError):validate(x,y)

def test_full_run_and_report(tmp_path):
    x,y=demo_data(n=300);r=run(x,y,trials=5)
    assert len(r['history'])==5 and len(r['failure_curve'])==11
    assert 0<=r['stressed']['balanced_accuracy']<=1
    assert len(r['repair_delta_ci95'])==2
    path=write_report(r,tmp_path)
    assert '__RESULT_JSON__' not in path.read_text(encoding='utf-8')
    assert json.loads((tmp_path/'results.json').read_text(encoding='utf-8'))['seed']==42

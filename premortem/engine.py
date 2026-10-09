"""Reproducible bounded stress testing for numeric binary classifiers."""
from dataclasses import dataclass, asdict, replace
import hashlib
import platform
from importlib.metadata import version
from . import __version__
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix)

@dataclass(frozen=True)
class Scenario:
    feature: str
    offset: float = 0.0
    noise: float = 0.0
    missing: float = 0.0
    outliers: float = 0.0
    prevalence: float = 0.5

    def __post_init__(self):
        limits = {'offset':(-1.5,1.5), 'noise':(0,1), 'missing':(0,.4),
                  'outliers':(0,.1), 'prevalence':(.2,.8)}
        for k,(lo,hi) in limits.items():
            if not lo <= getattr(self,k) <= hi:
                raise ValueError(f'{k} must be between {lo} and {hi}')

def demo_data(seed=42, n=2400):
    x,y = make_classification(n_samples=n,n_features=8,n_informative=5,
        n_redundant=2,class_sep=1.3,flip_y=.03,random_state=seed)
    return pd.DataFrame(x,columns=[f'signal_{i+1}' for i in range(8)]), pd.Series(y,name='target')

def validate(x,y):
    if len(x)<200 or len(x)!=len(y) or x.shape[1]<1:
        raise ValueError('Need at least 200 rows, matching targets, and one numeric feature.')
    if not all(pd.api.types.is_numeric_dtype(t) for t in x.dtypes):
        raise ValueError('This version supports numeric features only; remove IDs and encode categories first.')
    if set(y.dropna().unique())!={0,1} or y.isna().any():
        raise ValueError('Target must contain both 0 and 1, with no missing labels.')
    if y.value_counts().min()<30:
        raise ValueError('Each class needs at least 30 rows.')
    if np.isinf(x.to_numpy(dtype=float)).any() or x.isna().all().any():
        raise ValueError('Features cannot contain infinity or be entirely missing.')
    if x.columns.duplicated().any():
        raise ValueError('Feature names must be unique.')

def split_data(x,y,seed):
    train_x, rest_x, train_y, rest_y = train_test_split(x,y,test_size=.4,stratify=y,random_state=seed)
    search_x,test_x,search_y,test_y = train_test_split(rest_x,rest_y,test_size=.5,stratify=rest_y,random_state=seed+1)
    return train_x,search_x,test_x,train_y,search_y,test_y

def model():
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                         StandardScaler(),LogisticRegression(max_iter=2000,random_state=0))

def stress(x,s,scale,seed):
    """Measurement corruption; labels remain valid only under this assumption.

    Separate random arrays keep ablations paired even when a component is disabled.
    """
    if s.feature not in x.columns:
        raise ValueError('Unknown scenario feature')
    rng=np.random.default_rng(seed)
    a=x.copy().astype(float)
    z=a[s.feature].to_numpy(copy=True)
    normal=rng.normal(size=len(z)); missing=rng.random(len(z))
    spikes=rng.random(len(z)); signs=rng.choice([-1,1],size=len(z))
    z += scale[s.feature]*(s.offset+s.noise*normal)
    z += (spikes<s.outliers)*signs*5*scale[s.feature]
    z[missing<s.missing]=np.nan
    a[s.feature]=z
    return a

def weights(y,prevalence):
    y=np.asarray(y)
    return np.where(y==1,prevalence/(y==1).mean(),(1-prevalence)/(y==0).mean())

def metrics(y,p,w=None):
    pred=p>=.5
    return {'accuracy':float(accuracy_score(y,pred,sample_weight=w)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred,sample_weight=w)),
        'precision':float(precision_score(y,pred,sample_weight=w,zero_division=0)),
        'recall':float(recall_score(y,pred,sample_weight=w,zero_division=0)),
        'f1':float(f1_score(y,pred,sample_weight=w,zero_division=0)),
        'roc_auc':float(roc_auc_score(y,p,sample_weight=w)),
        'brier':float(brier_score_loss(y,p,sample_weight=w)),
        'confusion_matrix':confusion_matrix(y,pred,sample_weight=w,labels=[0,1]).tolist()}

def evaluate(m,x,y,s,scale,seeds):
    w=weights(y,s.prevalence)
    records=[metrics(y,m.predict_proba(stress(x,s,scale,k))[:,1],w) for k in seeds]
    return {k:np.mean([r[k] for r in records],axis=0).tolist() for k in records[0]}

def sample_scenario(rng,columns):
    return Scenario(str(rng.choice(columns)),float(rng.uniform(-1.5,1.5)),
        float(rng.uniform(0,1)),float(rng.uniform(0,.4)),float(rng.uniform(0,.1)),float(rng.uniform(.2,.8)))

def search(m,x,y,scale,seed,trials,backend):
    seeds=[seed+100,seed+101,seed+102]; history=[]
    def assess(s):
        result=evaluate(m,x,y,s,scale,seeds)
        history.append({'trial':len(history)+1,**asdict(s),**result})
        return result['balanced_accuracy']
    if backend=='optuna':
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        def objective(t):
            s=Scenario(t.suggest_categorical('feature',list(x.columns)),
                t.suggest_float('offset',-1.5,1.5),t.suggest_float('noise',0,1),
                t.suggest_float('missing',0,.4),t.suggest_float('outliers',0,.1),
                t.suggest_float('prevalence',.2,.8))
            return assess(s)
        study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective,n_trials=trials,n_jobs=1)
    elif backend=='random':
        rng=np.random.default_rng(seed)
        for _ in range(trials): assess(sample_scenario(rng,x.columns))
    else: raise ValueError('backend must be random or optuna')
    worst=min(history,key=lambda r:r['balanced_accuracy'])
    return Scenario(**{k:worst[k] for k in Scenario.__dataclass_fields__}),history

def paired_interval(y,p0,p1,seed):
    """Stratified paired bootstrap for one fixed corruption realization."""
    rng=np.random.default_rng(seed); y=np.asarray(y); values=[]
    groups=[np.flatnonzero(y==c) for c in [0,1]]
    for _ in range(400):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        values.append(balanced_accuracy_score(y[idx],p1[idx]>=.5)-balanced_accuracy_score(y[idx],p0[idx]>=.5))
    return [float(v) for v in np.quantile(values,[.025,.975])]

def run(x=None,y=None,seed=42,trials=60,backend='random',source='synthetic demonstration'):
    if not 5<=trials<=500: raise ValueError('Use 5 to 500 trials.')
    if x is None: x,y=demo_data(seed)
    x=x.reset_index(drop=True); y=y.reset_index(drop=True)
    validate(x,y)
    tx,vx,hx,ty,vy,hy=split_data(x,y,seed)
    if tx.isna().all().any(): raise ValueError('A training feature is entirely missing.')
    scale=tx.std().replace(0,1).fillna(1)
    base=model().fit(tx,ty)
    worst,history=search(base,vx,vy,scale,seed,trials,backend)
    # Candidate repair sees training rows only. Selection uses search/validation only.
    augmented=[tx]+[stress(tx,worst,scale,seed+200+k) for k in range(3)]
    candidate=model().fit(pd.concat(augmented,ignore_index=True),pd.concat([ty]*4,ignore_index=True))
    vseeds=[seed+300+k for k in range(5)]
    base_v=evaluate(base,vx,vy,worst,scale,vseeds)['balanced_accuracy']
    repair_v=evaluate(candidate,vx,vy,worst,scale,vseeds)['balanced_accuracy']
    clean_v=metrics(vy,base.predict_proba(vx)[:,1])['balanced_accuracy']
    repair_clean_v=metrics(vy,candidate.predict_proba(vx)[:,1])['balanced_accuracy']
    selected=repair_v>base_v and repair_clean_v>=clean_v-.02
    # No model/scenario selection below this line: held-out assessment only.
    seeds=[seed+400+k for k in range(5)]
    clean=metrics(hy,base.predict_proba(hx)[:,1]); repaired_clean=metrics(hy,candidate.predict_proba(hx)[:,1])
    stressed=evaluate(base,hx,hy,worst,scale,seeds)
    repaired=evaluate(candidate,hx,hy,worst,scale,seeds)
    ablations=[]
    for component,value in [('offset',0),('noise',0),('missing',0),('outliers',0)]:
        score=evaluate(base,hx,hy,replace(worst,**{component:value}),scale,seeds)['balanced_accuracy']
        ablations.append({'component_removed':component,'balanced_accuracy':score,
                          'recovery_pp':100*(score-stressed['balanced_accuracy'])})
    curve=[]
    for intensity in np.linspace(0,1,11):
        s=replace(worst,offset=worst.offset*intensity,noise=worst.noise*intensity,
            missing=worst.missing*intensity,outliers=worst.outliers*intensity)
        a=evaluate(base,hx,hy,s,scale,seeds); b=evaluate(candidate,hx,hy,s,scale,seeds)
        curve.append({'intensity':float(intensity),'baseline':a['balanced_accuracy'],'repair':b['balanced_accuracy']})
    corrupted=stress(hx,worst,scale,seeds[0])
    interval=paired_interval(hy,base.predict_proba(corrupted)[:,1],candidate.predict_proba(corrupted)[:,1],seed+500)
    drop=100*(clean['balanced_accuracy']-stressed['balanced_accuracy'])
    review='HIGH' if stressed['balanced_accuracy']<.65 or drop>=15 else 'MEDIUM' if drop>=5 else 'LOW'
    fingerprint=hashlib.sha256(pd.util.hash_pandas_object(pd.concat([x,y.rename('__target__')],axis=1),index=True).values.tobytes()).hexdigest()
    return {'version':__version__,'source':source,'seed':seed,'trials':trials,'backend':backend,
        'data_sha256':fingerprint,'environment':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,**({'optuna':version('optuna')} if backend=='optuna' else {})},
        'split_rows':{'train':len(tx),'search':len(vx),'test':len(hx)},'feature_scale':scale.to_dict(),
        'worst_scenario':asdict(worst),'baseline':clean,'stressed':stressed,'repair_clean':repaired_clean,'repair_stressed':repaired,
        'repair_selected_on_validation':bool(selected),'validation':{'baseline_stress':base_v,'repair_stress':repair_v,'baseline_clean':clean_v,'repair_clean':repair_clean_v},
        'repair_delta_ci95':interval,'drop_pp':drop,'review_priority':review,'history':history,'ablations':ablations,'failure_curve':curve,
        'limitations':['Bounded measurement-corruption simulation, not a prediction of future failures.',
        'Worst discovered scenario is not a proven global worst case. Search changes one feature at a time with combined corruption types.',
        'Label prevalence reweighting affects accuracy and precision, but not the balanced-accuracy search objective.',
        'Ablation recovery is diagnostic, not proof of causality. Interactions need further study.',
        'Bootstrap interval covers rows for one fixed scenario and corruption seed, not search or training uncertainty.',
        'Review priority is a transparent heuristic, not a certified deployment risk score.',
        'Numeric binary IID split only. Time-series, grouped rows and concept shifts need different protocols.',
        'Repair is targeted augmentation; its wider robustness has not been established.']}

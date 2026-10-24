"""Shared constants and small helpers for the graph experiments.

FEATS     - the 165 supplied features.
BASE      - the same LightGBM settings as the recency reference.
RELATIONS - the six neighbour types used for graph features:
            in / out          = direct predecessors / successors,
            in2 / out2        = two hops back / forward,
            shared_successor  = other transactions that pay into the same successor,
            shared_predecessor= other transactions funded by the same predecessor.
The other helpers load saved validation predictions, build blends and compute metrics.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss

FEATS=[f'feat_{i}' for i in range(1,166)]
BASE=dict(n_estimators=500,learning_rate=.03,num_leaves=31,subsample=.8,
          subsample_freq=1,colsample_bytree=.5,verbose=-1,n_jobs=4)
RELATIONS=['in','out','in2','out2','shared_successor','shared_predecessor']

def emit(obj):print(json.dumps(obj,default=str),flush=True)

def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=str))

# Validation only: load the saved predictions of the previous ensemble for a fold.
def controls(root,frame,cut,end):
    for folder in ['screen','additional_checks']:
        path=Path(root)/folder/f'fold_{cut}_{end}.csv'
        if not path.exists():continue
        d=pd.read_csv(path).set_index('txId').loc[frame.txId]
        np.testing.assert_array_equal(d.label.to_numpy(),frame.label.to_numpy())
        b=d.reference.to_numpy()
        p=(d.logit_annotation3.to_numpy() if 'logit_annotation3' in d else
           expit(.8*logit(np.clip(b,1e-15,1-1e-15))+.2*logit(np.clip(d.annotation3.to_numpy(),1e-15,1-1e-15))))
        return dict(reference=b,incumbent=p)
    raise FileNotFoundError(f'Missing saved comparison {cut}:{end}')

# The three ways a new model was compared: alone, 20% in log-odds, 20% in probability.
def blends(anchor,p,name):
    return {name:p,f'{name}_logit20':expit(.8*logit(np.clip(anchor,1e-7,1-1e-7))+
                .2*logit(np.clip(p,1e-7,1-1e-7))),f'{name}_prob20':.8*anchor+.2*p}

# AUC / AP / log loss for each prediction column, plus AUC per time step.
def metrics(frame,columns,cut,end):
    records=[]
    for name,p in columns.items():
        assert np.isfinite(p).all(),name
        r=dict(cutoff=cut,end=end,name=name,auc=roc_auc_score(frame.label,p),
               ap=average_precision_score(frame.label,p),
               logloss=log_loss(frame.label,np.clip(p,1e-7,1-1e-7)))
        r['period_auc']={}
        for t in sorted(frame.time_step.unique()):
            ix=np.flatnonzero(frame.time_step.to_numpy()==t)
            if frame.label.iloc[ix].nunique()==2:
                r['period_auc'][str(int(t))]=roc_auc_score(frame.label.iloc[ix],np.asarray(p)[ix])
        records.append(r)
    return records

def summarize(out):
    rows=[]
    for path in Path(out).glob('fold_*/metrics.json'):rows.extend(json.loads(path.read_text()))
    if not rows:return
    d=pd.DataFrame(rows).drop(columns='period_auc')
    d.to_csv(Path(out)/'results.csv',index=False)
    s=d[d.cutoff.isin([25,28,30])].groupby('name').agg(folds=('auc','size'),auc=('auc','mean'),ap=('ap','mean'))
    s.sort_values('auc',ascending=False).to_csv(Path(out)/'summary.csv')
    emit(dict(event='summary',results=s.sort_values('auc',ascending=False).to_dict('index')))

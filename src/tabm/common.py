"""Shared helpers for the TabM stage (model T).

Data loading with sanity checks, temporal fold selection, recency weights,
metrics, and a cache manifest that refuses to reuse outputs if the input files,
code or settings changed.
"""
from pathlib import Path
import hashlib,json,platform,sys
import numpy as np
import pandas as pd
from scipy.special import logit,expit
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss

FEATURES=[f'feat_{i}' for i in range(1,166)]
# Validation folds (cutoff, last step) used when comparing TabM variants.
FOLDS=[(25,35),(28,32),(30,35),(21,35)]
# Variants compared on validation. The final model T is 'tabm_ple_uniform'
# (piecewise-linear embeddings, uniform weights): it was the strongest on the
# long 21->22-35 check. The CatBoost/LightGBM names refer to a separate script
# not included here.
MODELS=['tabm_ple_tau5','tabm_ple_uniform','tabm_plain_tau5','cat_tau5','cat_uniform','lgbm_reference']
# TabM settings: 32 ensemble members sharing most parameters, 3 blocks of width 512,
# dropout 0.1, AdamW (lr 0.002, weight decay 3e-4), batch 512, 48-bin piecewise-linear
# embeddings of size 16; early stopping on the inner window (patience 12).
TABM=dict(k=32,n_blocks=3,d_block=512,dropout=.1,lr=.002,weight_decay=.0003,batch_size=512,n_bins=48,d_embedding=16,max_epochs=100,min_epochs=5,patience=12)

def emit(**x):print(json.dumps(x,default=str),flush=True)
def save(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,indent=2,default=str)+'\n')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
# Load the CSVs as float32 and check the expected structure (train 1-35, test 36-49).
def load(data):
    data=Path(data);train=pd.read_csv(data/'train.csv');test=pd.read_csv(data/'test.csv');sample=pd.read_csv(data/'sample_submission.csv')
    assert train.txId.is_unique and test.txId.is_unique and set(train.txId).isdisjoint(test.txId)
    assert train.time_step.max()==35 and test.time_step.min()==36 and test.time_step.max()==49
    assert set(train.label.dropna().unique())=={0,1}
    for f in [train,test]:
        f[FEATURES]=f[FEATURES].astype('float32')
        assert np.isfinite(f[FEATURES].to_numpy()).all()
    np.testing.assert_array_equal(test['index'],sample['index'])
    assert sample['index'].is_unique
    return train,test,sample
# prefix = labelled rows up to the cutoff; query = labelled rows in (cutoff, end].
def prefix(train,cut):return train[train.label.notna()&train.time_step.le(cut)].copy()
def query(train,cut,end):return train[train.label.notna()&train.time_step.between(cut+1,end)].copy()
# tau=None gives uniform weights (used by the final TabM); tau=5 gives recency weights.
def weights(frame,cut,tau,normalize=True):
    w=np.ones(len(frame)) if tau is None else np.exp((frame.time_step.to_numpy()-cut)/tau)
    if normalize:w=w/w.mean()
    return w
def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return dict(auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,ap=float(average_precision_score(y,p)) if len(np.unique(y))==2 else None,logloss=float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),n=len(y),illicit=int(y.sum()))
def signature(data,root,configuration):
    return dict(inputs={n:sha(Path(data)/n) for n in ['train.csv','test.csv','sample_submission.csv']},
                source={p.name:sha(p) for p in sorted((Path(root)/'src').glob('*.py'))},configuration=configuration)
# Refuse to mix cached results from different inputs/code/settings.
def guard(folder,manifest):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);p=folder/'manifest.json'
    if p.exists():assert json.loads(p.read_text())==manifest,'Cached inputs/settings changed: use a new output directory.'
    else:save(p,manifest)
def common_args():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',default='0,1,2');p.add_argument('--folds',default='25:35,28:32,30:35,21:35');p.add_argument('--device',default='cuda')
    p.add_argument('--models',default='');p.add_argument('--final',default='');p.add_argument('--smoke',action='store_true')
    return p
def parse(p):
    a=p.parse_args();a.seeds=[int(s) for s in a.seeds.split(',')];a.folds=[tuple(map(int,x.split(':'))) for x in a.folds.split(',')]
    return a

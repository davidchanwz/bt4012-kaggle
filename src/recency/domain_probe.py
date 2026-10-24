"""Experiment (not in the final model): covariate-shift reweighting.

A classifier is trained to tell training rows from test rows using features
only (never test labels). Features that are strongly time-correlated are
removed first. The classifier's out-of-fold scores then softly increase the
weight of training rows that look like the test period. This "future-similarity"
blend scored 0.95120 publicly and was not selected. It is kept because
finalize.py can still fit it if listed in a lock file.
"""
from pathlib import Path
import argparse,json,time
import numpy as np,pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score
from recency_lab import FEATS,BASE,load


def adapt_weights(F,target,cutoff):
 # target is explicitly restricted to numeric predictors: no target-label access.
 target=target[FEATS]
 w=np.exp((F.time_step.values-cutoff)/5)
 # Drop features whose Spearman correlation with time_step is > 0.5 (clock-like).
 rho=F[FEATS].corrwith(F.time_step,method='spearman').abs()
 cols=list(rho[rho<=.5].index)
 X=pd.concat([F[cols],target[cols]],ignore_index=True)
 z=np.r_[np.zeros(len(F),dtype=int),np.ones(len(target),dtype=int)]
 dw=np.r_[w/w.mean(),np.full(len(target),len(F)/len(target))]
 cv=list(StratifiedKFold(3,shuffle=True,random_state=4012).split(X,z))
 history=[];removed=list(rho[rho>.5].index);oof=None
 # Repeatedly drop the 3 most useful features until train vs test is hard to separate.
 for iteration in range(7):
  oof=np.empty(len(X));gain=np.zeros(len(cols))
  for a,b in cv:
   m=lgb.LGBMClassifier(n_estimators=180,learning_rate=.05,num_leaves=7,
       min_child_samples=100,reg_lambda=10,verbose=-1,n_jobs=4,random_state=4012)
   m.fit(X.iloc[a][cols],z[a],sample_weight=dw[a])
   oof[b]=m.predict_proba(X.iloc[b][cols])[:,1]
   gain+=m.booster_.feature_importance('gain')
  auc=roc_auc_score(z,oof,sample_weight=dw)
  history.append({'iteration':iteration,'features':len(cols),'domain_auc':auc})
  if auc<=.8:break
  if iteration==6:break
  drop=[cols[i] for i in np.argsort(-gain)[:3]]
  removed+=drop;cols=[c for c in cols if c not in drop]
 # If the two periods are still almost perfectly separable, give up (no reweighting).
 if history[-1]['domain_auc']>.95:
  correction=np.ones(len(F));fallback=True
 else:
  p=np.clip(oof[:len(F)],.05,.95)
  # Soft, capped importance weight based on how test-like each training row looks.
  correction=np.sqrt(np.clip(p/(1-p),.25,4))
  fallback=False
 adjusted=w*correction
 adjusted*=w.sum()/adjusted.sum()
 diagnostics={'history':history,'removed':removed,'fallback':fallback,
  'correction_quantiles':np.quantile(correction,[0,.1,.5,.9,1]).tolist(),
  'effective_sample_size':float(adjusted.sum()**2/(adjusted**2).sum())}
 return adjusted,diagnostics


def run(full,out,folds):
 out.mkdir(parents=True,exist_ok=True);records=[]
 full=full.astype({c:'float32' for c in FEATS})
 for cut,end in folds:
  F=full[(~full.is_test)&full.label.notna()&full.time_step.le(cut)]
  V=full[(~full.is_test)&full.label.notna()&full.time_step.between(cut+1,end)]
  started=time.time();w,diag=adapt_weights(F,V[FEATS],cut)
  (out/f'domain_{cut}_{end}.json').write_text(json.dumps(diag,indent=2))
  predictions=[]
  for seed in [0,1,2]:
   m=lgb.LGBMClassifier(**BASE,random_state=seed)
   m.fit(F[FEATS],F.label.astype(int),sample_weight=w)
   predictions.append(m.predict_proba(V[FEATS])[:,1])
  p=np.mean(predictions,axis=0)
  V[['txId','time_step','label']].assign(domain=p).to_csv(out/f'fold_{cut}_{end}.csv',index=False)
  rec={'cutoff':cut,'end':end,'auc':roc_auc_score(V.label,p),
       'ap':average_precision_score(V.label,p),'seconds':time.time()-started,
       'domain_auc':diag['history'][-1]['domain_auc'],'fallback':diag['fallback']}
  print(json.dumps(rec),flush=True);records.append(rec)
  pd.DataFrame(records).to_csv(out/'results.csv',index=False)


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True)
 a.add_argument('--cache',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
 args=a.parse_args();run(load(args.data,args.cache),args.out,[(25,30),(28,32),(30,35)])

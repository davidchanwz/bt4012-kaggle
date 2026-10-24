"""Blend helper used by finalize.py, plus extra validation checks for a shortlist.

`blend` defines how an extra model is combined with the reference:
- 'logit_' prefix: sigmoid(0.8*logit(reference) + 0.2*logit(other))   <- used for the annotation model
- 'blend_' prefix: 0.5*reference + 0.5*other
Running this file directly re-checks shortlisted candidates on extra cutoffs.
These folds overlap with earlier ones, so they are not independent holdouts.
"""
from pathlib import Path
import argparse,json,time
import numpy as np,pandas as pd
import lightgbm as lgb
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,average_precision_score
from recency_lab import load,FEATS,BASE,model_and_data
from domain_probe import adapt_weights
from rank_probe import fine_tune


def blend(candidate,base,other):
 if candidate.startswith('logit_'):
  # Blend in log-odds space; clipping avoids logit(0) or logit(1).
  return expit(.8*logit(np.clip(base,1e-15,1-1e-15))+.2*logit(np.clip(other,1e-15,1-1e-15)))
 if candidate.startswith('blend_'):return .5*base+.5*other
 return other


def components(full,F,V,cut,seeds,names):
 """Fit the reference and each candidate on 3 seeds for one fold and return their averaged predictions."""
 base_ps=[];parts={name:[] for name in names}
 domain_weights=None
 if any(name.endswith('domain') for name in names):
  domain_weights,_=adapt_weights(F,V[FEATS],cut)
 for seed in seeds:
  bp,bm=model_and_data('reference',F,V,seed,cut,full);base_ps.append(bp)
  for name in names:
   core=name.removeprefix('blend_').removeprefix('logit_')
   if core=='domain':
    m=lgb.LGBMClassifier(**BASE,random_state=seed)
    m.fit(F[FEATS],F.label.astype(int),sample_weight=domain_weights)
    p=m.predict_proba(V[FEATS])[:,1]
   elif core=='rank_finetune':p,_=fine_tune(F,V,seed,cut,bm)
   else:p,_=model_and_data(core,F,V,seed,cut,full)
   parts[name].append(p)
 return np.mean(base_ps,axis=0),{k:np.mean(v,axis=0) for k,v in parts.items()}


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True)
 a.add_argument('--cache',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
 a.add_argument('--candidates',required=True);a.add_argument('--folds',default='27:30,32:35,20:35,25:35')
 args=a.parse_args();names=args.candidates.split(',');args.out.mkdir(parents=True,exist_ok=True)
 f=load(args.data,args.cache).astype({c:'float32' for c in FEATS});records=[]
 for cut,end in [tuple(map(int,x.split(':'))) for x in args.folds.split(',')]:
  start=time.time();F=f[(~f.is_test)&f.label.notna()&f.time_step.le(cut)]
  V=f[(~f.is_test)&f.label.notna()&f.time_step.between(cut+1,end)]
  base,parts=components(f,F,V,cut,[0,1,2],names)
  preds=V[['txId','time_step','label']].assign(reference=base)
  for name,other in parts.items():preds[name]=blend(name,base,other)
  for name in ['reference']+names:
   p=preds[name]
   rec={'cutoff':cut,'end':end,'name':name,'auc':roc_auc_score(V.label,p),
       'ap':average_precision_score(V.label,p),'delta':roc_auc_score(V.label,p)-roc_auc_score(V.label,base),
       'seconds':time.time()-start}
   print(json.dumps(rec),flush=True);records.append(rec)
  preds.to_csv(args.out/f'fold_{cut}_{end}.csv',index=False)
  pd.DataFrame(records).to_csv(args.out/'results.csv',index=False)

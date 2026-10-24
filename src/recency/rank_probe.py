"""Experiment (not in the final model): pairwise ranking fine-tune.

Continues training the reference LightGBM with a custom loss that mixes
binary log loss (70%) and a pairwise ranking loss (30%) that directly rewards
scoring illicit rows above licit rows, which is what AUC measures. Gains on
validation were tiny, so it was not selected.
"""
from pathlib import Path
import argparse,json,time
import numpy as np,pandas as pd
import lightgbm as lgb
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score
from recency_lab import load,FEATS,model_and_data


def fine_tune(F,V,seed,cutoff,base,rounds=150):
 y=F.label.astype(int).values;w=np.exp((F.time_step.values-cutoff)/5)
 pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
 pp=w[pos]/w[pos].sum();pn=w[neg]/w[neg].sum()
 rng=np.random.default_rng(20261015+seed);n=len(F);pairs=8*n
 scale=w.sum()/(2*pairs)
 # Custom LightGBM objective: returns gradient and hessian of the mixed loss.
 def objective(pred,dataset):
  p=expit(pred)
  grad=w*(p-y);hess=w*p*(1-p)
  # Sample (illicit, licit) pairs, favouring recent rows.
  i=rng.choice(pos,pairs,p=pp);j=rng.choice(neg,pairs,p=pn)
  q=expit(pred[j]-pred[i]);h=q*(1-q)
  rg=(np.bincount(j,weights=q,minlength=n)-np.bincount(i,weights=q,minlength=n))*scale
  rh=(np.bincount(j,weights=h,minlength=n)+np.bincount(i,weights=h,minlength=n))*scale
  return .7*grad+.3*rg,np.maximum(.7*hess+.3*rh,1e-8)
 params=base.booster_.params.copy()
 for key in ['num_iterations','num_boost_round','objective','metric']:
  params.pop(key,None)
 params.update(objective=objective,metric='None',learning_rate=.01,num_leaves=15,
               min_child_samples=30,verbosity=-1)
 ds=lgb.Dataset(F[FEATS],label=y,weight=w,free_raw_data=False)
 # Warm start from the already-trained reference model.
 model=lgb.train(params,ds,num_boost_round=rounds,init_model=base.booster_)
 return expit(model.predict(V[FEATS],raw_score=True)),model


def run(full,out,folds,seeds):
 out.mkdir(parents=True,exist_ok=True);records=[]
 full=full.astype({c:'float32' for c in FEATS})
 for cut,end in folds:
  F=full[(~full.is_test)&full.label.notna()&full.time_step.le(cut)]
  V=full[(~full.is_test)&full.label.notna()&full.time_step.between(cut+1,end)]
  started=time.time();base_ps=[];rank_ps=[]
  for seed in seeds:
   bp,base=model_and_data('reference',F,V,seed,cut,full)
   rp,_=fine_tune(F,V,seed,cut,base)
   base_ps.append(bp);rank_ps.append(rp)
  bp=np.mean(base_ps,axis=0);rp=np.mean(rank_ps,axis=0)
  pred=V[['txId','time_step','label']].assign(reference=bp,rank_finetune=rp)
  pred.to_csv(out/f'fold_{cut}_{end}.csv',index=False)
  for name,p in [('reference',bp),('rank_finetune',rp),('rank_blend',.5*bp+.5*rp)]:
   rec={'cutoff':cut,'end':end,'name':name,'auc':roc_auc_score(V.label,p),
       'delta':roc_auc_score(V.label,p)-roc_auc_score(V.label,bp),
       'ap':average_precision_score(V.label,p),'seconds':time.time()-started}
   print(json.dumps(rec),flush=True);records.append(rec)
  pd.DataFrame(records).to_csv(out/'results.csv',index=False)


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True)
 a.add_argument('--cache',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
 a.add_argument('--folds',default='25:30,28:32,30:35');a.add_argument('--seeds',default='0,1,2')
 args=a.parse_args();run(load(args.data,args.cache),args.out,
  [tuple(map(int,z.split(':'))) for z in args.folds.split(',')],list(map(int,args.seeds.split(','))))

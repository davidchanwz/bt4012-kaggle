"""Recency-weighted LightGBM: shared helpers and the temporal-validation lab.

Pipeline role (stage 1 of ensemble B):
- `load` reads train.csv + test.csv into one frame (cached as a pickle).
- `model_and_data` fits one model for one seed. The final pipeline uses two
  configurations from CONFIGS:
    * 'reference'   - LightGBM on all 165 features, rows weighted by
                      exp((time_step - cutoff) / 5) so recent steps count more
                      (concept drift: illicit activity changes over time).
    * 'annotation3' - a 3-class model (licit / illicit / unknown) that uses the
                      unlabelled rows without calling them licit.
  The other CONFIGS are experiments that were compared on validation and not
  used in the final model (other decay rates, capacity, CatBoost/XGBoost,
  a compact graph feature set, and 'id_feature'/'id_only', which check that
  txId carries no label information).
- `evaluate` runs the temporal validation: train on steps <= cutoff, predict
  steps cutoff+1..end, average the predictions of 3 seeds, then compute AUC.
  Default folds: 25->26-30, 28->29-32, 30->31-35.

Only the four supplied competition CSV files are used.
"""
from pathlib import Path
from collections import deque
import argparse, json, time, warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
# All 165 anonymised features are used as-is (no feature removal).
FEATS=[f'feat_{i}' for i in range(1,166)]
# Reference LightGBM settings (kept fixed after validation; bigger/smaller
# trees did not give a consistent improvement).
BASE=dict(n_estimators=500, learning_rate=.03, num_leaves=31, subsample=.8,
          subsample_freq=1, colsample_bytree=.5, verbose=-1, n_jobs=4)
# Named experiment configurations. 'reference' = recency weighting with tau=5.
CONFIGS={
 'reference':{},
 'unweighted':{'tau':None},
 'tau3':{'tau':3}, 'tau8':{'tau':8},
 'weight_floor':{'floor':.10},
 'leaf15':{'params':{'num_leaves':15,'n_estimators':900,'learning_rate':.02}},
 'leaf63':{'params':{'num_leaves':63,'min_child_samples':10}},
 'feature_bag':{'params':{'colsample_bytree':.3,'n_estimators':750,'learning_rate':.025}},
 'dart':{'params':{'boosting_type':'dart','n_estimators':600,'learning_rate':.05}},
 'compact_graph':{'graph':True},
 'cat6':{'family':'cat','depth':6},
 'cat8':{'family':'cat','depth':8},
 'xgb':{'family':'xgb'},
 'annotation3':{'family':'annotation'},
 'bin511':{'params':{'max_bin':511}},
 'bin1023':{'params':{'max_bin':1023}},
 'bin1023rare':{'params':{'max_bin':1023,'min_data_in_bin':1}},
 'recent15':{'window':15},
 'id_feature':{'id':True},
 'id_only':{'id_only':True},
}


def load(data,cache):
 """Load train.csv and test.csv into one frame with an is_test flag (cached)."""
 cache.mkdir(parents=True,exist_ok=True)
 p=cache/'raw.pkl'
 if p.exists():return pd.read_pickle(p)
 train=pd.read_csv(data/'train.csv');test=pd.read_csv(data/'test.csv')
 train['is_test']=False;test['is_test']=True
 f=pd.concat([train,test],ignore_index=True)
 f.to_pickle(p)
 return f


# Compact topology features for the 'compact_graph' experiment only.
# Not used by the final model (the final graph features are in src/graph/).
def graph_features(full,data):
 e=pd.read_csv(data/'txs_edgelist.csv')
 ids=pd.Index(np.unique(np.r_[full.txId.values,e.values.ravel()]))
 s,t=ids.get_indexer(e.txId1),ids.get_indexer(e.txId2)
 A=sparse.csr_matrix((np.ones(len(s)),(s,t)),shape=(len(ids),len(ids)))
 A.sum_duplicates();A.data[:]=1
 AT=A.T.tocsr();ind=np.diff(AT.indptr);out=np.diff(A.indptr);n=len(ids)
 remain=ind.copy();q=deque(np.flatnonzero(remain==0));order=[]
 while q:
  u=q.popleft();order.append(u)
  for v in A.indices[A.indptr[u]:A.indptr[u+1]]:
   remain[v]-=1
   if remain[v]==0:q.append(v)
 # Kahn's algorithm gives a topological order (money only flows forward).
 assert len(order)==n,'Expected an acyclic transaction graph'
 upstream=np.zeros(n);downstream=np.zeros(n);before=np.zeros(n);after=np.zeros(n)
 for u in order:
  for v in A.indices[A.indptr[u]:A.indptr[u+1]]:
   upstream[v]=max(upstream[v],upstream[u]+1)
   if out[u]==1 and ind[v]==1:before[v]=before[u]+1
 for u in reversed(order):
  for v in A.indices[A.indptr[u]:A.indptr[u+1]]:
   downstream[u]=max(downstream[u],downstream[v]+1)
   if out[u]==1 and ind[v]==1:after[u]=after[v]+1
 row=ids.get_indexer(full.txId)
 visible=np.zeros(n);visible[row[full.label.notna()|full.is_test]]=1
 g={'graph_in':ind,'graph_out':out,'graph_upstream_depth':upstream,
    'graph_downstream_depth':downstream,'graph_chain_before':before,'graph_chain_after':after,
    'graph_relative_position':upstream/(upstream+downstream+1),
    'graph_fan_balance':(ind-out)/(ind+out+1)}
 for direction,B,degree in [('in',AT,ind),('out',A,out)]:
  den=np.maximum(degree,1)
  for name,values in [('in',ind),('out',out),('source',ind==0),('sink',out==0),
                       ('visible',visible),('up_depth',upstream),('down_depth',downstream)]:
   g[f'graph_{direction}_neighbor_{name}']=(B@values.astype(float))/den
 return pd.DataFrame({k:v[row] for k,v in g.items()},index=full.index)


def model_and_data(name,F,V,seed,cutoff,full=None):
 """Fit one configuration for one seed and return (validation predictions, model).

 F: labelled training rows (time_step <= cutoff); V: rows to predict.
 full: the whole frame, only needed by 'annotation3' (it also uses unknown rows).
 """
 c=CONFIGS[name];family=c.get('family','lgb');tau=c.get('tau',5)
 if c.get('window'):F=F[F.time_step>cutoff-c['window']]
 # Recency weight: a row from the cutoff step gets 1, every 5 steps back
 # counts about 2.7x less (step 30 -> 0.37, step 20 -> 0.05 when cutoff=35).
 w=np.ones(len(F)) if tau is None else np.exp((F.time_step.values-cutoff)/tau)
 if 'floor' in c:w=c['floor']+(1-c['floor'])*w
 cols=FEATS+([x for x in F if x.startswith('graph_')] if c.get('graph') else [])
 if c.get('id'):cols=cols+['txId']
 if c.get('id_only'):cols=['txId']
 Xf,Xv=F[cols],V[cols]
 if family=='annotation':
  # Unknown is an annotation-status class, not a legitimate/illicit label.
  # Ratios of the two known classes are evaluated against the binary target.
  # Use ALL training rows up to the cutoff, including unknown ones (class 2).
  U=full[(~full.is_test)&full.time_step.le(cutoff)]
  y=U.label.fillna(2).astype(int)
  w=np.exp((U.time_step.values-cutoff)/tau)
  # Keep total unknown weight comparable to known weight.
  unknown=y.eq(2).values
  w[unknown]*=min(1,w[~unknown].sum()/max(w[unknown].sum(),1e-12))
  m=lgb.LGBMClassifier(**{**BASE,'random_state':seed,'objective':'multiclass','num_class':3})
  m.fit(U[FEATS],y,sample_weight=w)
  # Conditional illicit score: P(illicit) / (P(licit) + P(illicit)).
  # This removes the 'unknown' option so the score is comparable to a binary model.
  p=m.predict_proba(Xv);return p[:,1]/np.maximum(p[:,0]+p[:,1],1e-15),m
 if family=='cat':
  m=CatBoostClassifier(iterations=1200,learning_rate=.035,depth=c['depth'],l2_leaf_reg=5,
    random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False)
 elif family=='xgb':
  m=XGBClassifier(n_estimators=1400,learning_rate=.025,max_depth=4,min_child_weight=2,
    subsample=.85,colsample_bytree=.7,reg_lambda=2,n_jobs=4,random_state=seed,tree_method='hist')
 else:
  m=lgb.LGBMClassifier(**{**BASE,**c.get('params',{}),'random_state':seed})
 m.fit(Xf,F.label.astype(int),sample_weight=w)
 return m.predict_proba(Xv)[:,1],m


def evaluate(full,data,out,names,folds,seeds):
 """Temporal validation: for each (cutoff, end) fold, fit each config on 3 seeds and record AUC/AP/log loss."""
 out.mkdir(parents=True,exist_ok=True)
 # This exactly reproduces the saved public-reference model's input precision.
 full=full.astype({f:'float32' for f in FEATS})
 if any(CONFIGS[name].get('graph') for name in names):
  full=pd.concat([full,graph_features(full,data)],axis=1)
 rec=[];seed_records=[]
 for cutoff,end in folds:
  # Temporal split: fit on labelled rows up to the cutoff, validate on the
  # labelled rows of the following steps (never random k-fold).
  F=full[full.label.notna()&(~full.is_test)&full.time_step.le(cutoff)]
  V=full[full.label.notna()&(~full.is_test)&full.time_step.between(cutoff+1,end)]
  preds=V[['txId','time_step','label']].copy()
  for name in names:
   start=time.time();ps=[]
   for seed in seeds:
    p,_=model_and_data(name,F,V,seed,cutoff,full)
    ps.append(p)
    seed_records.append({'cutoff':cutoff,'end':end,'name':name,'seed':seed,
                         'auc':roc_auc_score(V.label,p)})
   # Average the seeds' predictions first, then compute AUC on the average.
   p=np.mean(ps,axis=0);preds[name]=p
   per=[]
   for _,d in preds.assign(p=p).groupby('time_step'):
    if d.label.nunique()==2:per.append(roc_auc_score(d.label,d.p))
   r={'cutoff':cutoff,'end':end,'name':name,'auc':roc_auc_score(V.label,p),
      'ap':average_precision_score(V.label,p),'logloss':log_loss(V.label,p),
      'mean_period_auc':np.mean(per),'seconds':round(time.time()-start,2)}
   rec.append(r);print(json.dumps(r),flush=True)
   preds.to_csv(out/f'fold_{cutoff}_{end}.csv',index=False)
   pd.DataFrame(rec).to_csv(out/'results.csv',index=False)
   pd.DataFrame(seed_records).to_csv(out/'seed_results.csv',index=False)
 return rec


# Check that the saved 0.95113 recency submission can be reproduced exactly
# (float32 inputs reproduce it; float64 changes the trees slightly).
def reproduce(full,data,out):
 out.mkdir(parents=True,exist_ok=True)
 F=full[full.label.notna()&(~full.is_test)];V=full[full.is_test]
 old=pd.read_csv(data/'submission_lgbm_recency.csv').target.values
 records=[]
 for dtype in ['float64','float32']:
  ff=F.copy();vv=V.copy();ff[FEATS]=ff[FEATS].astype(dtype);vv[FEATS]=vv[FEATS].astype(dtype)
  ps=[model_and_data('reference',ff,vv,s,35,full)[0] for s in range(5)]
  p=np.mean(ps,axis=0)
  pd.DataFrame({'index':V['index'].astype(int),'target':p}).to_csv(out/f'reproduction_{dtype}.csv',index=False)
  rec={'dtype':dtype,'max_abs_difference':float(abs(p-old).max()),
    'mean_abs_difference':float(abs(p-old).mean()),'spearman':float(pd.Series(p).rank().corr(pd.Series(old).rank()))}
  print(json.dumps(rec),flush=True);records.append(rec)
 (out/'reproduction.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
 p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--models',default=','.join(CONFIGS));p.add_argument('--folds',default='25:30,28:32,30:35')
 p.add_argument('--seeds',default='0,1,2');p.add_argument('--reproduce',action='store_true')
 a=p.parse_args();f=load(a.data,a.cache)
 if a.reproduce:reproduce(f,a.data,a.out)
 else:evaluate(f,a.data,a.out,a.models.split(','),[tuple(map(int,z.split(':'))) for z in a.folds.split(',')],list(map(int,a.seeds.split(','))))

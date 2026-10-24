"""Validation experiments for the graph LightGBM.

Compares graph feature sets and visibility policies on the temporal folds
(25->26-30, 28->29-32, 30->31-35, 3 seeds each). Each model is scored alone and
blended with the previous ensemble. 'matched_relations' (all six relations,
matched visibility) was selected; `matrix` is reused by finalize_graph.py.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import BASE,controls,blends,metrics,save_json,emit,summarize
from graph_data import load,selected_columns

MODELS={
    'coverage':('matched','coverage'),
    # Trained with ALL neighbour features but tested with matched ones:
    # shows how much the train/test availability mismatch hurts.
    'visible_direct':('visible','direct'),
    'matched_direct':('matched','direct'),
    'matched_relations':('matched','relations'),
    'reconstructed_direct':('reconstructed','direct'),
}


def matrix(panel,ix,cols,names):
    """Select rows/columns from the cached panel as a float32 DataFrame with column names."""
    return pd.DataFrame(np.asarray(panel[np.ix_(ix,cols)],np.float32),columns=[names[i] for i in cols])


def run_fold(args,cut,end):
    """Fit every requested graph model for one temporal fold and save predictions and metrics."""
    folder=args.out/f'fold_{cut}_{end}';folder.mkdir(parents=True,exist_ok=True)
    rows,masked,manifest=load(args.cache,'matched')
    panels={'matched':masked}
    fi=np.flatnonzero((~rows.is_test&rows.time_step.le(cut)).to_numpy())
    vi=np.flatnonzero((~rows.is_test&rows.time_step.between(cut+1,end)).to_numpy())
    f=rows.iloc[fi];v=rows.iloc[vi]
    anchor=controls(args.old_root,v,cut,end)
    preds=v[['txId','time_step','label']].reset_index(drop=True)
    if (folder/'predictions.csv').exists():
        previous=pd.read_csv(folder/'predictions.csv').set_index('txId').loc[v.txId]
        for col in previous:
            if col not in preds:preds[col]=previous[col].to_numpy()
    for name,p in anchor.items():preds[name]=p
    for name in args.models.split(','):
        start=time.time();policy,kind=MODELS[name]
        emit(dict(event='tree_model_start',cutoff=cut,end=end,name=name))
        if policy not in panels:panels[policy]=load(args.cache,policy)[1]
        testpolicy='matched' if policy=='visible' else policy
        cols=selected_columns(kind,manifest)
        xf=matrix(panels[policy],fi,cols,manifest['columns'])
        xv=matrix(panels[testpolicy],vi,cols,manifest['columns'])
        oracle=matrix(panels[policy],vi,cols,manifest['columns']) if policy=='visible' else None
        # Same recency weighting as the reference model.
        weights=np.exp((f.time_step.to_numpy()-cut)/5)
        ps=[];os=[];importances=[]
        for seed in map(int,args.seeds.split(',')):
            path=folder/f'{name}_seed_{seed}.npz'
            if path.exists():
                cache=np.load(path);p=cache['p'];gain=cache['gain']
                if oracle is not None:os.append(cache['oracle'])
            else:
                model=lgb.LGBMClassifier(**BASE,random_state=seed)
                model.fit(xf,f.label.astype(int),sample_weight=weights)
                p=model.predict_proba(xv)[:,1]
                gain=model.booster_.feature_importance(importance_type='gain')
                payload=dict(p=p,gain=gain)
                if oracle is not None:
                    op=model.predict_proba(oracle)[:,1];os.append(op);payload['oracle']=op
                np.savez_compressed(path,**payload)
            ps.append(p);importances.append(gain)
            emit(dict(event='tree_seed_complete',cutoff=cut,name=name,seed=seed,
                      auc=metrics(v,{'p':p},cut,end)[0]['auc']))
        p=np.mean(ps,axis=0)
        for col,values in blends(anchor['incumbent'],p,name).items():preds[col]=values
        if os:preds[name+'_visible_oracle_diagnostic']=np.mean(os,axis=0)
        pd.DataFrame({'feature':xf.columns,'mean_gain':np.mean(importances,axis=0)}).sort_values('mean_gain',ascending=False).to_csv(folder/f'{name}_importance.csv',index=False)
        preds.to_csv(folder/'predictions.csv',index=False)
        rec=metrics(v,{c:preds[c].to_numpy() for c in preds if c not in ['txId','time_step','label']},cut,end)
        save_json(folder/'metrics.json',rec)
        emit(dict(event='tree_model_complete',cutoff=cut,name=name,seconds=time.time()-start,
                  scores={r['name']:r['auc'] for r in rec if r['name'].startswith(name) or r['name']=='incumbent'}))
    summarize(args.out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--old-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--models',default=','.join(MODELS));p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--folds',default='25:30,28:32,30:35');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    save_json(args.out/f'config_{time.time_ns()}.json',dict(args=vars(args),models=MODELS,base=BASE))
    for cut,end in [tuple(map(int,s.split(':'))) for s in args.folds.split(',')]:run_fold(args,cut,end)

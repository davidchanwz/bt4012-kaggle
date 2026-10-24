"""Node cache for the message network (+ an imputation experiment).

`prepare` is used by the final pipeline. It saves, for every node in the graph:
raw feature vectors, a 'visible' flag (labelled training rows and test rows),
whether features were supplied at all, the time step, the adjacency matrix and
344 neighbour-context columns. No fraud labels are used.

The rest of the file is an experiment that was NOT selected: a ridge regression
learns to predict the hidden features of unknown nodes from their neighbours,
and the imputed values are then used as graph features.
"""
from pathlib import Path
import argparse,json,time
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
from common import FEATS,BASE,controls,blends,metrics,save_json,emit,summarize
from graph_data import topology,relation_summaries,selected_columns,load
from graph_lgbm import matrix

# Count-like context columns that are log-transformed before the ridge model (experiment).
LOG_COLUMNS=[330,331,333,334,336,337,338,339,340,341]
CONFIG=dict(alpha=100.,tau=5.,outputs=165)


def prepare(data,cache,basecache):
    cache.mkdir(parents=True,exist_ok=True)
    if (cache/'context_manifest.json').exists():return
    t=pd.read_csv(data/'train.csv').astype({c:'float32' for c in FEATS})
    q=pd.read_csv(data/'test.csv').astype({c:'float32' for c in FEATS})
    t['is_test']=False;q['is_test']=True
    full=pd.concat([t,q],ignore_index=True);e=pd.read_csv(data/'txs_edgelist.csv')
    ids=pd.Index(np.unique(np.r_[full.txId.to_numpy(),e.to_numpy().ravel()]))
    a=sparse.csr_matrix((np.ones(len(e),np.float32),
        (ids.get_indexer(e.txId1),ids.get_indexer(e.txId2))),shape=(len(ids),len(ids)))
    a.sum_duplicates();a.data[:]=1
    row=ids.get_indexer(full.txId)
    _,comp=connected_components(a,directed=False)
    mapping=pd.DataFrame({'c':comp[row],'t':full.time_step}).groupby('c').t.first()
    period=mapping.reindex(comp).to_numpy()
    x=np.zeros((len(ids),165),np.float32);x[row]=full[FEATS].to_numpy(np.float32)
    # Same visibility rule as graph_data.py: features of unknown training rows are hidden.
    visible=np.zeros(len(ids),bool);visible[row[(full.label.notna()|full.is_test).to_numpy()]]=True
    supplied=np.zeros(len(ids),bool);supplied[row]=True
    z=x.copy();z[~visible]=0
    at=a.T.tocsr();one=np.ones(len(ids),np.float32)
    di=at@one;do=a@one;ci=at@visible.astype(np.float32);co=a@visible.astype(np.float32)
    mi=np.full_like(z,np.nan);mo=mi.copy()
    np.divide(at@z,ci[:,None],out=mi,where=ci[:,None]>0)
    np.divide(a@z,co[:,None],out=mo,where=co[:,None]>0)
    structs,_=topology(a)
    stats=np.stack([di,ci,np.divide(ci,di,out=np.zeros_like(ci),where=di>0),
                    do,co,np.divide(co,do,out=np.zeros_like(co),where=do>0)],axis=1)
    context=np.concatenate([mi,mo,stats,structs],axis=1).astype(np.float32)
    assert context.shape[1]==344
    np.save(cache/'context.npy',context);np.save(cache/'raw_vectors.npy',x)
    sparse.save_npz(cache/'adjacency.npz',a)
    pd.DataFrame({'txId':ids.to_numpy(),'time_step':period,'visible':visible,'supplied':supplied}).to_csv(cache/'nodes.csv',index=False)
    base=json.loads((basecache/'manifest.json').read_text())
    save_json(cache/'context_manifest.json',dict(input_sha256=base['input_sha256'],n_nodes=len(ids),n_inputs=344,
        definition='Incoming/outgoing means and coverage plus topology; no own vector or class-label inputs'))
    emit(dict(event='imputation_context_ready',shape=context.shape))


def preprocess(context,ix):
    x=np.array(context[ix],np.float32,copy=True)
    x[:,LOG_COLUMNS]=np.log1p(x[:,LOG_COLUMNS])
    return x


# Experiment: train a ridge model on unknown training nodes (their features are
# supplied but hidden) to predict features from neighbour context. No fraud labels.
def fit_imputer(nodes,context,raw,cut):
    support=(context[:,331]+context[:,334])>0
    fit=np.flatnonzero((nodes.supplied&~nodes.visible&nodes.time_step.le(cut)).to_numpy()&support)
    assert len(fit)>1000
    xs=StandardScaler();ys=StandardScaler()
    x=xs.fit_transform(preprocess(context,fit));x=np.nan_to_num(x,nan=0.)
    y=ys.fit_transform(np.array(raw[fit],np.float32,copy=True))
    weight=np.exp((nodes.time_step.iloc[fit].to_numpy()-cut)/CONFIG['tau'])
    weight/=weight.mean()
    model=Ridge(alpha=CONFIG['alpha'],solver='cholesky')
    model.fit(x,y,sample_weight=weight)
    return dict(model=model,x_scaler=xs,y_scaler=ys,cutoff=cut,config=CONFIG,n_fit=len(fit))


def impute(bundle,context,ix):
    output=[]
    for start in range(0,len(ix),8192):
        x=preprocess(context,ix[start:start+8192])
        x=np.nan_to_num(bundle['x_scaler'].transform(x),nan=0.)
        z=bundle['model'].predict(x)
        output.append(bundle['y_scaler'].inverse_transform(z).astype(np.float32))
    return np.concatenate(output)


# Experiment: rebuild the graph panel with imputed values for hidden nodes.
def build_panel(args,cut,end):
    folder=args.cache/f'cut_{cut}';folder.mkdir(exist_ok=True)
    path=folder/'panel.npy'
    if path.exists():return path
    nodes=pd.read_csv(args.cache/'nodes.csv')
    context=np.load(args.cache/'context.npy',mmap_mode='r')
    raw=np.load(args.cache/'raw_vectors.npy',mmap_mode='r')
    a=sparse.load_npz(args.cache/'adjacency.npz')
    start=time.time();bundle=fit_imputer(nodes,context,raw,cut)
    joblib.dump(bundle,folder/'imputer.joblib')
    visible=nodes.visible.to_numpy()
    support=(context[:,331]+context[:,334])>0
    fill=np.flatnonzero(~visible&support)
    z=np.array(raw,np.float32,copy=True);z[~visible]=0
    z[fill]=impute(bundle,context,fill)
    used=visible.copy();used[fill]=True
    rows,_,manifest=load(args.base_cache,'matched')
    row=pd.Index(nodes.txId).get_indexer(rows.txId)
    means,stats=relation_summaries(a,z,used,visible,row)
    structs,_=topology(a)
    panel=np.concatenate([np.asarray(raw[row]),*means,stats,structs[row]],axis=1).astype(np.float32)
    assert np.isfinite(z).all() and not np.isinf(panel).any()
    np.save(path,panel)
    # Held-out unknown attributes are diagnostic targets only, never model inputs.
    vi=np.flatnonzero((nodes.supplied&~nodes.visible&nodes.time_step.between(cut+1,end)).to_numpy()&support)
    diagnostics=dict(cutoff=cut,end=end,n_fit=bundle['n_fit'],n_imputed=len(fill),seconds=time.time()-start)
    if len(vi):
        y=np.asarray(raw[vi]);p=z[vi];scale=bundle['y_scaler'].scale_
        ci=np.asarray(context[vi,331]);co=np.asarray(context[vi,334])
        naive=(np.nan_to_num(context[vi,:165])*ci[:,None]+np.nan_to_num(context[vi,165:330])*co[:,None])/(ci+co)[:,None]
        diagnostics.update(n_validation_unknown=len(vi),
            ridge_normalized_rmse=float(np.sqrt(np.mean(((p-y)/scale)**2))),
            neighbour_mean_normalized_rmse=float(np.sqrt(np.mean(((naive-y)/scale)**2))),
            training_mean_normalized_rmse=float(np.sqrt(np.mean(((bundle['y_scaler'].mean_-y)/scale)**2))))
    save_json(folder/'diagnostics.json',diagnostics)
    emit(dict(event='learned_panel_ready',**diagnostics))
    return path


def run_fold(args,cut,end):
    folder=args.out/f'fold_{cut}_{end}';folder.mkdir(parents=True,exist_ok=True)
    path=build_panel(args,cut,end);panel=np.load(path,mmap_mode='r')
    rows,_,manifest=load(args.base_cache,'matched')
    fi=np.flatnonzero((~rows.is_test&rows.time_step.le(cut)).to_numpy())
    vi=np.flatnonzero((~rows.is_test&rows.time_step.between(cut+1,end)).to_numpy())
    f=rows.iloc[fi];v=rows.iloc[vi];anchor=controls(args.old_root,v,cut,end)
    cols=selected_columns('direct',manifest)
    xf=matrix(panel,fi,cols,manifest['columns']);xv=matrix(panel,vi,cols,manifest['columns'])
    ps=[]
    for seed in [0,1,2]:
        seedpath=folder/f'seed_{seed}.npy'
        if seedpath.exists():p=np.load(seedpath)
        else:
            model=lgb.LGBMClassifier(**BASE,random_state=seed)
            model.fit(xf,f.label.astype(int),sample_weight=np.exp((f.time_step.to_numpy()-cut)/5))
            p=model.predict_proba(xv)[:,1];np.save(seedpath,p)
        ps.append(p);emit(dict(event='learned_imputation_seed',cutoff=cut,seed=seed,
                              auc=metrics(v,{'p':p},cut,end)[0]['auc']))
    values={**anchor,**blends(anchor['incumbent'],np.mean(ps,axis=0),'learned_imputed_direct')}
    pd.DataFrame({**{c:v[c].to_numpy() for c in ['txId','time_step','label']},**values}).to_csv(folder/'predictions.csv',index=False)
    rec=metrics(v,values,cut,end);save_json(folder/'metrics.json',rec)
    emit(dict(event='learned_imputation_complete',cutoff=cut,end=end,scores={r['name']:r['auc'] for r in rec}))
    summarize(args.out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--base-cache',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--old-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--folds',default='25:30,28:32,30:35');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);args.cache.mkdir(parents=True,exist_ok=True)
    save_json(args.out/f'config_{time.time_ns()}.json',dict(args=vars(args),imputer=CONFIG,base=BASE))
    prepare(args.data,args.cache,args.base_cache)
    for cut,end in [tuple(map(int,s.split(':'))) for s in args.folds.split(',')]:run_fold(args,cut,end)

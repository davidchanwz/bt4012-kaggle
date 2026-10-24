"""Stage T: TabM on the raw 165 features (final model 'tabm_ple_uniform').

TabM (Gorishniy et al., 2024) is an MLP that trains k=32 ensemble members at
once with mostly shared weights. Numerical features go through piecewise-linear
embeddings (rtdl_num_embeddings, 48 bins) before the network.

Steps for one model:
1. Preprocessing, fitted on the training rows only: drop constant columns,
   quantile-transform to a normal distribution, clip to [-5, 5].
2. Inner epoch selection: train on steps <= cutoff-3, validate on the next 3
   steps, keep the best epoch (>= 5, patience 12). For the final model this is
   train <= 32 / validate 33-35, which selected 17 epochs.
3. Refit on all labelled rows up to the cutoff for that number of epochs, with
   each seed. The final run uses cutoff 35 and seeds 0-4 and predicts the test set.
4. Save the weights, bins and preprocessing, reload them and check the
   predictions are identical.

Final run (from the notebook):
  python src/tabm/train_tabm.py --data DATA --root . --out runs/tabm \
      --models tabm_ple_uniform --final tabm_ple_uniform --seeds 0,1,2,3,4 --device cuda
Outputs: runs/tabm/tabm_ple_uniform/fold_35_49/seed_*/predictions.csv (txId, time_step, p).
"""
from pathlib import Path
import copy,json,time,random,importlib.metadata
import numpy as np
import pandas as pd
import joblib
import torch
import tabm
import rtdl_num_embeddings as embeddings
from sklearn.preprocessing import QuantileTransformer
from common import *
torch.set_num_threads(4)

# Fit preprocessing on training rows only: drop constant columns + quantile normalisation.
def prep(frame):
    x=frame[FEATURES].to_numpy(np.float32)
    keep=np.ptp(x,axis=0)>0
    assert keep.any()
    q=QuantileTransformer(n_quantiles=min(1000,max(10,len(x)//30)),output_distribution='normal',subsample=10**9,random_state=4012)
    q.fit(x[:,keep]);return dict(keep=keep,transformer=q)
def transform(bundle,frame):
    return np.asarray(np.clip(bundle['transformer'].transform(frame[FEATURES].to_numpy(np.float32)[:,bundle['keep']]),-5,5),np.float32)
# Build TabM; 'ple' variants use piecewise-linear embeddings with bins computed
# from the training data (or reloaded bins when restoring a saved model).
def build(x,kind,cfg,bins=None):
    if 'ple' in kind:
        if bins is None:bins=embeddings.compute_bins(torch.from_numpy(x),n_bins=min(cfg['n_bins'],len(x)-1))
        num=embeddings.PiecewiseLinearEmbeddings(bins,d_embedding=cfg['d_embedding'],activation=False,version='B')
    else:num=None
    model=tabm.TabM.make(n_num_features=x.shape[1],d_out=1,num_embeddings=num,k=cfg['k'],n_blocks=cfg['n_blocks'],d_block=cfg['d_block'],dropout=cfg['dropout'])
    return model,bins
@torch.inference_mode()
# Prediction = mean of the 32 members' probabilities.
def predict(model,x,device,batch=512):
    model.eval();ps=[]
    for start in range(0,len(x),batch):
        logits=model(torch.as_tensor(x[start:start+batch],device=device)).squeeze(-1)
        ps.append(torch.sigmoid(logits).mean(1).cpu().numpy())
    return np.concatenate(ps)
# Train one model. epochs=None -> inner selection mode (early stopping on v);
# otherwise train for exactly `epochs` epochs and predict v.
def fit(f,v,cut,seed,kind,folder,device,cfg,epochs=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    b=prep(f);x=transform(b,f);xv=transform(b,v)
    model,bins=build(x,kind,cfg);model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    xt=torch.as_tensor(x,device=device);yt=torch.as_tensor(f.label.to_numpy(np.float32),device=device)
    # The final 'uniform' model uses equal weights for all training rows.
    tau=None if 'uniform' in kind else 5
    wt=torch.as_tensor(weights(f,cut,tau),dtype=torch.float32,device=device)
    rng=np.random.default_rng(seed+100000)
    best=-np.inf;chosen=0;best_state=None;history=[];started=time.time()
    for epoch in range(1,(epochs or cfg['max_epochs'])+1):
        model.train();losses=[]
        for ix in np.array_split(rng.permutation(len(f)),int(np.ceil(len(f)/cfg['batch_size']))):
            idx=torch.as_tensor(ix,device=device)
            logits=model(xt[idx]).squeeze(-1)
            # Train every implicit member separately; do not first average logits.
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,yt[idx,None].expand_as(logits),reduction='none').mean(1)
            loss=(loss*wt[idx]).mean()
            assert torch.isfinite(loss),'Non-finite training loss'
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(loss.item())
        row=dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.time()-started)
        if epochs is None:
            # Inner selection: track AUC on the validation steps and keep the best epoch.
            row['inner_auc']=metrics(v.label,predict(model,xv,device))['auc']
            if epoch>=cfg['min_epochs'] and row['inner_auc']>best:
                best=row['inner_auc'];chosen=epoch;best_state=copy.deepcopy({k:t.cpu() for k,t in model.state_dict().items()})
        history.append(row)
        if epoch==1 or epoch%5==0:emit(model=kind,seed=seed,cutoff=cut,**row)
        if epochs is None and chosen and epoch-chosen>=cfg['patience']:break
    if epochs is None:
        assert chosen>0;model.load_state_dict(best_state)
    else:chosen=epochs
    p=predict(model,xv,device)
    # Save everything needed to reproduce the predictions, then reload and compare.
    torch.save(model.state_dict(),folder/'weights.pt');joblib.dump(b,folder/'preprocessing.joblib')
    torch.save(bins,folder/'bins.pt')
    save(folder/'training.json',dict(seed=seed,cutoff=cut,kind=kind,epochs=chosen,cfg=cfg,history=history))
    restored_b=joblib.load(folder/'preprocessing.joblib');restored_bins=torch.load(folder/'bins.pt',map_location='cpu',weights_only=True)
    restored,_=build(x,kind,cfg,bins=restored_bins);restored.load_state_dict(torch.load(folder/'weights.pt',map_location='cpu',weights_only=True));restored.to(device)
    p2=predict(restored,transform(restored_b,v),device)
    diff=float(np.max(np.abs(p-p2)));assert diff<=1e-6
    save(folder/'reload_check.json',dict(max_abs_difference=diff,passed=True))
    out=v[['txId','time_step']].copy();out['p']=p;out.to_csv(folder/'predictions.csv',index=False)
    del model,restored,xt,yt,wt
    if device.type=='cuda':torch.cuda.empty_cache()
    return chosen,p
# One fold + seed: choose the epoch count on the inner window (once per fold),
# then refit on all labelled rows <= cut and predict steps cut+1..end
# (end=49 means: predict the test set).
def one(args,train,test,kind,cut,end,seed,device,cfg):
    f=prefix(train,cut);v=test if end==49 else query(train,cut,end)
    outer=args.out/kind/f'fold_{cut}_{end}'/f'seed_{seed}'
    if (outer/'predictions.csv').exists():return
    inner=args.out/kind/f'fold_{cut}_{end}'/'inner'
    if (inner/'training.json').exists():epochs=json.loads((inner/'training.json').read_text())['epochs']
    else:
        epochs,_=fit(prefix(train,cut-3),query(train,cut-3,cut),cut-3,4012,kind,inner,device,cfg)
    _,p=fit(f,v,cut,seed,kind,outer,device,cfg,epochs=epochs)
    emit(event='outer_complete',kind=kind,cutoff=cut,end=end,seed=seed,epochs=epochs,auc=None if end==49 else metrics(v.label,p)['auc'])
def main(a):
    device=torch.device(a.device)
    if device.type=='cuda':assert torch.cuda.is_available(),'Enable a Kaggle GPU accelerator before running.'
    cfg=TABM.copy()
    # --smoke: tiny configuration for a quick code check, never used for submissions.
    if a.smoke:cfg.update(k=2,n_blocks=2,d_block=32,n_bins=8,d_embedding=4,max_epochs=5,min_epochs=2,patience=2,batch_size=128)
    models=a.models.split(',') if a.models else ['tabm_ple_tau5','tabm_ple_uniform','tabm_plain_tau5']
    assert set(models)<=set(MODELS)
    runtime=dict(torch=torch.__version__,tabm=importlib.metadata.version('tabm'),embeddings=importlib.metadata.version('rtdl_num_embeddings'),numpy=np.__version__,sklearn=importlib.metadata.version('scikit-learn'),device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None)
    guard(a.out/'tabm_manifest',signature(a.data,a.root,dict(models=models,cfg=cfg,runtime=runtime,smoke=a.smoke)))
    save(a.out/'tabm_runtime.json',runtime)
    train,test,_=load(a.data)
    if a.smoke:
        # A small real-data prefix check, never eligible as a competition candidate.
        f=prefix(train,20).groupby('label',group_keys=False).sample(n=128,random_state=4012)
        v=query(train,20,25).groupby('label',group_keys=False).sample(n=64,random_state=4012)
        for kind in models:fit(f,v,20,0,kind,a.out/'smoke'/kind,device,cfg,epochs=2)
        return
    # --final MODEL: fit the chosen model on steps 1-35 for every seed and predict the test set.
    if a.final:
        assert a.final in models
        for seed in a.seeds:one(a,train,test,a.final,35,49,seed,device,cfg)
    else:
        for cut,end in a.folds:
            for kind in models:
                for seed in a.seeds:one(a,train,test,kind,cut,end,seed,device,cfg)
if __name__=='__main__':main(parse(common_args()))

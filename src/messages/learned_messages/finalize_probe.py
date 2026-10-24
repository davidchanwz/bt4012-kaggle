"""Stage 4 final fit: train the message network on steps 1-35 and finish ensemble B.

1. Checks the caches were built from the same input files (SHA-256).
2. Selects the number of supervised epochs on an inner window
   (train <= 32, validate 33-35); the original run selected 16.
3. Refits 5 seeds (0-4) with pretraining on all labelled steps 1-35.
4. Blends the mean network probability n with the stage-3 output m (--anchor):
     B = sigmoid(0.8*logit(m) + 0.2*logit(n))
   and writes runs/final/submission_neural_graph_probe.csv (public 0.95261).
5. Reloads every saved model and the normaliser and checks the predictions and
   the CSV are reproduced exactly.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import torch
import joblib
from scipy.special import expit,logit
import message_model as base


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def run(args):
    """Fit the final message network (5 seeds), blend it into ensemble B and verify the output."""
    args.out.mkdir(parents=True,exist_ok=True)
    hashes={name:sha(args.data/name) for name in ['train.csv','test.csv','txs_edgelist.csv']}
    for manifest in [args.node_cache/'context_manifest.json',args.graph_cache/'manifest.json']:
        assert json.loads(manifest.read_text())['input_sha256']==hashes
    # The selected variant (pretrained network, 20% log-odds blend) is fixed in the lock file.
    lock=json.loads(args.lock.read_text());assert lock['candidate']=='pretrained_logit20'
    nodes,raw,a=base.read_source(args.node_cache,args.graph_cache)
    schedule=args.out/'epoch_selection.json'
    if schedule.exists():selection=json.loads(schedule.read_text())
    else:
        # Inner epoch selection: train on steps <= 32, validate on 33-35.
        inner,_=base.prepare(nodes,raw,a,32,35,include_targets=True)
        _,selection=base.train(inner,32,4012,True,True,tag='final_inner')
        base.save_json(schedule,selection)
        del inner
    epochs=selection['best_epoch']
    # Final graphs: train on steps <= 35, predict test steps 36-49.
    groups,transformer=base.prepare(nodes,raw,a,35,49,include_targets=True)
    joblib.dump(transformer,args.out/'transformer.joblib')
    test=pd.read_csv(args.data/'test.csv',usecols=['txId','index'])
    sample=pd.read_csv(args.data/'sample_submission.csv')
    anchor=pd.read_csv(args.anchor)
    np.testing.assert_array_equal(test['index'],sample['index'])
    np.testing.assert_array_equal(anchor['index'],sample['index'])
    models=args.out/'models';models.mkdir(exist_ok=True)
    seeds=lock['final_seeds'];predictions=[]
    for seed in seeds:
        model_path=models/f'neural_graph_seed_{seed}.pt'
        if model_path.exists():
            model=base.MessageNet(True)
            model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True))
        else:
            model,history=base.train(groups,35,seed,True,True,epochs=epochs,tag='final_outer')
            torch.save(model.state_dict(),model_path)
            base.save_json(models/f'neural_graph_seed_{seed}_history.json',history)
        d=base.predict(model,groups,36,49).set_index('txId').loc[test.txId]
        p=d.p.to_numpy();predictions.append(p)
        base.emit(event='final_seed_complete',seed=seed,rows=len(p),epochs=epochs)
    # Average the 5 seeds, then blend 20% in log-odds space with the previous ensemble.
    neural=np.mean(predictions,0)
    target=expit(.8*logit(np.clip(anchor.target.to_numpy(),1e-7,1-1e-7))
                 +.2*logit(np.clip(neural,1e-7,1-1e-7)))
    destination=args.out/'submission_neural_graph_probe.csv'
    pd.DataFrame({'index':sample['index'],'target':target}).to_csv(destination,index=False)
    np.savez_compressed(args.out/'component_predictions.npz',anchor=anchor.target.to_numpy(),neural=neural,
                       **{f'neural_seed_{s}':p for s,p in zip(seeds,predictions)})
    # Reproducibility check: reload the normaliser and every model and re-predict.
    restored=joblib.load(args.out/'transformer.joblib')
    reload_groups,_=base.prepare(nodes,raw,a,35,49,transformer=restored)
    checks=[];reloaded=[]
    for seed,p in zip(seeds,predictions):
        model=base.MessageNet(True)
        model.load_state_dict(torch.load(models/f'neural_graph_seed_{seed}.pt',map_location='cpu',weights_only=True))
        p2=base.predict(model,reload_groups,36,49).set_index('txId').loc[test.txId].p.to_numpy()
        np.testing.assert_array_equal(p,p2)
        checks.append(dict(seed=seed,max_abs_difference=float(abs(p-p2).max())))
        reloaded.append(p2)
    target2=expit(.8*logit(np.clip(anchor.target.to_numpy(),1e-7,1-1e-7))
                  +.2*logit(np.clip(np.mean(reloaded,0),1e-7,1-1e-7)))
    np.testing.assert_array_equal(target,target2)
    disk=pd.read_csv(destination)
    assert list(disk)==['index','target'] and len(disk)==len(sample)==15329
    assert disk['index'].is_unique and np.isfinite(disk.target).all() and disk.target.between(0,1).all()
    np.testing.assert_array_equal(disk['index'],sample['index'])
    np.testing.assert_allclose(disk.target,target2,rtol=0,atol=2e-15)
    base.save_json(args.out/'submission_checks.json',dict(rows=len(disk),training_rows=int(nodes.label.notna().sum()),
        feature_dimension=165,hidden_width=base.CFG['width'],layers=2,epochs=epochs,seeds=seeds,
        all_models_and_transformer_reloaded=True,seed_checks=checks,formula_exactly_reproduced=True,
        correct_sample_order=True,unique_indices=True,finite_bounded_probabilities=True,
        input_sha256=hashes,sample_submission_sha256=sha(args.data/'sample_submission.csv'),
        anchor_source=str(args.anchor),anchor_sha256=sha(args.anchor),
        public_auc=None,private_auc=None,submission_sha256=sha(destination),
        experimental_only=True,long_horizon_regression=True,
        rank_correlation_to_anchor=float(pd.Series(target).rank().corr(anchor.target.rank())),
        mean_abs_change_to_anchor=float(np.abs(target-anchor.target.to_numpy()).mean())))
    base.save_json(args.out/'feature_policy.json',dict(normalizer_fit='known training prefix only',
        observed_vectors='supplied known/query rows',missing_nodes='edges retained with learned missing state',
        fraud_training='known periods1–35',pretraining='feature reconstruction from supplied periods1–35',
        pretraining_targets_are_not_fraud_labels=True,source_cfg=base.CFG))
    base.emit(event='final_checks_passed',rows=len(disk),file=str(destination))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['data','node-cache','graph-cache','anchor','lock','out']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())

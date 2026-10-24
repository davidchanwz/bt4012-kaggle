"""Stage 3 of ensemble B: refit the graph LightGBM on all labelled steps 1-35.

1. Builds/loads the 'matched' graph panel (1,187 features, see graph_data.py).
2. Fits 5 seeds of recency-weighted LightGBM (same settings as the reference).
3. Blends the mean graph probability g with the stage-2 output u (--anchor):
     submission_missing_neighbours_conservative.csv = 0.8*u + 0.2*g   <- USED (public 0.95210)
     submission_missing_neighbours_blend.csv        = sigmoid(0.8*logit(u)+0.2*logit(g))
   The probability blend was preferred after extra validation checks.
4. Saves the models, reloads them and checks the predictions and files match exactly.
"""
from pathlib import Path
import argparse,hashlib,json,shutil
import numpy as np
import pandas as pd
from scipy.special import expit,logit
import lightgbm as lgb
from common import BASE,save_json,emit
from graph_data import build,load,selected_columns,sha
from graph_lgbm import matrix


def run(args):
    """Refit the locked graph candidate on steps 1-35, blend with the anchor and verify everything."""
    manifest=build(args.data,args.cache)
    lock=json.loads(args.lock.read_text())
    assert lock['candidate']=='matched_relations_logit20'
    rows,panel,_=load(args.cache,lock['graph_policy'])
    # fi = labelled training rows (steps 1-35), vi = test rows.
    fi=np.flatnonzero(~rows.is_test.to_numpy());vi=np.flatnonzero(rows.is_test.to_numpy())
    cols=selected_columns(lock['feature_set'],manifest)
    xf=matrix(panel,fi,cols,manifest['columns']);xv=matrix(panel,vi,cols,manifest['columns'])
    sample=pd.read_csv(args.data/'sample_submission.csv');anchor=pd.read_csv(args.anchor)
    np.testing.assert_array_equal(rows.iloc[vi]['index'].astype(int),sample['index'])
    np.testing.assert_array_equal(anchor['index'],sample['index'])
    args.out.mkdir(parents=True,exist_ok=True)
    modeldir=args.out/'models';modeldir.mkdir(exist_ok=True)
    ps=[];gains=[]
    for seed in lock['final_seeds']:
        path=modeldir/f'graph_seed_{seed}.txt'
        if path.exists():
            model=lgb.Booster(model_file=str(path));p=model.predict(xv,num_threads=4)
            gain=model.feature_importance(importance_type='gain')
        else:
            model=lgb.LGBMClassifier(**BASE,random_state=seed)
            # Recency weights exp((t - 35) / 5), as in the reference model.
            model.fit(xf,rows.label.iloc[fi].astype(int),sample_weight=np.exp((rows.time_step.iloc[fi].to_numpy()-35)/5))
            p=model.predict_proba(xv)[:,1]
            model.booster_.save_model(str(path));gain=model.booster_.feature_importance(importance_type='gain')
        ps.append(p);gains.append(gain)
        emit(dict(event='final_graph_seed',seed=seed,n_features=len(cols)))
    graph=np.mean(ps,axis=0);a=anchor.target.to_numpy()
    clip=lock['clip']
    target=expit(.8*logit(np.clip(a,*clip))+.2*logit(np.clip(graph,*clip)))
    sub=pd.DataFrame({'index':sample['index'].astype(int),'target':target})
    destination=args.out/'submission_missing_neighbours_blend.csv'
    sub.to_csv(destination,index=False)
    # The blend used by the final pipeline: 80% previous ensemble + 20% graph model.
    conservative=.8*a+.2*graph
    conservative_path=args.out/'submission_missing_neighbours_conservative.csv'
    pd.DataFrame({'index':sample['index'].astype(int),'target':conservative}).to_csv(conservative_path,index=False)
    np.savez_compressed(args.out/'component_predictions.npz',anchor=a,graph=graph,
                       **{f'graph_seed_{s}':p for s,p in zip(lock['final_seeds'],ps)})
    shutil.copy2(args.anchor,args.out/'anchor_unlabelled_blend.csv')
    save_json(args.out/'feature_schema.json',dict(graph_policy=lock['graph_policy'],feature_set=lock['feature_set'],
        columns=list(xf.columns),precision='float32',base_parameters=BASE,recency_tau=5,
        test_feature_sha256=hashlib.sha256(xv.to_numpy().tobytes(order='C')).hexdigest()))
    pd.DataFrame({'feature':xf.columns,'mean_gain':np.mean(gains,axis=0)}).sort_values('mean_gain',ascending=False).to_csv(args.out/'feature_importance.csv',index=False)
    # Reproducibility check: reload every saved model and confirm identical predictions.
    reload_ps=[];checks=[]
    for seed,p in zip(lock['final_seeds'],ps):
        restored=lgb.Booster(model_file=str(modeldir/f'graph_seed_{seed}.txt'))
        assert restored.feature_name()==list(xv.columns)
        p2=restored.predict(xv,num_threads=4)
        np.testing.assert_array_equal(p2,p)
        reload_ps.append(p2);checks.append(dict(seed=seed,max_abs_difference=float(abs(p-p2).max())))
    reloaded=expit(.8*logit(np.clip(a,*clip))+.2*logit(np.clip(np.mean(reload_ps,axis=0),*clip)))
    np.testing.assert_array_equal(target,reloaded)
    disk=pd.read_csv(destination)
    assert list(disk)==['index','target'] and len(disk)==15329
    assert disk['index'].is_unique and np.isfinite(disk.target).all()
    assert disk.target.between(0,1).all()
    np.testing.assert_array_equal(disk['index'],sample['index'])
    np.testing.assert_allclose(disk.target,reloaded,rtol=0,atol=2e-15)
    conservative_reload=.8*a+.2*np.mean(reload_ps,axis=0)
    np.testing.assert_array_equal(conservative,conservative_reload)
    conservative_disk=pd.read_csv(conservative_path)
    assert list(conservative_disk)==['index','target'] and len(conservative_disk)==15329
    assert conservative_disk['index'].is_unique and np.isfinite(conservative_disk.target).all()
    assert conservative_disk.target.between(0,1).all()
    np.testing.assert_array_equal(conservative_disk['index'],sample['index'])
    np.testing.assert_allclose(conservative_disk.target,conservative_reload,rtol=0,atol=2e-15)
    result=dict(rows=len(disk),training_rows=len(fi),n_features=len(cols),schema=list(disk),
        correct_index_order=True,finite_bounded_probabilities=True,all_test_rows_reloaded=True,
        formula_reproduced=True,seed_checks=checks,
        rank_correlation_to_incumbent=float(pd.Series(target).rank().corr(pd.Series(a).rank())),
        mean_abs_change_from_incumbent=float(np.abs(target-a).mean()),
        input_sha256={**manifest['input_sha256'],'sample_submission.csv':sha(args.data/'sample_submission.csv')},
        anchor_source=str(args.anchor.resolve()),anchor_sha256=sha(args.anchor),
        submission_sha256=sha(destination),public_auc=None,private_auc=None,
        recommended_probability_blend=dict(file=conservative_path.name,
            formula='0.8*incumbent_probability+0.2*mean_graph_probability',
            submission_sha256=sha(conservative_path),all_test_rows_reloaded=True,formula_reproduced=True,
            finite_bounded_probabilities=True,correct_index_order=True,
            rank_correlation_to_incumbent=float(pd.Series(conservative).rank().corr(pd.Series(a).rank())),
            mean_abs_change_from_incumbent=float(np.abs(conservative-a).mean()),
            selected_after_additional_checks=True))
    save_json(args.out/'submission_checks.json',result)
    emit(dict(event='final_graph_checks_passed',rows=len(disk),n_features=len(cols),
              rank_correlation=result['rank_correlation_to_incumbent']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--anchor',type=Path,required=True)
    p.add_argument('--lock',type=Path,default=Path(__file__).with_name('candidate_lock.json'))
    p.add_argument('--out',type=Path,required=True);run(p.parse_args())

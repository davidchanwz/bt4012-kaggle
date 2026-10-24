"""Final fit of stage 1 + stage 2 of ensemble B on all labelled steps 1-35.

1. Fits the recency-weighted reference LightGBM with 5 seeds (0-4) and averages them.
2. Fits the selected extra candidate from the lock file (selected_lock.json ->
   'logit_annotation3', the 3-class annotation-status model), also 5 seeds.
3. Blends them in log-odds space: u = sigmoid(0.8*logit(reference) + 0.2*logit(annotation))
   and writes runs/annotation/submission_unlabelled_blend.csv (public 0.95164).

Every model is saved, and every submission is checked for row count, index order
and valid probabilities. Candidates were chosen on validation only; this script
never looks at test labels and never uploads anything.
"""
from pathlib import Path
import argparse,json,hashlib,platform
import numpy as np,pandas as pd
import lightgbm as lgb
from recency_lab import load,FEATS,BASE,model_and_data
from domain_probe import adapt_weights
from rank_probe import fine_tune
from check_finalists import blend


def run(data,cache,out,lock):
 """Fit the reference and locked candidates on steps 1-35, blend, and write checked submission files."""
 out.mkdir(parents=True,exist_ok=True);models=out/'models';models.mkdir(exist_ok=True)
 # Which extra model(s) to fit; fixed in advance in the lock file.
 names=json.loads(lock.read_text())['candidates']
 f=load(data,cache).astype({c:'float32' for c in FEATS})
 # F = all labelled training rows (steps 1-35); T = test rows (steps 36-49).
 F=f[(~f.is_test)&f.label.notna()];T=f[f.is_test]
 sample=pd.read_csv(data/'sample_submission.csv')
 assert np.array_equal(T['index'].values,sample['index'].values)
 # Optional: if the earlier recency submission is present, check it is reproduced.
 reference_path=data/'submission_lgbm_recency.csv'
 reference=pd.read_csv(reference_path) if reference_path.exists() else None
 if reference is not None:
  assert np.array_equal(reference['index'].values,sample['index'].values)
 base_ps=[];parts={name:[] for name in names}
 dw=None
 if any(name.endswith('domain') for name in names):
  dw,diag=adapt_weights(F,T[FEATS],35)
  (out/'final_domain_diagnostics.json').write_text(json.dumps(diag,indent=2))
 # Five seeds per model; seed choice alone moved hard-fold AUC by ~0.01.
 for seed in range(5):
  bp,bm=model_and_data('reference',F,T,seed,35,f);base_ps.append(bp)
  bm.booster_.save_model(str(models/f'reference_seed_{seed}.txt'))
  for name in names:
   core=name.removeprefix('blend_').removeprefix('logit_')
   if core=='domain':
    m=lgb.LGBMClassifier(**BASE,random_state=seed)
    m.fit(F[FEATS],F.label.astype(int),sample_weight=dw)
    p=m.predict_proba(T[FEATS])[:,1]
   elif core=='rank_finetune':p,m=fine_tune(F,T,seed,35,bm)
   else:p,m=model_and_data(core,F,T,seed,35,f)
   parts[name].append(p)
   booster=m.booster_ if hasattr(m,'booster_') else m
   booster.save_model(str(models/f'{core}_seed_{seed}.txt'))
  print('Completed final seed',seed,flush=True)
 base=np.mean(base_ps,axis=0)
 difference=None if reference is None else float(abs(base-reference.target.values).max())
 if difference is not None:
  assert difference<1e-12,f'Reference no longer reproduces: {difference}'
 all_preds={'reference':base}
 # Average the seeds of each model, then blend with the reference (see check_finalists.blend).
 for name,ps in parts.items():all_preds[name]=blend(name,base,np.mean(ps,axis=0))
 file_names={'reference':'submission_reference_control.csv',
             'logit_annotation3':'submission_unlabelled_blend.csv',
             'blend_domain':'submission_future_similarity_blend.csv',
             'rank_finetune':'submission_rank_finetune.csv',
             'blend_leaf63':'submission_capacity_blend.csv'}
 checks={'reference_max_abs_difference':difference,'python':platform.python_version(),
         'lightgbm':lgb.__version__,'submissions':{}}
 for name,p in all_preds.items():
  assert len(p)==len(sample) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
  path=out/file_names.get(name,f'submission_{name}.csv')
  pd.DataFrame({'index':sample['index'],'target':p}).to_csv(path,index=False)
  reread=pd.read_csv(path)
  assert reread.columns.tolist()==['index','target']
  assert np.array_equal(reread['index'],sample['index'])
  checks['submissions'][name]={'file':path.name,'rows':len(p),'mean':float(p.mean()),
    'rank_correlation_with_reference':float(pd.Series(p).rank().corr(pd.Series(base).rank())),
    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 (out/'submission_checks.json').write_text(json.dumps(checks,indent=2))
 print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True)
 a.add_argument('--cache',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
 a.add_argument('--lock',type=Path,default=Path(__file__).with_name('finalist_lock.json'))
 args=a.parse_args();run(args.data,args.cache,args.out,args.lock)

"""Validation helper + an experiment that was not selected.

Used by section 8 of the final notebook with --controls-only: it refits the
earlier stages of ensemble B (reference, annotation-status model, graph model)
on the temporal folds and saves their predictions, which message_model.py then
uses as the comparison baseline.

The experiment itself (not selected) adds within-period percentile ranks of every
feature, to test whether relative values transfer better across time than raw
values. The plan was written in PROTOCOL.md before fitting.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, average_precision_score

FEATURES = [f'feat_{i}' for i in range(1, 166)]
EXTRA = [f'extra_feat_{i}' for i in range(1, 166)]
BASE = dict(n_estimators=500, learning_rate=.03, num_leaves=31,
            subsample=.8, subsample_freq=1, colsample_bytree=.5,
            verbose=-1, n_jobs=4)


def emit(**record):
    print(json.dumps(record), flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def load(data):
    train = pd.read_csv(data / 'train.csv')
    test = pd.read_csv(data / 'test.csv')
    train['is_test'] = False
    test['is_test'] = True
    full = pd.concat([train, test], ignore_index=True)
    return full.astype({name: 'float32' for name in FEATURES})


def query_rows(full):
    return full.loc[full.is_test | full.label.notna()].copy().reset_index(drop=True)


# Within-period percentile rank of every feature (experiment).
def ranked(rows):
    values = rows.groupby('time_step', sort=False)[FEATURES].rank(method='average', pct=True)
    values.columns = EXTRA
    return pd.concat([rows[FEATURES], values.astype('float32')], axis=1)


# Control: raw features duplicated, same width as raw + ranks (experiment).
def duplicated(rows):
    extra = rows[FEATURES].copy()
    extra.columns = EXTRA
    return pd.concat([rows[FEATURES], extra], axis=1)


def check_transform():
    toy = pd.DataFrame({name: [1., 1., 4., 100., 999.] for name in FEATURES})
    toy['time_step'] = [1, 1, 1, 2, 1]
    toy['label'] = [0., 1., 0., np.nan, np.nan]
    toy['is_test'] = [False, False, False, True, False]
    q = query_rows(toy)
    a = ranked(q)
    np.testing.assert_allclose(a.extra_feat_1, [.5, .5, 1., 1.])
    changed = toy.copy()
    changed.loc[changed.label.notna(), 'label'] = 1 - changed.loc[changed.label.notna(), 'label']
    np.testing.assert_array_equal(a.to_numpy(), ranked(query_rows(changed)).to_numpy())
    changed = toy.copy()
    changed.loc[4, FEATURES] = -12345
    np.testing.assert_array_equal(a.to_numpy(), ranked(query_rows(changed)).to_numpy())
    changed = toy.copy()
    changed.loc[3, FEATURES] = -678
    np.testing.assert_array_equal(a.iloc[:3].to_numpy(), ranked(query_rows(changed)).iloc[:3].to_numpy())
    assert len(q) == 4 and q.is_test.sum() == 1
    constant = q.copy()
    constant[FEATURES] = 2.
    np.testing.assert_allclose(ranked(constant).extra_feat_1, [2/3, 2/3, 2/3, 1.])
    return dict(ties=True, constant_features=True, fraud_label_value_invariance=True,
                hidden_unknown_vector_invariance=True, period_isolation=True,
                unlabelled_test_queries_included=True)


def fit_predict(name, full, rows, x, fi, vi, cut, seed, folder, graph=None):
    path = folder / f'{name}_seed_{seed}.npz'
    if path.exists():
        saved = np.load(path)
        np.testing.assert_array_equal(saved['txId'], rows.txId.iloc[vi])
        return saved['p']
    started = time.time()
    f, v = rows.iloc[fi], rows.iloc[vi]
    weights = np.exp((f.time_step.to_numpy() - cut) / 5)
    params = {**BASE, 'random_state': seed}
    if name == 'annotation':
        u = full.loc[(~full.is_test) & full.time_step.le(cut)]
        labels = u.label.fillna(2).astype(int)
        weights = np.exp((u.time_step.to_numpy() - cut) / 5)
        unknown = labels.eq(2).to_numpy()
        weights[unknown] *= min(1., weights[~unknown].sum() / max(weights[unknown].sum(), 1e-12))
        model = lgb.LGBMClassifier(**params, objective='multiclass', num_class=3)
        model.fit(u[FEATURES], labels, sample_weight=weights)
        allp = model.predict_proba(v[FEATURES])
        p = allp[:, 1] / np.maximum(allp[:, 0] + allp[:, 1], 1e-15)
    else:
        if name == 'graph':
            panel, names = graph
            xf = pd.DataFrame(np.asarray(panel[fi], dtype=np.float32), columns=names)
            xv = pd.DataFrame(np.asarray(panel[vi], dtype=np.float32), columns=names)
        else:
            xf, xv = x.iloc[fi], x.iloc[vi]
        model = lgb.LGBMClassifier(**params)
        model.fit(xf, f.label.astype(int), sample_weight=weights)
        p = model.predict_proba(xv)[:, 1]
    gain = model.booster_.feature_importance(importance_type='gain')
    np.savez_compressed(path, txId=v.txId.to_numpy(), p=p, gain=gain)
    emit(event='seed_complete', model=name, cutoff=cut, seed=seed,
         auc=roc_auc_score(v.label, p), seconds=round(time.time()-started, 2))
    return p


def logit_aux(reference, annotation):
    return expit(.8 * logit(np.clip(reference, 1e-15, 1-1e-15))
                 + .2 * logit(np.clip(annotation, 1e-15, 1-1e-15)))


def existing_predictions(old_root, cut, end, ids):
    path = old_root / f'combined_fold_{cut}_{end}.csv'
    if not path.exists():
        return None
    old = pd.read_csv(path).set_index('txId')
    if not pd.Index(ids).isin(old.index).all():
        return None
    old = old.loc[ids]
    return {name: old[col].to_numpy() for name, col in
            [('reference', 'reference'), ('unknown_blend', 'incumbent'),
             ('graph', 'matched_relations'), ('incumbent', 'matched_relations_prob20')]}


def score(d, cut, end, group='pooled'):
    if d.label.nunique() < 2:
        return []
    base = roc_auc_score(d.label, d.incumbent)
    records = []
    for name in ['reference', 'unknown_blend', 'incumbent', 'duplicate_raw',
                 'duplicate_blend50', 'period_rank', 'period_rank_blend50']:
        if name not in d:
            continue
        p = d[name]
        auc = roc_auc_score(d.label, p)
        records.append(dict(cutoff=cut, end=end, group=group, name=name, n=len(d),
                            illicit=int(d.label.sum()), auc=auc, delta_vs_incumbent=auc-base,
                            ap=average_precision_score(d.label, p)))
    return records


def evaluate(args):
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / 'implementation_checks.json', check_transform())
    full = load(args.data)
    rows = query_rows(full)
    xr = ranked(rows)
    xd = duplicated(rows)
    gm = pd.read_csv(args.graph_cache / 'rows.csv')
    np.testing.assert_array_equal(rows.txId, gm.txId)
    graph = (np.load(args.graph_cache / 'matched.npy', mmap_mode='r'),
             json.loads((args.graph_cache / 'manifest.json').read_text())['columns'])
    save_json(args.out / 'config.json', dict(base=BASE, recency_tau=5, seeds=args.seeds,
                cutoffs=args.cutoffs, features=FEATURES + EXTRA,
                data_sha256={name: sha(args.data/name) for name in ['train.csv', 'test.csv']},
                protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md'))))
    records, periods, horizons = [], [], []
    for cut, end in [tuple(map(int, pair.split(':'))) for pair in args.cutoffs.split(',')]:
        folder = args.out / f'fold_{cut}_{end}'
        folder.mkdir(exist_ok=True)
        fi = np.flatnonzero((~rows.is_test & rows.time_step.le(cut)).to_numpy())
        vi = np.flatnonzero((~rows.is_test & rows.time_step.between(cut+1, end)).to_numpy())
        v = rows.iloc[vi]
        # Reuse saved control predictions if available, otherwise refit them below.
        controls = existing_predictions(args.old_root, cut, end, v.txId)
        if controls is None:
            components = {}
            for name in ['reference', 'annotation', 'graph']:
                predictions = [fit_predict(name, full, rows, rows[FEATURES], fi, vi,
                                          cut, seed, folder, graph) for seed in args.seeds]
                components[name] = np.mean(predictions, axis=0)
            unknown = logit_aux(components['reference'], components['annotation'])
            # Rebuild the ensemble up to stage 3: reference -> +annotation (log-odds 80/20) -> +graph (80/20).
            controls = dict(reference=components['reference'], unknown_blend=unknown,
                            graph=components['graph'], incumbent=.8*unknown+.2*components['graph'])
            # Refit controls must reproduce all overlapping previously saved rows.
            prior_end = {25: 30, 28: 32, 30: 35, 32: 35}.get(cut)
            if prior_end is not None:
                short = v.time_step.le(prior_end).to_numpy()
                old = existing_predictions(args.old_root, cut, prior_end, v.txId[short])
                if old is not None:
                    for name in controls:
                        np.testing.assert_allclose(controls[name][short], old[name], rtol=0, atol=3e-14)
                    emit(event='overlap_reproduction_passed', cutoff=cut, rows=int(short.sum()))
        d = v[['txId', 'time_step', 'label']].reset_index(drop=True)
        for name, p in controls.items():
            d[name] = p
        if not args.controls_only:
            for name, x in [('duplicate_raw', xd), ('period_rank', xr)]:
                ps = [fit_predict(name, full, rows, x, fi, vi, cut, seed, folder) for seed in args.seeds]
                d[name] = np.mean(ps, axis=0)
            d['duplicate_blend50'] = .5*d.incumbent + .5*d.duplicate_raw
            d['period_rank_blend50'] = .5*d.incumbent + .5*d.period_rank
        d.to_csv(folder / 'predictions.csv', index=False)
        windows = [end] + ([30] if cut == 25 and end > 30 else [])
        for window_end in windows:
            part = d[d.time_step.le(window_end)]
            result = score(part, cut, window_end)
            records.extend(result)
            emit(event='fold_complete', cutoff=cut, end=window_end,
                 scores={r['name']: r['auc'] for r in result})
            for period, group in part.groupby('time_step'):
                periods.extend(score(group, cut, window_end, group=f'period_{period}'))
            for low, high in [(1, 3), (4, 7), (8, 14)]:
                group = part[part.time_step.between(cut+low, cut+high)]
                horizons.extend(score(group, cut, window_end, group=f'horizon_{low}_{high}'))
        pd.DataFrame(records).to_csv(args.out / 'results.csv', index=False)
        pd.DataFrame(periods).to_csv(args.out / 'period_results.csv', index=False)
        pd.DataFrame(horizons).to_csv(args.out / 'horizon_results.csv', index=False)
    results = pd.DataFrame(records)
    primary = results[((results.cutoff == 25) & (results.end == 30))
                      | ((results.cutoff == 28) & (results.end == 32))
                      | ((results.cutoff == 30) & (results.end == 35))]
    summary = primary.groupby('name').agg(mean_auc=('auc','mean'),
                  mean_delta=('delta_vs_incumbent','mean'),
                  worst_delta=('delta_vs_incumbent','min'),
                  wins=('delta_vs_incumbent', lambda x: int((x>0).sum())))
    summary.to_csv(args.out / 'primary_summary.csv')
    emit(event='screen_complete', summary=summary.reset_index().to_dict(orient='records'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--graph-cache', type=Path, required=True)
    parser.add_argument('--old-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cutoffs', default='25:35,28:32,30:35,21:35')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--controls-only', action='store_true', help='Rebuild reference/current-best controls without fitting new candidates.')
    args = parser.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(',')]
    evaluate(args)

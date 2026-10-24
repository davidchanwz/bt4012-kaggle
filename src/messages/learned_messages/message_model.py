"""Stage 4 of ensemble B: a small directed message-passing network (PyTorch).

Model (MessageNet):
- Each transaction's 165 features are quantile-normalised (fitted on labelled
  training rows up to the cutoff only) and encoded into a 64-dim vector.
  Transactions without features (unknown training rows, and the hidden test-period
  nodes) get a learned "missing" vector instead, matching what is available at test time.
- Two residual layers mix each node with the average of its incoming and of its
  outgoing neighbours (separate weights for each direction, as in GraphSAGE).
- A small MLP head predicts P(illicit) from [original encoding, final encoding].

Training (`train`):
- Optional pretraining (15 epochs, used in the final model): hide 20% of visible
  nodes and learn to reconstruct their features from the graph (GraphMAE-style).
  These targets are features, not fraud labels.
- Supervised training on labelled rows with binary cross-entropy, each time step
  weighted by exp((t - cutoff)/5) as in the tree models.
- The number of epochs is chosen on an inner window (train <= cutoff-3, validate
  the next 3 steps) and the model is then refit up to the cutoff.

Running this file directly evaluates the variants 'self_only' (no edges),
'messages' and 'pretrained' on the temporal folds. finalize_probe.py fits the
final model.
"""
from pathlib import Path
import argparse
import copy
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
from scipy.special import expit, logit
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(4)
# Hyperparameters (fixed before the final fit).
CFG = dict(width=64, dropout=.1, lr=.001, weight_decay=.0001,
           max_epochs=35, min_epochs=5, patience=6, pretrain_epochs=15, tau=5.)


def emit(**r):
    print(json.dumps(r), flush=True)


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + '\n')


def torch_matrix(a):
    a = a.tocoo()
    return torch.sparse_coo_tensor(torch.from_numpy(np.stack([a.row, a.col]).astype(np.int64)),
                                  torch.from_numpy(a.data.astype(np.float32)), a.shape,
                                  check_invariants=True).coalesce()


# Row-normalise the adjacency matrix so a product gives the mean over neighbours.
def normalized(a):
    degree = np.asarray(a.sum(1)).ravel()
    return sparse.diags(1 / np.maximum(degree, 1)) @ a


# One residual message-passing layer: own transform + incoming mean + outgoing mean.
class Layer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.own = nn.Linear(width, width)
        self.incoming = nn.Linear(width, width, bias=False)
        self.outgoing = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.drop = nn.Dropout(CFG['dropout'])

    def forward(self, h, incoming, outgoing, edges):
        z = self.own(h)
        if edges:
            z = z + self.incoming(torch.sparse.mm(incoming, h))
            z = z + self.outgoing(torch.sparse.mm(outgoing, h))
        return h + self.drop(F.relu(self.norm(z)))


# edges=False gives a control model that ignores the graph ('self_only').
class MessageNet(nn.Module):
    def __init__(self, edges=True):
        super().__init__()
        width = CFG['width']
        self.edges = edges
        self.encoder = nn.Sequential(nn.Linear(165, width), nn.ReLU(), nn.Linear(width, width))
        # Learned vector used for nodes whose features are hidden.
        self.missing = nn.Parameter(torch.zeros(width))
        # 5 metadata inputs: log in-degree, log out-degree, visible flag,
        # number of visible incoming / outgoing neighbours.
        self.metadata = nn.Linear(5, width)
        self.layers = nn.ModuleList([Layer(width), Layer(width)])
        self.predictor = nn.Sequential(nn.LayerNorm(2*width), nn.Linear(2*width, width),
                                       nn.ReLU(), nn.Dropout(CFG['dropout']), nn.Linear(width, 1))
        # Only used during pretraining, to reconstruct masked features.
        self.decoder = nn.Linear(width, 165)

    def encode(self, g, visible=None):
        visible = g['visible'] if visible is None else visible
        # Mask before encoding as well as replacing the encoded hidden state.
        selected = visible.bool()
        h = self.missing[None, :].expand(len(visible), -1).clone()
        h[selected] = self.encoder(g['x'][selected])
        metadata = g['metadata'].clone()
        metadata[:, 2] = visible
        metadata[:, 3] = torch.sparse.mm(g['incoming'], visible[:, None]).squeeze(1)
        metadata[:, 4] = torch.sparse.mm(g['outgoing'], visible[:, None]).squeeze(1)
        h = h + self.metadata(metadata)
        original = h
        for layer in self.layers:
            h = layer(h, g['incoming'], g['outgoing'], self.edges)
        return original, h

    def forward(self, g):
        original, h = self.encode(g)
        return self.predictor(torch.cat([original, h], 1)).squeeze(-1)


# Load the node cache and attach labels for the modelled rows (labelled train + test).
def read_source(cache, graph_cache):
    nodes = pd.read_csv(cache / 'nodes.csv')
    rows = pd.read_csv(graph_cache / 'rows.csv')
    index = pd.Index(nodes.txId)
    positions = index.get_indexer(rows.txId)
    assert (positions >= 0).all()
    nodes['label'] = np.nan
    nodes.loc[positions, 'label'] = rows.label.to_numpy()
    nodes['query'] = False
    nodes.loc[positions, 'query'] = True
    a = sparse.load_npz(cache / 'adjacency.npz').tocsr()
    src, dst = a.nonzero()
    np.testing.assert_array_equal(nodes.time_step.to_numpy()[src], nodes.time_step.to_numpy()[dst])
    return nodes, np.load(cache / 'raw_vectors.npy', mmap_mode='r'), a


# Build one graph per time step up to `end`. Normalisation is fitted only on
# labelled rows with time_step <= cut, so no future information leaks in.
def prepare(nodes, raw, a, cut, end, include_targets=False, transformer=None):
    fit = np.flatnonzero((nodes.label.notna() & nodes.time_step.le(cut)).to_numpy())
    q = transformer
    if q is None:
        q = QuantileTransformer(n_quantiles=min(1000, len(fit)), output_distribution='normal',
                                subsample=None, random_state=4012)
        q.fit(np.asarray(raw[fit]))
    groups = []
    for period in range(1, end+1):
        ix = np.flatnonzero(nodes.time_step.eq(period).to_numpy())
        if not len(ix):
            continue
        n = nodes.iloc[ix]
        visible = n.visible.to_numpy().astype(np.float32)
        x = np.zeros((len(ix), 165), np.float32)
        present = visible > 0
        # Hidden nodes keep a zero vector here; the model swaps in the learned missing vector.
        x[present] = np.nan_to_num(np.clip(q.transform(np.asarray(raw[ix[present]])), -5, 5)).astype(np.float32)
        sub = a[ix][:, ix]
        outgoing = torch_matrix(normalized(sub))
        incoming = torch_matrix(normalized(sub.T))
        di = np.asarray(sub.sum(0)).ravel()
        do = np.asarray(sub.sum(1)).ravel()
        metadata = np.stack([np.log1p(di), np.log1p(do), visible, np.zeros(len(ix)), np.zeros(len(ix))], 1).astype(np.float32)
        label = n.label.to_numpy(np.float32)
        query = n['query'].to_numpy(copy=True)
        known = np.isfinite(label)
        g = dict(period=period, txId=n.txId.to_numpy(), x=torch.from_numpy(x),
                 visible=torch.from_numpy(visible), metadata=torch.from_numpy(metadata),
                 incoming=incoming, outgoing=outgoing,
                 known=torch.from_numpy(known), query=torch.from_numpy(query),
                 y=torch.from_numpy(np.nan_to_num(label)), n_known=int(known.sum()))
        # Reconstruction targets (features, not labels) only for periods up to the cutoff.
        if include_targets and period <= cut:
            target = np.zeros((len(ix), 165), np.float32)
            supplied = n.supplied.to_numpy()
            target[supplied] = np.nan_to_num(np.clip(q.transform(np.asarray(raw[ix[supplied]])), -5, 5)).astype(np.float32)
            g['target'] = torch.from_numpy(target)
            g['supplied'] = torch.from_numpy(supplied)
        groups.append(g)
    return groups, q


@torch.no_grad()
# Predict P(illicit) for the modelled rows in time steps start..end.
def predict(model, groups, start, end):
    model.eval()
    records = []
    for g in groups:
        if start <= g['period'] <= end:
            p = torch.sigmoid(model(g)).numpy()
            selected = g['query'].numpy()
            records.append(pd.DataFrame(dict(txId=g['txId'][selected],
                time_step=g['period'], label=g['y'].numpy()[selected], p=p[selected])))
    return pd.concat(records, ignore_index=True)


def train(groups, cut, seed, edges, pretrained=False, epochs=None, tag=''):
    """Train a MessageNet on all time steps <= cut and return (model, training info).

    If epochs is None, the epoch count is selected on steps cut+1..cut+3 (inner
    validation); otherwise the model is trained for exactly `epochs` epochs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = MessageNet(edges)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    fit = [g for g in groups if g['period'] <= cut and g['n_known']]
    start = time.time()
    if pretrained:
        for epoch in range(1, CFG['pretrain_epochs']+1):
            losses = []
            model.train()
            for index in rng.permutation(len(fit)):
                g = fit[index]
                visible = g['visible'].clone()
                # Pretraining: randomly hide 20% of nodes and reconstruct their features.
                visible[torch.rand(len(visible)) < .2] = 0.
                selected = g['supplied'] & (visible == 0)
                if not selected.any():
                    continue
                _, h = model.encode(g, visible)
                reconstruction = model.decoder(h[selected])
                # Normalized features are clipped; MSE is a pilot objective,
                # not the scaled-cosine objective of the GraphMAE paper.
                loss = F.mse_loss(reconstruction, g['target'][selected])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(loss.item())
            if epoch == 1 or epoch % 5 == 0:
                emit(event='pretrain', tag=tag, seed=seed, epoch=epoch,
                     loss=float(np.mean(losses)), seconds=round(time.time()-start, 2))
        # Reset optimizer so the supervised-stage policy matches the control.
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    total_weight = sum(g['n_known'] * np.exp((g['period']-cut)/5) for g in fit)
    history, best, best_epoch, best_state = [], -np.inf, 0, None
    limit = CFG['max_epochs'] if epochs is None else epochs
    for epoch in range(1, limit+1):
        model.train()
        losses = []
        for index in rng.permutation(len(fit)):
            g = fit[index]
            logits = model(g)[g['known']]
            targets = g['y'][g['known']]
            # Recency weight for this time step (same idea as the tree models).
            weight = np.exp((g['period']-cut)/5) * len(fit) / total_weight
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='sum') * weight
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(loss.item())
        record = dict(epoch=epoch, loss=float(np.mean(losses)), seconds=round(time.time()-start, 2))
        if epochs is None:
            # Inner validation on the 3 steps after the (inner) cutoff to pick the epoch count.
            d = predict(model, groups, cut+1, cut+3)
            auc = roc_auc_score(d.label, d.p)
            record['inner_auc'] = auc
            if epoch >= CFG['min_epochs'] and auc > best+1e-7:
                best, best_epoch, best_state = auc, epoch, copy.deepcopy(model.state_dict())
        history.append(record)
        if epoch == 1 or epoch % 5 == 0:
            emit(event='epoch', tag=tag, seed=seed, edges=edges, pretrained=pretrained, **record)
        if epochs is None and best_epoch and epoch-best_epoch >= CFG['patience']:
            break
    if epochs is None:
        model.load_state_dict(best_state)
    return model, dict(best_epoch=best_epoch if epochs is None else epochs,
                       history=history, cfg=CFG, seconds=time.time()-start)


# Validation: compare variants on temporal folds against the previous ensemble.
def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out/'config.json', dict(args=vars(args), cfg=CFG))
    nodes, raw, a = read_source(args.node_cache, args.graph_cache)
    summaries = []
    for cut, end in [tuple(map(int,p.split(':'))) for p in args.folds.split(',')]:
        folder = args.out/f'fold_{cut}_{end}'
        folder.mkdir(exist_ok=True)
        groups, transformer = prepare(nodes, raw, a, cut, end, include_targets='pretrained' in args.models)
        inner = None
        controls = pd.read_csv(args.controls/f'fold_{cut}_{end}'/'predictions.csv').set_index('txId')
        output = controls[['time_step','label','incumbent']].copy()
        for name in args.models:
            edges = name != 'self_only'
            pretrain = name == 'pretrained'
            schedule = folder/f'{name}_schedule.json'
            if schedule.exists():
                chosen = json.loads(schedule.read_text())
            else:
                if inner is None:
                    inner, _ = prepare(nodes, raw, a, cut-3, cut, include_targets='pretrained' in args.models)
                # Epoch count chosen on the inner window before refitting on the full fold.
                _, chosen = train(inner, cut-3, 4012, edges, pretrain, tag=f'inner_{cut}_{name}')
                save_json(schedule, chosen)
            ps = []
            for seed in args.seeds:
                path = folder/f'{name}_seed_{seed}.csv'
                if path.exists():
                    predictions = pd.read_csv(path)
                else:
                    model, history = train(groups, cut, seed, edges, pretrain,
                                            epochs=chosen['best_epoch'], tag=f'outer_{cut}_{name}')
                    predictions = predict(model, groups, cut+1, end)
                    predictions.to_csv(path, index=False)
                    torch.save(model.state_dict(), folder/f'{name}_seed_{seed}.pt')
                    save_json(folder/f'{name}_seed_{seed}_history.json', history)
                predictions = predictions.set_index('txId').loc[output.index]
                np.testing.assert_array_equal(predictions.label, output.label)
                ps.append(predictions.p.to_numpy())
                emit(event='neural_seed_complete', cutoff=cut, name=name, seed=seed,
                     auc=roc_auc_score(output.label, predictions.p), epochs=chosen['best_epoch'])
            p = np.mean(ps, 0)
            output[name] = p
            output[name+'_logit20'] = expit(.8*logit(np.clip(output.incumbent,1e-7,1-1e-7))
                                           +.2*logit(np.clip(p,1e-7,1-1e-7)))
            output.to_csv(folder/'predictions.csv')
            joblib.dump(transformer, folder/'transformer.joblib')
            windows = [end]+([30] if cut==25 and end>30 else [])
            for window in windows:
                d = output[output.time_step.le(window)]
                for column in [name, name+'_logit20']:
                    auc = roc_auc_score(d.label,d[column])
                    record = dict(cutoff=cut,end=window,name=column,auc=auc,
                                  delta=auc-roc_auc_score(d.label,d.incumbent))
                    summaries.append(record)
                    emit(event='neural_result', **record)
            pd.DataFrame(summaries).to_csv(args.out/'results.csv',index=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--node-cache',type=Path,required=True)
    parser.add_argument('--graph-cache',type=Path,required=True)
    parser.add_argument('--controls',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--folds',default='25:35,28:32,30:35')
    parser.add_argument('--models',default='self_only,messages,pretrained')
    parser.add_argument('--seeds',default='0,1,2')
    args=parser.parse_args()
    args.models=args.models.split(',')
    args.seeds=[int(s) for s in args.seeds.split(',')]
    run(args)

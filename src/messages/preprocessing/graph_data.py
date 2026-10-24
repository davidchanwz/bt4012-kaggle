"""Build the graph feature matrix ("panel") used by the graph LightGBM.

Graph: one node per transaction, one directed edge A->B when B spends an output
of A (txs_edgelist.csv). The graph splits into 49 components, one per time step.

Key issue - feature availability. test.csv only contains labelled transactions:
the unknown transactions of steps 36-49 appear in the edge list but have no
features. In training, unknown transactions DO have features. If we used them,
training rows would see richer neighbourhoods than test rows. So the 'matched'
policy (the one used in the final model) hides the features of unknown training
rows while keeping their nodes and edges, so that training looks like test.

Output columns per transaction (1,187 in total):
- the 165 original features,
- for each of the 6 relations: the mean of each feature over neighbours whose
  features are available (6 x 165 = 990; NaN if no neighbour is available),
- for each relation: total neighbours, available, used, coverage (24),
- 8 topology features (degrees, depth in the flow, chain lengths, position).
No labels are used anywhere in these features.

Panels are built for three policies: 'visible' (use all supplied features),
'matched' (final), and 'reconstructed' (fill hidden nodes from neighbours;
an experiment). They are cached as .npy files.
"""
from pathlib import Path
from collections import deque
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from common import FEATS,RELATIONS,save_json,emit


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


# Structural features from the edge list only (no features, no labels).
def topology(a):
    at=a.T.tocsr();ind=np.diff(at.indptr);out=np.diff(a.indptr)
    remaining=ind.copy();queue=deque(np.flatnonzero(remaining==0));order=[]
    while queue:
        u=queue.popleft();order.append(u)
        for v in a.indices[a.indptr[u]:a.indptr[u+1]]:
            remaining[v]-=1
            if remaining[v]==0:queue.append(v)
    # Topological order exists because money can only flow forward in time.
    assert len(order)==a.shape[0],'Expected acyclic transaction graph'
    up=np.zeros(a.shape[0],np.float32);down=up.copy()
    before=up.copy();after=up.copy()
    for u in order:
        for v in a.indices[a.indptr[u]:a.indptr[u+1]]:
            # up_depth = longest chain of payments leading into this transaction.
            up[v]=max(up[v],up[u]+1)
            # chain_before = length of a simple one-in/one-out chain ending here (peeling chains).
            if out[u]==1 and ind[v]==1:before[v]=before[u]+1
    for u in reversed(order):
        for v in a.indices[a.indptr[u]:a.indptr[u+1]]:
            down[u]=max(down[u],down[v]+1)
            if out[u]==1 and ind[v]==1:after[u]=after[v]+1
    values=np.stack([ind,out,up,down,before,after,up/(up+down+1),
                     (ind-out)/(ind+out+1)],axis=1).astype(np.float32)
    names=['in_degree','out_degree','up_depth','down_depth','chain_before','chain_after',
           'relative_position','fan_balance']
    return values,['graph_'+s for s in names]


def relation_summaries(a,x,used,real,rows):
    """Means and coverage for six path types. No labels accepted by this API."""
    at=a.T.tocsr()
    source=np.array(x,dtype=np.float32,copy=True)
    # Neighbours whose features are hidden contribute nothing to the sums.
    source[~used]=0
    used=used.astype(np.float32);real=real.astype(np.float32)
    one=np.ones(a.shape[0],np.float32)
    outdeg=np.asarray(a@one).ravel();indeg=np.asarray(at@one).ravel()
    # Sparse matrix products: sum of features over incoming / outgoing neighbours.
    sin=np.asarray(at@source);sout=np.asarray(a@source)
    cin=np.asarray(at@used).ravel();cout=np.asarray(a@used).ravel()
    rin=np.asarray(at@real).ravel();rout=np.asarray(a@real).ravel()
    means=[];statistics=[]
    for relation in RELATIONS:
        if relation=='in':s,c,r,total=sin,cin,rin,indeg
        elif relation=='out':s,c,r,total=sout,cout,rout,outdeg
        elif relation=='in2':s,c,r,total=at@sin,at@cin,at@rin,at@indeg
        elif relation=='out2':s,c,r,total=a@sout,a@cout,a@rout,a@outdeg
        # Co-spenders: paths A->B<-C, excluding the transaction itself.
        elif relation=='shared_successor':
            s=a@sin-outdeg[:,None]*source
            c=a@cin-outdeg*used;r=a@rin-outdeg*real;total=a@indeg-outdeg
        else:
            s=at@sout-indeg[:,None]*source
            c=at@cout-indeg*used;r=at@rout-indeg*real;total=at@outdeg-indeg
        c=np.maximum(np.asarray(c).ravel(),0)
        r=np.maximum(np.asarray(r).ravel(),0)
        total=np.maximum(np.asarray(total).ravel(),0)
        den=c[rows]
        m=np.full((len(rows),x.shape[1]),np.nan,np.float32)
        # Mean over available neighbours; stays NaN when none are available
        # (LightGBM handles missing values natively).
        np.divide(np.asarray(s)[rows],den[:,None],out=m,where=den[:,None]>0)
        means.append(m)
        coverage=np.divide(r[rows],total[rows],out=np.zeros(len(rows),np.float32),where=total[rows]>0)
        statistics.append(np.stack([total[rows],r[rows],c[rows],coverage],axis=1))
    return means,np.concatenate(statistics,axis=1).astype(np.float32)


# Experiment only: fill hidden nodes with the mean of their neighbours.
def reconstruct(a,x,visible,passes=2):
    b=(a+a.T).tocsr();b.data[:]=1
    z=np.array(x,dtype=np.float32,copy=True);z[~visible]=0
    used=visible.copy();records=[]
    for step in range(passes):
        count=np.asarray(b@used.astype(np.float32)).ravel()
        sums=b@z
        fill=(~visible)&(count>0)
        next_z=z.copy();next_z[fill]=sums[fill]/count[fill,None]
        z=next_z;used=visible|fill
        records.append(dict(pass_number=step+1,n_reconstructed=int((used&~visible).sum())))
    np.testing.assert_array_equal(z[visible],x[visible])
    return z,used,records


def build(data,cache):
    """Build (or reuse) the cached graph panels for all three visibility policies."""
    cache.mkdir(parents=True,exist_ok=True)
    inputs={n:sha(data/n) for n in ['train.csv','test.csv','txs_edgelist.csv']}
    complete=cache/'manifest.json'
    if complete.exists():
        old=json.loads(complete.read_text())
        assert old['input_sha256']==inputs,'Inputs changed: use a new graph cache'
        assert old.get('version')==1
        emit(dict(event='reuse_graph_cache',cache=str(cache)));return old
    start=time.time()
    train=pd.read_csv(data/'train.csv').astype({c:'float32' for c in FEATS})
    test=pd.read_csv(data/'test.csv').astype({c:'float32' for c in FEATS})
    train['is_test']=False;test['is_test']=True
    full=pd.concat([train,test],ignore_index=True)
    e=pd.read_csv(data/'txs_edgelist.csv')
    ids=pd.Index(np.unique(np.r_[full.txId.to_numpy(),e.to_numpy().ravel()]))
    a=sparse.csr_matrix((np.ones(len(e),np.float32),
        (ids.get_indexer(e.txId1),ids.get_indexer(e.txId2))),shape=(len(ids),len(ids)))
    a.sum_duplicates();a.data[:]=1
    assert np.all(a.diagonal()==0)
    row=ids.get_indexer(full.txId)
    n_components,components=connected_components(a,directed=False)
    periods=pd.DataFrame({'component':components[row],'period':full.time_step})
    # Sanity checks: every connected component lies inside one time step,
    # and there are exactly 49 components (no edges between time steps).
    assert periods.groupby('component').period.nunique().max()==1
    period_map=periods.groupby('component').period.first()
    assert len(period_map)==n_components==49
    node_period=period_map.reindex(components).to_numpy()
    src,dst=a.nonzero();assert np.array_equal(node_period[src],node_period[dst])
    x=np.zeros((len(ids),165),np.float32);x[row]=full[FEATS].to_numpy(np.float32)
    supplied=np.zeros(len(ids),bool);supplied[row]=True
    # 'visible' = rows that would also have features at test time:
    # labelled training rows and test rows. Unknown training rows are hidden.
    visible=np.zeros(len(ids),bool)
    target=(full.label.notna()|full.is_test).to_numpy()
    visible[row[target]]=True
    # Output rows = labelled training rows + test rows (the rows we model).
    rows=row[target]
    metadata=full.loc[target,['txId','time_step','label','is_test','index']].reset_index(drop=True)
    metadata.to_csv(cache/'rows.csv',index=False)
    structures,struct_names=topology(a)
    columns=FEATS+[f'graph_{r}_mean_{j}' for r in RELATIONS for j in range(1,166)]
    columns += [f'graph_{r}_{s}' for r in RELATIONS for s in ['total','available','used','coverage']]
    columns += struct_names
    audit=dict(nodes=len(ids),edges=a.nnz,components=n_components,
               every_component_one_period=True,training_unknown_vectors=int((supplied&~visible).sum()))
    reconstruction=[]
    for policy in ['visible','matched','reconstructed']:
        emit(dict(event='building_panel',policy=policy))
        if policy=='visible':z,used,real=x,supplied,supplied
        # Final policy: only 'visible' neighbours contribute to the means.
        elif policy=='matched':z,used,real=x,visible,visible
        else:
            z,used,reconstruction=reconstruct(a,x,visible)
            real=visible
        means,stats=relation_summaries(a,z,used,real,rows)
        panel=np.concatenate([x[rows],*means,stats,structures[rows]],axis=1).astype(np.float32)
        assert panel.shape==(len(rows),len(columns)) and not np.isinf(panel).any()
        np.testing.assert_array_equal(panel[:,:165],full.loc[target,FEATS].to_numpy(np.float32))
        np.save(cache/f'{policy}.npy',panel)
        for group,selector in [('training',~metadata.is_test.to_numpy()),('test',metadata.is_test.to_numpy())]:
            for rel in ['in','out']:
                j=RELATIONS.index(rel)*4
                total=stats[selector,j];available=stats[selector,j+1]
                has=total>0
                audit[f'{policy}_{group}_{rel}']=dict(
                    fraction_with_neighbours=float(has.mean()),
                    available_edge_fraction=float(available.sum()/max(total.sum(),1)),
                    all_hidden_given_neighbours=float((available[has]==0).mean()))
        emit(dict(event='panel_saved',policy=policy,shape=panel.shape,seconds=time.time()-start))
        del means,stats,panel
    manifest=dict(version=1,input_sha256=inputs,columns=columns,relations=RELATIONS,
        n_rows=len(rows),n_training=int((~metadata.is_test).sum()),n_test=int(metadata.is_test.sum()),
        audit=audit,reconstruction=reconstruction,seconds=time.time()-start)
    save_json(complete,manifest)
    return manifest


def load(cache,policy):
    manifest=json.loads((Path(cache)/'manifest.json').read_text())
    rows=pd.read_csv(Path(cache)/'rows.csv')
    panel=np.load(Path(cache)/f'{policy}.npy',mmap_mode='r')
    assert len(rows)==len(panel)==manifest['n_rows']
    return rows,panel,manifest


def selected_columns(kind,manifest):
    """Column subsets: 'coverage', 'direct' (in/out means only) or 'relations' (all 1,187; used in the final model)."""
    names=manifest['columns']
    raw=list(range(165))
    stats=[i for i,n in enumerate(names) if n.startswith('graph_') and '_mean_' not in n]
    direct=[i for i,n in enumerate(names) if n.startswith('graph_in_mean_') or n.startswith('graph_out_mean_')]
    if kind=='coverage':return raw+stats
    if kind=='direct':return raw+direct+stats
    if kind=='relations':return list(range(len(names)))
    raise ValueError(kind)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);args=p.parse_args()
    m=build(args.data,args.cache)
    emit(dict(event='graph_audit',audit=m['audit'],reconstruction=m['reconstruction']))

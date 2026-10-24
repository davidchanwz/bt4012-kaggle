"""Build the two caches needed by the graph and message-passing stages.

1. graph cache (preprocessing/graph_data.build): the 1,187-column graph panels.
2. node cache  (preprocessing/learned_imputation.prepare): one row per graph node
   (including the 46,668 nodes that only appear in the edge list), with the raw
   feature vectors, a 'visible' flag, the adjacency matrix and neighbour context.
Neither cache uses fraud labels. Run once; later stages reuse the caches.

Note: preprocessing/ contains copies of src/graph/{common,graph_data,graph_lgbm}.py
so that this folder is self-contained; the files are identical.
"""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'preprocessing'))
from graph_data import build
from learned_imputation import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['data','graph-cache','node-cache']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    build(a.data,a.graph_cache)
    prepare(a.data,a.node_cache,a.graph_cache)

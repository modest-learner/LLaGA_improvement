"""Small CPU checks for information interventions and LP edge isolation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, add_self_loops, degree
from eval.eval_pretrain import MP
from eval.lp_inputs import pair_ho, donors_for_pair, intervene, parse_binary
from research.prepare_clean_lp import clean_splits


def main():
    edge=torch.tensor([[0,1,1,2,2,3,3,4,4,0],[1,0,2,1,3,2,4,3,0,4]])
    x=torch.arange(20,dtype=torch.float32).reshape(5,4)
    centers=[0,1]
    subset,e,m,_=k_hop_subgraph(centers,2,edge,relabel_nodes=True)
    kill=((e[0]==m[0])&(e[1]==m[1]))|((e[0]==m[1])&(e[1]==m[0]))
    e=e[:,~kill]
    local=x[subset]
    e,_=remove_self_loops(e)
    e,_=add_self_loops(e)
    inv=degree(e[1],len(subset),dtype=x.dtype).pow(-.5)
    norm=inv[e[0]]*inv[e[1]]
    expected=[x[centers]]
    mp=MP()
    for _ in range(2):
        local=mp.propagate(e,x=local,norm=norm)
        expected.append(local[m])
    expected=torch.stack(expected,dim=1)
    assert torch.equal(expected,pair_ho(MP(),edge,x,centers,centers,2,'cpu'))
    donor=pair_ho(MP(),edge,x,[2,3],centers,2,'cpu')
    assert torch.equal(intervene(expected,'hop_mismatch',donor)[:,0],expected[:,0])
    assert torch.equal(intervene(expected,'hop_mismatch',donor)[:,1:],donor[:,1:])
    assert intervene(expected,'center_only').shape==(2,1,4)
    repeated=intervene(expected,'center_repeat')
    assert torch.equal(repeated[:,0],repeated[:,2])
    assert donors_for_pair([0,1],5,42)==donors_for_pair([0,1],5,42)
    assert not set(donors_for_pair([0,1],5,42))&{0,1}
    assert parse_binary('Yes, they connect.')=='yes'
    assert parse_binary('There is no connection.')=='no'
    assert parse_binary('yes or no') is None and parse_binary('unrelated') is None
    def row(a,b,label):
        return {'id':[a,b],'conversations':[{}, {'value':label}]}
    train=[row(0,1,'yes'),row(1,0,'yes'),row(1,2,'yes'),row(2,3,'yes'),
           row(0,2,'no'),row(0,3,'no'),row(1,3,'no'),row(3,3,'no'),row(3,4,'no')]
    test=[row(3,4,'yes'),row(1,4,'no')]
    splits,context,stats=clean_splits(Data(edge_index=edge,num_nodes=5),train,test,42)
    pairs=[p for rows in splits.values() for p,y in rows]
    assert len(pairs)==len(set(pairs))
    assert all(p not in context for p in pairs)
    assert stats['train']['duplicate_or_cross_split']==1
    assert stats['train']['self_pair']==1
    assert stats['train']['inconsistent_with_known_graph']==1
    print('PASS: original HO equality, center/mismatch interventions, deterministic donors, parser, split separation and hidden supervision.')


if __name__=='__main__':
    main()

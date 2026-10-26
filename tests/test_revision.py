import numpy as np
from Algs.initialization import ward_partition
from Algs.IKM import IKM
from Algs.KM import KM
from Algs.weights import A0
from experiments.revision_run import partition_labels, select_beta, summarize
import pytest


def test_ward_merge_cost_matches_direct_greedy():
    x = np.array([[0.,0.], [0.1,0.2], [2.,3.], [4.,8.], [8.,9.], [8.2,9.1]])
    for k in range(1, len(x)+1):
        c = [{i} for i in range(len(x))]
        while len(c)>k:
            costs = [(len(a)*len(b)/(len(a)+len(b))*np.sum((x[list(a)].mean(0)-x[list(b)].mean(0))**2), i,j)
                     for i,a in enumerate(c) for j,b in enumerate(c) if i<j]
            _,i,j = min(costs)
            c[i] |= c[j]; c.pop(j)
        assert {frozenset(t) for t in ward_partition(x,k)} == {frozenset(t) for t in c}


def test_ward_duplicates_still_exact_k():
    c = ward_partition(np.zeros((6,2)),4)
    assert len(c)==4 and all(c)
    partition_labels(c,6)


@pytest.mark.parametrize('cls',[IKM,KM])
@pytest.mark.parametrize('init',['random','kmeans++','ward'])
def test_optimizer_full_partition_and_repeatability(cls,init):
    x = np.array([[0.,.1],[.1,.2],[.2,.3],[5.,5.1],[5.1,5.2],[5.2,5.3]])
    outputs=[]
    for _ in range(2):
        np.random.seed(42)
        m=cls(A0,'2d',init=init)
        outputs.append(partition_labels(m.fit(x.copy(),[np.arange(2)],2),len(x)))
        assert m.converged_
    assert np.array_equal(*outputs)


def test_selector_does_not_use_full_label_metrics():
    y=np.tile(np.arange(3),10)
    rows=[dict(beta=1.,labels=(np.arange(30)%2).tolist(),ari=100),dict(beta=2.,labels=y.tolist(),ari=-100)]
    result=select_beta(rows,y)
    assert result['beta']==2.
    assert all(len(r['sample'])==6 for r in result['draws'])
    assert len(result['draws'])==50


def test_hits_use_full_precision_not_rounded_table_cells():
    s=summarize([dict(ari=v,nmi=v,converged=True) for v in [.801,.804,.804]])
    assert s['ari']['hits']==2


def test_partition_rejects_leftovers():
    with pytest.raises(ValueError): partition_labels([{0},{2}],3)

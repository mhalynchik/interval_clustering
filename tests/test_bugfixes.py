import numpy as np
import pytest
from Algs.BANCOIn import BANCOIn
from Algs.IKM import IKM
from Algs.KM import KM
from Algs.weights import A0,A1,A2


def test_banco_finishes_unweighted_local_search():
    x=np.random.default_rng(5).normal(size=(12,2));x-=x.mean(0)
    b=BANCOIn(A0,'3d');b.V=2;b.beta_degree=1.;b.idxs=[np.arange(2)]
    clusters,c=b.EXTANW(x,list(range(len(x))),np.empty((0,2)),[],1e-10)
    s=clusters[-1];value=len(s)*np.sum(c[-1]**2)
    for i in range(len(x)):
        t=s-{i} if i in s else s|{i}
        if t: assert len(t)*np.sum(x[list(t)].mean(0)**2)<=value+1e-8


def test_ikm_empty_distance_not_zero():
    m=IKM(A0,'2d');m.weights=np.ones(1);m.beta_degree=1
    assert m.obj_to_cl_d(np.array([10.]),np.array([-10.]),0,0)==400


def test_km_repairs_using_object_distances():
    m=KM(A0,'2d');m.K=3;m.weights=np.ones(1);m.beta_degree=1
    x=np.array([[0.],[1.],[10.],[11.],[20.],[21.]])
    labels,_=m.clusterupdate(x,np.zeros((3,1)))
    assert labels[5]!=0 and labels[4]!=0 and len(set(labels))==3


def test_km_centers_keep_cluster_ids():
    m=KM(A0,'2d');m.beta_degree=1;m.idxs=[np.arange(1)];m.K=3
    c=m.ceupdate(np.array([[0.],[10.]]),np.array([0,2]))
    assert c[2,0]==10 and c[0,0]==0


def test_a1_beta_one_simplex_minimum():
    x=np.array([[0.,0.],[2.,6.],[10.,10.],[12.,16.]])
    w=A1(x,[{0,1},{2,3}],b=1.,idxs=[np.arange(2)],d='2d')
    np.testing.assert_array_equal(w,[1.,0.])


def test_a2_beta_two_product_minimum():
    x=np.array([[0.,0.],[2.,6.],[10.,10.],[12.,16.]])
    w=A2(x,[{0,1},{2,3}],b=2.,idxs=[np.array([0]),np.array([1])],d='2d')
    D=np.array([30.,70.]);expected=np.prod(D)**.25/np.sqrt(D)
    np.testing.assert_allclose(w,expected,rtol=1e-12)
    assert np.prod(w)==pytest.approx(1.)


@pytest.mark.parametrize('b',[1.,1.1,2.,5.])
def test_weights_finite_including_constant_data(b):
    for x in [np.ones((4,2)),np.array([[0.,0.],[2.,6.],[10.,10.],[12.,16.]])]:
        for f,idxs in [(A1,[np.arange(2)]),(A2,[np.array([0]),np.array([1])])]:
            assert np.isfinite(f(x,[{0,1},{2,3}],b=b,idxs=idxs,d='2d')).all()


@pytest.mark.parametrize('cls',[IKM,KM])
def test_constant_columns_and_duplicate_points(cls):
    x=np.array([[0.,1.],[0.,1.],[5.,1.],[5.,1.]])
    m=cls(A0,'2d',init='ward');c=m.fit(x,[np.arange(2)],3)
    assert c is not None and len(c)==3 and all(c) and m.converged_


def test_nonconverged_sse_not_zero_placeholder():
    x=np.array([[0.,0.],[2.,6.],[10.,10.],[12.,16.]])
    m=KM(A0,'2d');m.fit(x,[np.arange(2)],2,max_repetitions=1,hyp_clusters=[{0},{1}])
    assert m.clusters['wd']>0


def test_banco_updates_weights_when_membership_changes_but_mean_does_not():
    calls=[]
    def weights(x,c,**kwargs):
        calls.append(tuple(len(s) for s in c))
        return np.ones((len(c),x.shape[1]))
    b=BANCOIn(weights,'3d');b.V=1;b.beta_degree=1.;b.idxs=[np.arange(1)]
    b.EXTANW(np.ones((2,1)),[0,1],np.empty((0,1)),[],1e-10)
    assert (2,) in calls


def test_banco_honors_iteration_limit():
    x=np.random.default_rng(5).normal(size=(12,2))
    b=BANCOIn(A0,'3d');b.fit(x,[np.arange(2)],2,max_repetitions=1)
    assert all(s['iterations']<=1 for s in b.extraction_status_)
    assert not b.converged_


def test_banco_reports_adaptive_cycle():
    b=BANCOIn(A0,'3d');b.V=1;b.beta_degree=1.;b.idxs=[np.arange(1)]
    def always_move(obj: np.ndarray, len_S: int, c: np.ndarray, w: np.ndarray, er: float=1e-5) -> bool:
        return True
    b.CUR['add']=always_move
    b.CUR['remove']=always_move
    b.EXTANW(np.array([[3.],[1.]]),[0,1],np.empty((0,1)),[],1e-10)
    assert b.extraction_status_[-1]['reason']=='cycle'
    assert not b.extraction_status_[-1]['converged']


def test_reference_and_final_partitions_reject_missing_or_overlapping_objects():
    from Algs.partitions import labels_from_clusters
    for groups in [[{0},{2}],[{0,1},{1,2}]]:
        with pytest.raises(ValueError): labels_from_clusters(groups,3)


@pytest.mark.parametrize('beta',[1.,1.1,2.,5.])
def test_a2_effective_weights_do_not_depend_on_exponent(beta):
    x=np.array([[0.,0.],[2.,6.],[10.,10.],[12.,16.]])
    idxs=[np.array([0]),np.array([1])]
    w=A2(x,[{0,1},{2,3}],b=beta,idxs=idxs,d='2d')
    reference=A2(x,[{0,1},{2,3}],b=1.,idxs=idxs,d='2d')
    np.testing.assert_allclose(w**beta,reference,rtol=1e-12)

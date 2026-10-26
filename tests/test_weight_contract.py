import numpy as np
import pytest
from Algs.weights import A1, A2, feature_groups
from experiments.revision_data import idxs_for_weights, score_partition


def test_interval_and_doubled_simplexes_are_distinct():
    x=np.array([[0.,0.,0.,0.],[2.,6.,20.,60.]])
    clusters=[{0,1}]
    for kind in ['IKM','BANCO']:
        w=A1(x,clusters,b=2,idxs=idxs_for_weights('A1',4,kind))
        assert w[:2].sum()==pytest.approx(1)
        assert w[2:].sum()==pytest.approx(1)
    w=A1(x,clusters,b=2,idxs=idxs_for_weights('A1',4,'KM'))
    assert w.sum()==pytest.approx(1)
    assert not np.isclose(w[:2].sum(),1)


def test_banco_variance_weights_are_invariant_to_replicating_entities():
    x=np.array([[0.,0.],[2.,6.],[10.,10.],[12.,16.]])
    y=np.repeat(x,3,axis=0)
    groups=[np.arange(2)]
    w=A1(x,[{0,1},{2,3}],b=2,idxs=groups,d='3d',dispersion='variance')
    repeated=A1(y,[set(range(6)),set(range(6,12))],b=2,idxs=groups,d='3d',dispersion='variance')
    np.testing.assert_allclose(w,repeated)
    # Independently calculated within-cluster variances plus global-variance mean.
    rho=np.var(x,axis=0).mean()
    D=np.array([1.,9.])+rho
    np.testing.assert_allclose(w[0], (1/D)/(1/D).sum())


@pytest.mark.parametrize('fn,groups',[(A1,[np.arange(2)]),(A2,[np.array([0]),np.array([1])])])
def test_regularized_weights_satisfy_stationarity(fn,groups):
    x=np.array([[0.,0.],[2.,6.],[10.,10.],[12.,16.]])
    c=[{0,1},{2,3}];beta=2.3
    w=fn(x,c,b=beta,idxs=groups,d='2d')
    raw=np.array([4.,36.])
    rho=np.concatenate([np.full(len(g),max(np.var(x[:,g],axis=0).mean(),1e-12)) for g in groups])
    D=raw+rho
    power=beta-1 if fn is A1 else beta
    np.testing.assert_allclose(D*w**power,np.repeat((D*w**power)[0],2))


def test_nmi_matches_manuscript_max_entropy():
    y=np.array([0,0,1,1,2,2]);c=[{0,1,2},{3,4,5}]
    from sklearn.metrics import mutual_info_score
    p=np.array([.5,.5]);q=np.array([1/3]*3)
    expected=mutual_info_score(y,[0,0,0,1,1,1])/max(-sum(p*np.log(p)),-sum(q*np.log(q)))
    assert score_partition(y,c)[1]==pytest.approx(expected)


@pytest.mark.parametrize('groups',[[],[[0,0,1]],[[0]],[[0,2]],[[0.0,1.0]],[[True,False]]])
def test_invalid_feature_groups_rejected(groups):
    with pytest.raises(ValueError):feature_groups(groups,2)


def test_dataset_reader_preserves_duplicate_reference_indices(tmp_path):
    from experiments.revision_data import load_dataset
    np.save(tmp_path/'data.npy',np.array([[0.,1.],[2.,3.]]))
    (tmp_path/'real_clusters.txt').write_text('0, 0\n1')
    with pytest.raises(ValueError,match='duplicate'):
        load_dataset(tmp_path)


def test_huge_finite_interval_preprocessing_retains_geometry():
    from Algs.optimization import preprocess
    x=np.repeat(np.array([[-1e308],[1e308],[-9e307],[9e307]]),2,axis=1)
    np.testing.assert_allclose(preprocess(x),np.repeat([[-1.],[1.],[-.9],[.9]],2,axis=1))


@pytest.mark.parametrize('fn,groups',[(A1,[np.arange(2)]),(A2,[np.array([0]),np.array([1])])])
@pytest.mark.parametrize('beta',[2000.,1e16])
def test_unrepresentable_exponent_domain_rejected(fn,groups,beta):
    x=np.array([[0.,0.],[2.,6.],[10.,10.],[12.,16.]])
    with pytest.raises(ValueError,match='supported range'):
        fn(x,[{0,1},{2,3}],b=beta,idxs=groups)


def test_optimizer_rejects_unsupported_beta_before_false_convergence():
    from Algs.KM import KM
    from Algs.weights import A0
    x=np.array([[0.,0.],[1.,1.],[2.,2.],[10.,10.]])
    with pytest.raises(ValueError,match='supported range'):
        KM(A0).fit(x,[np.arange(2)],2,beta_degree=2000.)


def test_huge_coordinate_optional_std_does_not_collapse():
    from Algs.optimization import preprocess
    x=np.array([[-1e308],[1e308],[-9e307],[9e307]])
    expected=np.array([[-1.],[1.],[-.9],[.9]])/np.sqrt(.905)
    np.testing.assert_allclose(preprocess(x,minmax_normilize=False,scale_by_var=True),expected)


def test_near_one_exponent_rejected_instead_of_erasing_metric():
    from Algs.KM import KM
    x=np.array([[0.,0.],[1.,0.],[2.,0.],[10.,0.]])
    with pytest.raises(ValueError,match='supported range'):
        KM(A1).fit(x,[np.arange(2)],2,hyp_clusters=[{0},{1}],beta_degree=1.001)
    with pytest.raises(ValueError,match='supported range'):
        A1(x,[{0,1},{2,3}],idxs=[np.arange(2)],b=1.000001)


def test_banco_rejects_unrepresentable_unscaled_cur():
    from Algs.BANCOIn import BANCOIn
    from Algs.weights import A0
    m=BANCOIn(A0,center_data=False,minmax_normilize=False)
    x=np.array([[-1e155,0.],[1e155,0.],[2e155,0.]])
    with pytest.raises(ValueError,match='Non-finite'):
        m.fit(x,[np.arange(2)],2)
    for rule in [m.cluster_update_add,m.cluster_update_remove]:
        with pytest.raises(ValueError,match='Non-finite CUR'):
            rule(x[0],2,x[1],np.ones(2))


def test_explained_variance_ratio_divides_before_multiplication():
    from Algs.KM import KM
    from Algs.weights import A0
    x=np.array([[-1e153],[1e153]])
    m=KM(A0,center_data=False,minmax_normilize=False,init='ward')
    m.fit(x,[np.arange(1)],1)
    assert m.clusters['wd']==pytest.approx(100.)

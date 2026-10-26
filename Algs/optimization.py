"""Weighted Lloyd iterations with explicit failure status and exact K groups."""
from typing import Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from .partitions import as_clusters,labels_from_clusters,centers_for_labels,assign_nonempty
from .initialization import ward_partition
from .weights import feature_groups


def validate_iterations(max_iterations):
    if (isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer))
            or max_iterations < 1):
        raise ValueError('max_repetitions must be a positive integer')


def validate_input(data, k, max_iterations):
    x = np.array(data, dtype=float, copy=True)
    if x.ndim != 2 or not np.isfinite(x).all() or x.shape[1] == 0:
        raise ValueError('Expected a finite two-dimensional feature matrix')
    if (isinstance(k, (bool, np.bool_))
            or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(x)):
        raise ValueError('K must be an integer between 1 and the number of objects')
    validate_iterations(max_iterations)
    return x


def validate_dimension(dim):
    if dim not in ('2d', '3d'):
        raise ValueError('d must be 2d or 3d')


def validate_init(init):
    if init not in ('random', 'kmeans++', 'ward'):
        raise ValueError('init must be random, kmeans++, or ward')


def validate_nonnegative(value, name):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or value < 0):
        raise ValueError(f'{name} must be finite and nonnegative')


def validate_options(idxs, n_features, beta, dim):
    validate_dimension(dim)
    validate_nonnegative(beta, 'beta')
    if beta != 1 and not 1.1 <= beta <= 5:
        raise ValueError('beta must be 1 or in the supported range [1.1, 5]')
    return feature_groups(idxs, n_features)


def preprocess(data, center_data=True, minmax_normilize=True, scale_by_var=False):
    """Apply the historical centering, half-range and optional z-score steps."""
    x = np.array(data, dtype=float, copy=True)
    # Use the ordinary calculation for normal-scale inputs, preserving its
    # rounding. Extended intermediates prevent range/mean overflow at the
    # boundary of float64 before normalization brings coordinates back to scale.
    if np.max(np.abs(x), initial=0.) > 1e150:
        x = x.astype(np.longdouble)
    try:
        with np.errstate(over='raise', invalid='raise'):
            if center_data:
                x -= x.mean(axis=0)
            if minmax_normilize:
                half = np.ptp(x, axis=0) / 2.
                x /= np.where(half < 1e-12, 1., half)
            if scale_by_var:
                if x.dtype.itemsize > np.dtype(float).itemsize:
                    # StandardScaler converts to float64 before computing its
                    # variance, losing the extended range used above.
                    x -= x.mean(axis=0)
                    scale = x.std(axis=0)
                    x /= np.where(scale == 0., 1., scale)
                else:
                    x = StandardScaler().fit_transform(x)
    except FloatingPointError as exc:
        raise ValueError('Preprocessing overflow; rescale the input coordinates') from exc
    with np.errstate(over='ignore', invalid='ignore'):
        x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Preprocessing cannot represent these coordinates as float64')
    return x


def default_feature_groups(n_features, weight_name, interval=False):
    """CLI endpoint groups; ordinary KM A1 uses one feature group."""
    if weight_name not in ('A0', 'A1', 'A2'):
        raise ValueError('weightsfunc must be A0, A1, or A2')
    if weight_name == 'A2' or (interval and weight_name == 'A1'):
        if n_features < 2 or n_features % 2:
            raise ValueError('Interval endpoint groups require an even number of features')
        half = n_features // 2
        return [np.arange(half), np.arange(half, n_features)]
    return [np.arange(n_features)]


def squared_distances(x,centers,weights,beta):
    return np.sum((x[:,None,:]-centers[None,:,:])**2 * np.asarray(weights)**beta,axis=2)


def initial_partition(x,k,init,hyp_clusters,sampler):
    validate_init(init)
    if hyp_clusters is None and init=='ward': hyp_clusters=ward_partition(x,k)
    if hyp_clusters is not None:
        if len(hyp_clusters)==0 or len(hyp_clusters)>k or any(len(c)==0 for c in hyp_clusters):
            raise ValueError('Initial groups must be nonempty and at most K')
        labels_from_clusters(hyp_clusters,len(x),require_complete=False)
        centers=x[np.random.choice(len(x),k,replace=False)].copy()
        for j,c in enumerate(hyp_clusters):centers[j]=x[sorted(c)].mean(axis=0)
    else:
        ids=np.random.choice(len(x),k,replace=False) if init=='random' else sampler(x,k)
        centers=x[ids].copy()
    return assign_nonempty(squared_distances(x,centers,np.ones(x.shape[1]),1.))


def lloyd(x,labels,k,weight_func,idxs,beta,dim,max_iterations) -> dict[str, Any]:
    validate_iterations(max_iterations)
    iteration=0
    seen={tuple(labels)};converged=False;reason='iteration_limit'
    for iteration in range(1,max_iterations+1):
        centers=centers_for_labels(x,labels,k)
        weights=weight_func(x,as_clusters(labels,k),b=beta,idxs=idxs,d=dim)
        if not np.isfinite(weights).all():raise ValueError('Non-finite feature weights')
        updated=assign_nonempty(squared_distances(x,centers,weights,beta),labels)
        if np.array_equal(updated,labels):
            converged=True;reason='fixed_point';break
        labels=updated
        key=tuple(labels)
        if key in seen:reason='cycle';break
        seen.add(key)
    centers=centers_for_labels(x,labels,k)
    weights=weight_func(x,as_clusters(labels,k),b=beta,idxs=idxs,d=dim)
    residual=x-centers[labels]
    with np.errstate(over='ignore', invalid='ignore'):
        sse=float(np.sum(residual**2));scatter=float(np.sum((x-x.mean(0))**2))
    if not np.isfinite(sse) or not np.isfinite(scatter):
        raise ValueError('Non-finite residual/scatter sums; enable scaling or rescale input')
    return {'labels':labels,'centers':centers,'weights':weights,'iterations':iteration,
            'converged':converged,'reason':reason,'sse':sse,'wd':100*(sse/scatter) if scatter>0 else 0.}

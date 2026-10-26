"""Feature weights for explicitly regularized dispersions.

Lloyd uses D[k,j] = SSE[k,j] + rho[group(j)], or pooled SSE for global
weights. rho is the mean population variance of the input columns in the
group, floored at 1e-12, and is independent of the current partition.
BANCO requests dispersion='variance': D[k,j] = SSE[k,j]/n[k] + rho,
matching its within-cluster WUR definition. Adaptive BANCO remains heuristic.

A1 minimizes sum(D*w**beta) with sum(w)=1 separately in each supplied group.
A2 minimizes the same expression with product(w)=1 across both endpoint groups.
At beta=1 A1 permits zero weights (a boundary case, outside the positive-weight
main experiment). At beta>1 weights use stable inverse powers.
"""
import numpy as np


def feature_groups(idxs, n_cols):
    """Validate an exhaustive, disjoint partition of coordinate indices."""
    try:
        groups = []
        for idx in idxs:
            members = list(idx)
            if any(isinstance(j, (bool, np.bool_)) for j in members):
                raise ValueError('Boolean feature indices are not allowed')
            groups.append(np.asarray(members))
    except (TypeError, ValueError) as exc:
        raise ValueError('Feature groups must be a sequence of index arrays') from exc
    if not groups or any(g.ndim != 1 or g.size == 0 or g.dtype.kind not in 'iu' for g in groups):
        raise ValueError('Feature groups must be nonempty one-dimensional integer arrays')
    flat = np.concatenate(groups)
    if np.any(flat < 0) or np.any(flat >= n_cols):
        raise ValueError('Feature index out of range')
    if flat.size != n_cols or np.unique(flat).size != n_cols:
        raise ValueError('Feature groups must cover each coordinate exactly once')
    return [g.astype(np.intp, copy=False) for g in groups]


def A0(data, clusters, **kwargs):
    dim = kwargs.get('d', '2d')
    if dim not in ('2d', '3d'):
        raise ValueError('Weight dimension must be 2d or 3d')
    shape = (len(clusters), data.shape[1]) if dim == '3d' else data.shape[1]
    return np.ones(shape, dtype=float)


def cluster_center(cl_data):
    return np.mean(cl_data, axis=0) if len(cl_data) else np.zeros(cl_data.shape[-1])


def weighted_cluster_variance(cl_data):
    return np.sum((cl_data - cluster_center(cl_data))**2, axis=0)


def _dispersions(data, clusters, idx, dim, dispersion='sse'):
    if dispersion not in ('sse', 'variance'):
        raise ValueError('dispersion must be sse or variance')
    part = np.asarray(data[:, idx], float)
    if not len(part) or not clusters:
        raise ValueError('Nonempty data and a cluster list are required')
    variances = np.array([weighted_cluster_variance(part[sorted(c)]) for c in clusters])
    if dispersion == 'variance':
        # Empty clusters occur only in compatibility helpers; their SSE is zero.
        variances /= np.maximum([len(c) for c in clusters], 1)[:, None]
    D = variances if dim == '3d' else variances.sum(axis=0)
    rho = max(float(np.mean(weighted_cluster_variance(part) / len(part))), 1e-12)
    result = D + rho
    if not np.isfinite(result).all():
        raise ValueError('Non-finite regularized dispersions')
    return result


def _beta(kwargs):
    beta = float(kwargs['b'])
    if not np.isfinite(beta) or (beta != 1 and not 1.1 <= beta <= 5):
        raise ValueError('beta must be 1 or in the supported range [1.1, 5]')
    return beta


def A1(data, clusters, **kwargs):
    beta = _beta(kwargs)
    dim = kwargs.get('d', '2d')
    weights = np.zeros_like(A0(data, clusters, d=dim))
    for idx in feature_groups(kwargs['idxs'], data.shape[1]):
        D = _dispersions(data, clusters, idx, dim, kwargs.get('dispersion', 'sse'))
        if beta == 1.:
            minima = D == D.min(axis=-1, keepdims=True)
            w = minima / minima.sum(axis=-1, keepdims=True)
        else:
            logs = np.log(D)
            logits = -(logs - logs.min(axis=-1, keepdims=True)) / (beta - 1.)
            w = np.exp(logits)
            w /= w.sum(axis=-1, keepdims=True)
        weights[..., idx] = w
    return weights


def A2(data, clusters, inf=1e10, **kwargs):
    beta = _beta(kwargs)
    dim = kwargs.get('d', '2d')
    groups = feature_groups(kwargs['idxs'], data.shape[1])
    if len(groups) != 2:
        raise ValueError('A2 requires lower and upper endpoint groups')
    D = np.concatenate([_dispersions(data, clusters, idx, dim, kwargs.get('dispersion', 'sse'))
                        for idx in groups], axis=-1)
    logs = np.log(D)
    w = np.exp((logs.mean(axis=-1, keepdims=True) - logs) / beta)
    weights = np.zeros_like(A0(data, clusters, d=dim))
    weights[..., np.concatenate(groups)] = w
    return weights

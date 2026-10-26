"""Initial partitions used by the existing IKM/KM optimizers."""
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree


def ward_partition(standardized_data, k):
    """Merge by Ward SSE increase, stopping at exactly k nonempty clusters.

    Input must already use the optimizer's coordinate scaling. SciPy's
    Ward linkage minimizes n_a*n_b/(n_a+n_b)*||mean_a-mean_b||**2.
    Labels and adaptive feature weights are not used for initialization.
    """
    x = np.asarray(standardized_data, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Ward requires a finite 2D data matrix")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(x):
        raise ValueError("k must be an integer between 1 and n")
    if k == len(x):
        return [{i} for i in range(len(x))]
    if k == 1:
        return [set(range(len(x)))]
    labels = cut_tree(linkage(x, method="ward", metric="euclidean"), n_clusters=[k]).ravel()
    return [set(np.flatnonzero(labels == j).tolist()) for j in range(k)]

"""Partition validation, nearest-center assignment, and nonempty repair."""
import numpy as np


def labels_from_clusters(clusters, n, require_complete=True):
    labels = np.full(n, -1, dtype=int)
    if clusters is None:
        raise ValueError('Expected nonempty index groups')
    group_count = 0
    for k, cluster in enumerate(clusters):
        group_count += 1
        try:
            members = list(cluster)
        except TypeError as error:
            raise ValueError('Expected a one-dimensional integer index group') from error
        if not members:
            raise ValueError('Index groups must be nonempty')
        if any(isinstance(i, (bool, np.bool_))
               or not isinstance(i, (int, np.integer)) for i in members):
            raise ValueError('Cluster contains an invalid object index')
        if any(i < 0 or i >= n for i in members):
            raise ValueError('Cluster contains an invalid object index')
        ids = np.asarray(members, dtype=int)
        if len(np.unique(ids)) != len(ids) or np.any(labels[ids] >= 0):
            raise ValueError('Clusters overlap or contain duplicate indices')
        labels[ids] = k
    if not group_count:
        raise ValueError('Index groups must be nonempty')
    if require_complete and np.any(labels < 0):
        raise ValueError('Partition leaves objects unassigned')
    return labels


def read_clusters(path, n, require_complete=True):
    """Read comma-separated object indices without losing duplicates or EOF data."""
    clusters = []
    with open(path, encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                members = [int(token.strip()) for token in line.strip().split(',')]
            except ValueError as error:
                raise ValueError(f'Invalid cluster indices on line {line_number}') from error
            clusters.append(members)
    labels_from_clusters(clusters, n, require_complete=require_complete)
    return clusters


def as_clusters(labels,k):
    return [set(np.flatnonzero(labels==j).tolist()) for j in range(k)]


def centers_for_labels(x,labels,k):
    centers=np.zeros((k,x.shape[1]))
    for j in range(k):
        members=x[labels==j]
        if len(members): centers[j]=members.mean(axis=0)
    return centers


def assign_nonempty(distances, previous=None):
    """Rows are objects, columns clusters. Retain previous assignments on ties."""
    distances=np.asarray(distances,float)
    n,k=distances.shape
    if not 1<=k<=n or not np.isfinite(distances).all():
        raise ValueError('Invalid distances or cluster count')
    labels=np.argmin(distances,axis=1)
    if previous is not None:
        previous=np.asarray(previous)
        valid=(previous>=0)&(previous<k)
        ids=np.flatnonzero(valid)
        tied=distances[ids,previous[ids]]==distances[ids,labels[ids]]
        labels[ids[tied]]=previous[ids[tied]]
    counts=np.bincount(labels,minlength=k)
    for empty in np.flatnonzero(counts==0):
        eligible=np.flatnonzero(counts[labels]>1)
        residuals=distances[eligible,labels[eligible]]
        chosen=eligible[np.argmax(residuals)]
        counts[labels[chosen]]-=1
        labels[chosen]=empty
        counts[empty]+=1
    return labels

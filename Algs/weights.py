import numpy as np

def A0(data, clusters, **kwargs):
  if kwargs.setdefault("d", "2d") == '3d':
    return np.full((len(clusters), data.shape[1]), 1., dtype=np.float32)
  return np.full(data.shape[1], 1., dtype=np.float32)

def cluster_center(cl_data):
  if cl_data.shape[0] == 0:
    # case when cluster is empty
    return np.full(cl_data.shape[-1], 0.)
  elif cl_data.shape[0] == 1:
    return cl_data
  return np.mean(cl_data, axis=0)

def weighted_cluster_variance(cl_data: np.ndarray,                              # data array of obj for clastering with size: (obj_num, feature_num)
                              ) -> np.ndarray:                                  # return: feature variance within cluster

  # mean by objects writen |objects| times
  mean_data = np.full(cl_data.shape, cluster_center(cl_data))
  var = np.sum((cl_data - mean_data)**2, axis=0)
  return var

def A1(data, clusters, **kwargs):
  beta = kwargs['b'] # power of weights
  idxs = kwargs['idxs'] # indicies of lower/upper boundaries


  w_shape = (len(clusters), data.shape[1]) if kwargs.setdefault("d", "2d") == '3d' else data.shape[1]
  weights = np.zeros(w_shape, dtype=np.float32)
  for idx in idxs:
    data_part = data[:, idx]
    var = np.zeros((len(clusters), data_part.shape[1]), dtype=np.float32)
    for id, cluster in enumerate(clusters):
      cl_data = data_part[list(cluster)]
      var[id] = weighted_cluster_variance(cl_data)

    D = var if kwargs.setdefault("d", "2d") == '3d' else np.sum(var, axis=0)
    D += np.mean(weighted_cluster_variance(data_part) / data_part.shape[0])

    D_u = ((1 / D)**(1 / (beta - 1))).sum(axis=-1)
    D = D**(1 / (beta - 1))
    weights[..., idx] = 1 / (D * np.array([D_u]).T)
  return weights


def A2(data, clusters, inf=1e10, **kwargs):

  idxs = kwargs['idxs'] # indicies of lower/upper boundaries

  w_shape = (len(clusters), data.shape[1]) if kwargs.setdefault("d", "2d") == '3d' else data.shape[1]

  adequacy_a = [weighted_cluster_variance(data[list(cluster)][:, idxs[0]]) for cluster in clusters]
  adequacy_b = [weighted_cluster_variance(data[list(cluster)][:, idxs[1]]) for cluster in clusters]

  adequacy_a = np.array(adequacy_a) if kwargs['d'] == '3d' else np.sum(adequacy_a, axis=0)
  adequacy_b = np.array(adequacy_b) if kwargs['d'] == '3d' else np.sum(adequacy_b, axis=0)

  adequacy_a += np.mean(weighted_cluster_variance(data[:, idxs[0]]) / data.shape[0])
  adequacy_b += np.mean(weighted_cluster_variance(data[:, idxs[1]]) / data.shape[0])

  prod = (np.prod(adequacy_a, axis=-1) * np.prod(adequacy_b, axis=-1))**(1 / (2* len(idxs[0])))

  weights = np.zeros(w_shape, dtype=np.float64)
  weights[..., idxs[0]] = np.array([prod]).T / adequacy_a
  weights[..., idxs[1]] = np.array([prod]).T / adequacy_b

  return weights
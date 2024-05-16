import argparse
from weights import A0, A1, A2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import copy
import random


class IKM():
  def __init__(self, 
               weight_func, 
               d = '3d', 
               **kwargs):
    self.w_update_f = weight_func
    self.d = d

    #normalization flags
    self.FLAG_CENTER_DATA = kwargs.setdefault("center_data", True)
    self.FLAG_MINMAX_NORAMILIZE_DATA = kwargs.setdefault("minmax_normilize", True)
    self.FLAG_SCALE_BY_VAR = kwargs.setdefault("scale_by_var", True)



  def normalize_data(self, 
                     data: np.ndarray
                     ):
    """
    Standardizes data given the data matrix.

    Args:
        data (numpy.ndarray): The data matrix.

    Returns:
        numpy.ndarray: Standardized data.
    """ 

    # 0. Data preprocessing
    # 0.1. centering
    if self.FLAG_CENTER_DATA:
      data -= np.full(data.shape, np.mean(data, axis=0))

    # 0.2. normilize
    if self.FLAG_MINMAX_NORAMILIZE_DATA:
      data = MinMaxScaler().fit_transform(data)

    if self.FLAG_SCALE_BY_VAR:
      data = StandardScaler().fit_transform(data)
    return data


  def cluster_best_prototype(self, 
                             cl_data: np.ndarray
                             ) -> np.ndarray:
    """
    Calculates the cluster center given the data matrix of cluster members.

    Args:
        cl_data (numpy.ndarray): The data matrix of cluster members.

    Returns:
        numpy.ndarray: The cluster center.
    """

    if cl_data.shape[0] == 0:
      # case when cluster is empty
      return np.full(self.V, 0.)
    elif cl_data.shape[0] == 1:
      return cl_data
    return np.mean(cl_data, axis=0)

  def euclidian_d(self,
                  x: np.ndarray,
                  y: np.ndarray,
                  ) -> np.ndarray:
    """
    Calculates the squared difference between x and y.

    Args:
        x  (numpy.ndarray): object 1
        y  (numpy.ndarray): object 2

    Returns:
        numpy.ndarray: Squared difference between x and y.
    """

    return (x - y)**2

  def obj_to_cl_d(self, 
                  obj:                    np.ndarray, 
                  cl_best_representative: np.ndarray, 
                  k:                      int, 
                  cluster_len:            int
                  ) -> float:
    """
    Calculates the weighted Euclidian distance between x and y.

    Args:
        obj                     (numpy.ndarray):  object 
        cl_best_representative  (numpy.ndarray):  cluster best representative
        k                       (int):            cluster index
        cluster_len             (int):            cluster length

    Returns:
        numpy.ndarray: Weighted Euclidian distancebetween x and y.
    """

    if cluster_len == 0:
      return 0
    weights = self.weights[k]**self.beta_degree \
              if self.d == '3d' \
              else self.weights**self.beta_degree

    return np.sum(weights * self.euclidian_d(obj, cl_best_representative))

  # Algorithm:
  # 0. Data preprocessing:
  #   0.1. centering
  #   0.2. normalization
  # 1. initialization block:
  #   1.1. choose C random best prototypes
  #   1.2. get their feature representations
  #   1.3. initialize weights
  # 2. conditional block for clusters initialization:
  #   2.1. initialization from random clusters
  #   2.2. initialization from user-specified clusters
  # 3. completion of clusters with unmathed objects
  # repeat:
  #   4. representation step
  #   5. weighting step
  #   6. partition step
  # until iteration without changing clusters
  def fit(self,
          data:             np.ndarray,                                                       # data
          idxs:             list,                                                                 # ids of lower and upper bounds in data
          K:                int,                                                                    # number of clusters
          max_repetitions:  int = 1000,                                          # max algorithm iterations
          hyp_clusters:     list = None,                                            # hypothetical clusters to initialize
          beta_degree:      float = 1.,):                                            # weights degree
    """
    Performs IKM clustering on the input data matrix X.

    Args:
        data            (numpy.ndarray):  data array of objs for clustering.
        idxs            (list):           ids of lower and upper bounds in data.
        K               (int):            the number of clusters to be found.
        max_repetitions (int):            max algorithm iterations.
        hyp_clusters    (list):           hypothetical clusters to initialize.
        beta_degree     (float):          weights degree.

    Returns:
        list: k found clusters.
    """

    self.N, self.V = data.shape
    self.idxs = idxs
    self.beta_degree = beta_degree

    # 0. Data preprocessing:
    data = self.normalize_data(data)

    # 1. initialization block
    # 1.1 choose C random best prototypes
    g_idx = random.sample(range(self.N), K)
    # 1.2. get their feature representations
    BP = data[g_idx]
    # 1.3. initialize weights
    self.weights = np.full((K, self.V), 1, dtype=np.float64)\
                   if self.d == '3d'\
                   else np.full(self.V, 1, dtype=np.float64)

    # 2. conditional block for clusters initialization:
    # 2.1. run from random clusters
    # 2.2. run from user-specified clusters
    if hyp_clusters is None:
      # 2.1. initialization from scratch
      clusters = [set([g_idx[i]]) for i in range(K)]
    else:
      # 2.2. initialization from hypothetical clusters
      for k in range(K):
        BP[k] = self.cluster_best_prototype(data[list(hyp_clusters[k])])
      clusters = copy.deepcopy(hyp_clusters)

    obj_to_cluster_map = {e_i:k for k in range(K) for e_i in clusters[k]}
    # list of unmatched objects
    I_k = list(set(range(self.N)) - set(obj_to_cluster_map.keys()))

    # 3. completion of clusters
    # add rest objects to clusters with minimum(by prototypes) distances
    min_id = {e_i:np.argmin([self.obj_to_cl_d(data[e_i], BP[m], m, len(list(clusters[m]))) for m in range(K)]) for e_i in I_k}
    for k in range(K):
      clusters[k].update({e_i for e_i in I_k\
                      if min_id[e_i] == k})
      I_k = list(set(I_k) - clusters[k])
      obj_to_cluster_map.update({e_i:k for e_i in clusters[k]})


    for i in range(max_repetitions):
      # 4. representation step
      for k in range(K):
        BP[k] = self.cluster_best_prototype(data[list(clusters[k])])

      # 5. weighting step
      self.weights = self.w_update_f(data, clusters, b=beta_degree, idxs=self.idxs, d=self.d)

      # 6. Defenition of the best partition
      test = False
      for e_i in range(self.N):
        P_h = obj_to_cluster_map[e_i]
        d = [self.obj_to_cl_d(data[e_i], BP[m], m, len(list(clusters[m]))) for m in range(K)]
        P_k = np.argmin(d)
        if P_h != P_k:
          test = True
          clusters[P_k].add(e_i)
          clusters[P_h].remove(e_i)
          obj_to_cluster_map[e_i] = P_k

      if not test:
        break
    return clusters
  



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataroot', required=True, help='path to dataset')
  parser.add_argument('--weightsfunc', required=True, help='weights function from the list: A0(without weights) | A1(Euclidian) } A2(Corvalho)')


  parser.add_argument('--niter', type=int, default=1000, help='max algorithm iterations')
  parser.add_argument('--hypclroot', required=False, help='path to hypothetical clusters to initialize')
  parser.add_argument('--realclroot', required=False, help='path to real clusters to calculate ARI and NMI scores')
  parser.add_argument('--degree', type=float, default=1., help='weights degree')
  parser.add_argument('--error', type=float, default=1e-5, help='max error for cluster update rule')
  parser.add_argument('--d', default='3d', required=False, help='way to calculate weights: 2d | 3d')
  parser.add_argument('--K', required=False, help='number of clusters')
  parser.add_argument('--manualSeed', type=int, help='manual seed')



  opt = parser.parse_args()

  if opt.manualSeed is None:
    opt.manualSeed = np.random.randint(1, 10000, size=1)[0]
  print("Random Seed: ", opt.manualSeed)
  np.random.seed(opt.manualSeed)

  if opt.dataroot is None:
     raise ValueError("`dataroot` parameter is required")
  dataset = np.load(opt.dataroot)


  hyp_clusters = None
  if opt.hypclroot is not None:
    hyp_clusters = []
    with open(opt.hypclroot, 'r') as fp:
        for line in fp:
            x = line[:-1]
            hyp_clusters.append(set(map(int, x.split(", "))))
  
  real_clusters = None
  if opt.realclroot is not None:
    real_clusters = []
    with open(opt.realclroot, 'r') as fp:
        for line in fp:
            x = line[:-1]
            real_clusters.append(set(map(int, x.split(", "))))

  if opt.weightsfunc == 'A0':
    weights_func = A0
  elif opt.weightsfunc == 'A1':
    weights_func = A1
  else:
    weights_func = A2

  l, u = dataset.shape[1] // 2, dataset.shape[1]
  idxs = [np.arange(l), np.arange(l, u)]

  method = IKM(weights_func, opt.d)
  res = method.fit(dataset, idxs, int(opt.K), max_repetitions=int(opt.niter), hyp_clusters=hyp_clusters, beta_degree=float(opt.degree))
  for cl_id, cluster in enumerate(res):
    print(f"cluster [{cl_id}]: ", *cluster)

  if real_clusters is not None:
    r_labels = np.full(dataset.shape[0], 0)
    for i in range(len(real_clusters)):
      r_labels[list(real_clusters[i])] = i

    alg_labels = np.full(dataset.shape[0], 0)
    for i in range(len(res)):
      alg_labels[list(res[i])] = i

    print("ARI: ", round(adjusted_rand_score(r_labels, alg_labels), 2),
          "\tNMI: ", round(normalized_mutual_info_score(r_labels, alg_labels), 2))
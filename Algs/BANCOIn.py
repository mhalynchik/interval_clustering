import argparse
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Algs.partitions import read_clusters, labels_from_clusters
from Algs.optimization import (preprocess, validate_options,
                               validate_nonnegative, default_feature_groups)
from Algs.optimization import validate_input


class BANCOIn():
  def __init__(self,
               weight_func,
               d = '3d',
               **kwargs):

    self.w_update_f = weight_func
    if d != '3d':
      raise ValueError("BANCO requires d='3d' because its weights are cluster-specific")
    self.d = d

    #normalization flags
    self.FLAG_CENTER_DATA = kwargs.setdefault("center_data", True)
    self.FLAG_MINMAX_NORAMILIZE_DATA = kwargs.setdefault("minmax_normilize", True)
    self.FLAG_SCALE_BY_VAR = kwargs.setdefault("scale_by_var", False)


    self.CUR = {'remove': self.cluster_update_remove,
                'add': self.cluster_update_add}


  def normalize_data(self,
                     data: np.ndarray
                     ) -> np.ndarray:

    """
    Standardizes data given the data matrix.

    Args:
        data (numpy.ndarray): The data matrix.

    Returns:
        numpy.ndarray: Standardized data.
    """

    return preprocess(data, self.FLAG_CENTER_DATA,
                      self.FLAG_MINMAX_NORAMILIZE_DATA, self.FLAG_SCALE_BY_VAR)


  def cluster_center(self,
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
      return np.asarray(cl_data, dtype=np.float64).reshape(-1)
    return np.mean(cl_data, axis=0)


  def cluster_update_remove(self,
                            obj:    np.ndarray,
                            len_S:  int,
                            c:      np.ndarray,
                            w:      np.ndarray,
                            er:     float = 1e-5,
                            ) -> bool:
    """
    Determines whether an object should be removed from a cluster based on the weighted Euclidean distance
    to the cluster center.

    Args:
        obj   (numpy.ndarray):  object considered for inclusion to the cluster.
        len_S (int):            cluster length.
        c     (numpy.ndarray):  cluster centroid.
        w     (numpy.ndarray):  features weights.
        er    (float):          error.

    Returns:
        bool: decision whether to remove an object or not.
    """

    with np.errstate(over='ignore', invalid='ignore'):
      value = (len_S * np.inner(w * c, c) -
               2 * len_S * np.inner(w * c, obj) + np.inner(w * obj, obj))
    if not np.isfinite(value):
      raise ValueError('Non-finite CUR expression; rescale input coordinates')
    return bool(value > er)

  def cluster_update_add(self,
                         obj:   np.ndarray,
                         len_S: int,
                         c:     np.ndarray,
                         w:     np.ndarray,
                         er:    float = 1e-5,
                         ) -> bool:
    """
    Determines whether an object should be added to a cluster based on the weighted Euclidean distance
    to the cluster center.

    Args:
        obj   (numpy.ndarray):  object considered for inclusion to the cluster.
        len_S (int):            cluster length.
        c     (numpy.ndarray):  cluster centroid.
        w     (numpy.ndarray):  features weights.
        er    (float):          error.

    Returns:
        bool: decision whether to add an object or not.
    """

    with np.errstate(over='ignore', invalid='ignore'):
      value = (len_S * np.inner(w * c, c) -
               2 * len_S * np.inner(w * c, obj) - np.inner(w * obj, obj))
    if not np.isfinite(value):
      raise ValueError('Non-finite CUR expression; rescale input coordinates')
    return bool(value < -er)


  def EXTANW(self,
             data:      np.ndarray,
             I_k:       list,
             centroids: np.ndarray,
             clusters:  list,
             er:        float = 1e-5,
             ) -> tuple:
    """
    Performs EXTANW clustering to find anomalous cluster.

    Args:
        data      (numpy.ndarray):  data array of objs for clustering.
        I_k       (list):           list of unmatched objects.
        centroids (numpy.ndarray):  cluster centroids.
        clusters  (list):           founded clusters.
        er        (float):          error.

    Returns:
        tuple: A tuple containing the following elements:
            - clusters  (list):           updated clusters
            - centroids (numpy.ndarray):  updated centroids
    """


    validate_nonnegative(er,'er')
    with np.errstate(over='ignore', invalid='ignore'):
      norms = np.array([np.inner(data[i],data[i]) for i in I_k])
    if not np.isfinite(norms).all():
      raise ValueError('Non-finite seed distances; enable scaling or rescale input')
    c_idx = I_k[np.argmax(norms)]
    clusters.append({c_idx})
    centroids=np.vstack((centroids,data[c_idx]))
    w=np.ones(self.V,dtype=float)
    iteration=0
    seen=set();converged=False;reason='iteration_limit'
    for iteration in range(1,getattr(self,'max_repetitions_',1000)+1):
      key=(frozenset(clusters[-1]),w.tobytes())
      if key in seen:
        reason='cycle';break
      seen.add(key)
      changed=False
      for i in I_k:
        w_b=w**self.beta_degree
        if i in clusters[-1]:
          move=len(clusters[-1])>1 and self.CUR['remove'](data[i],len(clusters[-1]),centroids[-1],w_b,er)
          if move:clusters[-1].remove(i)
        else:
          move=self.CUR['add'](data[i],len(clusters[-1]),centroids[-1],w_b,er)
          if move:clusters[-1].add(i)
        if not move:continue
        changed=True
        centroids[-1]=self.cluster_center(data[sorted(clusters[-1])])
        # Dispersion and weights can change even if the mean is unchanged.
        w=self.w_update_f(data,clusters,b=self.beta_degree,idxs=self.idxs,d='3d',dispersion='variance')[-1]
        if not np.isfinite(w).all():raise ValueError('Non-finite BANCO weights')
      updated_w=self.w_update_f(data,clusters,b=self.beta_degree,idxs=self.idxs,d='3d',dispersion='variance')[-1]
      if not changed and np.array_equal(w,updated_w):
        converged=True;reason='fixed_point';break
      w=updated_w
    if not hasattr(self,'extraction_status_'):self.extraction_status_=[]
    self.extraction_status_.append({'converged':converged,'reason':reason,'iterations':iteration})
    return clusters,centroids


  def fit(self,
          data:             np.ndarray,                                           # data
          idxs:             list,                                                 # ids of lower and upper bounds in data
          K:                int,                                                  # number of clusters
          max_repetitions:  int   = 1000,                                         # max algorithm iterations
          hyp_clusters:     list | None  = None,                                         # hypothetical clusters to initialize
          beta_degree:      float = 1.,                                           # weights degree
          er:               float = 1e-5,                                         # error
          ) -> list:
    """
    Performs BANCOIn clustering on the input data matrix X.

    Args:
        data            (numpy.ndarray):  data array of objs for clustering.
        idxs            (list):           ids of lower and upper bounds in data.
        K               (int):            the number of clusters to be found.
        max_repetitions (int):            max algorithm iterations.
        hyp_clusters    (list):           hypothetical clusters to initialize.
        beta_degree     (float):          weights degree.
        er              (float):          error.

    Returns:
        list: k found clusters.
    """


    data=validate_input(data,K,max_repetitions)
    idxs=validate_options(idxs,data.shape[1],beta_degree,self.d)
    validate_nonnegative(er,'er')
    if hyp_clusters is not None:raise ValueError('BANCO does not accept initial clusters')
    self.max_repetitions_=max_repetitions
    self.extraction_status_=[]
    self.N, self.V = data.shape
    self.idxs = idxs
    self.beta_degree = beta_degree

    # 0. Data preprocessing:
    data = self.normalize_data(data)

    # 1. initialization block
    I_k = list(range(self.N))
    # 2. Iterative EXTAN
    clusters = []
    centroids = np.array([], dtype=np.float64).reshape(0, self.V)
    while len(I_k) != 0:
      clusters, centroids = self.EXTANW(data, I_k, centroids, clusters, er)
      I_k = list(set(I_k) - clusters[-1])
    # 3. Large clusters center selection
    self.converged_=all(t['converged'] for t in self.extraction_status_)
    self.n_iter_=sum(t['iterations'] for t in self.extraction_status_)
    S = sorted(clusters, key=lambda x: len(x), reverse=True)[:K]
    # 4. Output K anamalous clusters
    return S

if __name__ == "__main__":
  from Algs.weights import A0, A1, A2

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataroot', required=True, help='path to dataset')
  parser.add_argument('--weightsfunc', required=True, choices=['A0', 'A1', 'A2'], help='weights function from the list: A0(without weights) | A1(Euclidian) } A2(Corvalho)')


  parser.add_argument('--niter', type=int, default=1000, help='max algorithm iterations')
  parser.add_argument('--hypclroot', required=False, help='path to hypothetical clusters to initialize')
  parser.add_argument('--realclroot', required=False, help='path to real clusters to calculate ARI and NMI scores')
  parser.add_argument('--degree', type=float, default=1., help='weights degree')
  parser.add_argument('--error', type=float, default=1e-5, help='max error for cluster update rule')
  parser.add_argument('--d', default='3d', choices=['3d'], required=False, help='BANCO uses cluster-specific weights (3d)')
  parser.add_argument('--K', type=int, required=True, help='number of clusters')
  parser.add_argument('--manualSeed', type=int, help='manual seed')


  opt = parser.parse_args()

  if opt.manualSeed is None:
    opt.manualSeed = np.random.randint(1, 10000, size=1)[0]
  print("Random Seed: ", opt.manualSeed)
  np.random.seed(opt.manualSeed)

  if opt.dataroot is None:
     raise ValueError("`dataroot` parameter is required")
  dataset = np.load(opt.dataroot)


  hyp_clusters = (read_clusters(opt.hypclroot, len(dataset), require_complete=False)
                  if opt.hypclroot is not None else None)
  real_clusters = (read_clusters(opt.realclroot, len(dataset))
                   if opt.realclroot is not None else None)

  if opt.weightsfunc == 'A0':
    weights_func = A0
  elif opt.weightsfunc == 'A1':
    weights_func = A1
  else:
    weights_func = A2

  idxs = default_feature_groups(dataset.shape[1], opt.weightsfunc, interval=True)

  method = BANCOIn(weights_func, opt.d)
  res = method.fit(dataset, idxs, int(opt.K), max_repetitions=int(opt.niter), hyp_clusters=hyp_clusters, beta_degree=float(opt.degree), er=float(opt.error))
  for cl_id, cluster in enumerate(res):
    print(f"cluster [{cl_id}]: ", *cluster)


  if real_clusters is not None:
    r_labels=labels_from_clusters(real_clusters,dataset.shape[0])
    try:
      alg_labels=labels_from_clusters(res,dataset.shape[0])
    except ValueError as error:
      print(f'ARI/NMI not computed for partial initialization: {error}')
      raise SystemExit(0)

    print("ARI: ", round(adjusted_rand_score(r_labels, alg_labels), 2),
          "\tNMI: ", round(normalized_mutual_info_score(r_labels, alg_labels, average_method='max'), 2))

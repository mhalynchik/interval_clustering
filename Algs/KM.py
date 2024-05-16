import argparse
from weights import A0, A1, A2
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

class KM():
  def __init__(self, 
               weight_func, 
               d = '3d', 
               **kwargs):
      
    self.w_update_f = weight_func
    self.d = d
    self.verbose = kwargs.setdefault('verbose', False)
  
  def clusterupdate(self, 
                    Y:    np.ndarray, 
                    cent: np.ndarray
                    ) -> tuple:
    """
    Updates the cluster assignments and computes the squared Euclidean distance criterion.

    Args:
        Y       (numpy.ndarray): The standardized data matrix.
        cent    (numpy.ndarray): The matrix of cluster centers.

    Returns:
        tuple: A tuple containing the following elements:
            - labelc (numpy.ndarray): The vector of cluster labels for each observation.
            - wc (float): The value of the squared Euclidean distance criterion.
    """
    K, m = cent.shape
    N, m = Y.shape

    # Computing distances from each object to each center
    disto = np.zeros((K, N))
    wdisto = np.zeros((K, N))
    for k in range(K):
        cc = cent[k, :]
        Ck = np.repeat(cc[np.newaxis, :], N, axis=0)
        dif = Y - Ck
        weights = self.weights if self.d == '2d' else self.weights[k]
        weights = weights**self.beta_degree
        ddif = dif * dif
        wddif = weights * ddif
        wdisto[k] = np.sum(wddif, axis=1)
        disto[k] = np.sum(ddif, axis=1)

    # Using the Minimum distance rule for computing output
    aa, bb = np.min(disto, axis=0), np.argmin(wdisto, axis=0)
    if self.K > np.unique(bb).shape[0]:
        K_new = np.unique(bb).shape[0]
        flaw = self.K - np.unique(bb).shape[0]
        farthest_objs = np.sort(np.argpartition([np.min(wdisto, axis=1)], flaw, axis=1)[:,-flaw:], axis=1)[0]
        empty_cl = list(set(range(self.K)) - set(np.unique(bb)))
        for cl_id, obj_id in zip(empty_cl, farthest_objs):
            bb[obj_id] = cl_id

        if self.verbose:
            print("Found only {} out of {} clusters, separate the farthest objects: [{}] into a separate clusters".format(K_new, self.K, ", ".join(map(str, farthest_objs))))
    
    wc = np.sum(aa)  # Unexplained variance
    labelc = bb

    return labelc, wc
  
  def ceupdate(self, 
               X:       np.ndarray, 
               labelc:  np.ndarray
               ) -> np.ndarray:
    """
    Updates the cluster centers given the data matrix and cluster assignments.

    Args:
        X       (numpy.ndarray): The data matrix.
        labelc  (numpy.ndarray): The vector of cluster labels for each observation.

    Returns:
        numpy.ndarray: The updated cluster centers.
    """
    K = np.max(labelc) + 1
    clusters = [set(np.where(labelc == k)[0]) for k in range(K)]
    self.weights = self.w_update_f(X, clusters, b=self.beta_degree, idxs=self.idxs, d=self.d)


    K = np.max(labelc) + 1  # The number of clusters
    centres = np.zeros((K, X.shape[1]))
    num = 0
    for kk in range(K):
        num += 1
        clk = np.where(labelc == kk)[0]  # kk-th cluster list
        nc = len(clk)
        elemk = X[clk, :]  # Data matrix over the cluster
        if nc == 0:
            num -= 1
            if self.verbose:
                print(f'At index {num} cluster is empty')
        elif nc == 1:
            centres[num - 1, :] = elemk  # Center of singleton cluster = itself
        else:
            centres[num - 1, :] = np.mean(elemk, axis=0)

    return centres

  def fit(self, 
          X:                np.ndarray,
          idxs:             list, 
          k:                int,
          max_repetitions:  int = 1000,
          hyp_clusters:     list = None,
          beta_degree:      float = 1.
          ) -> list:
      """
      Performs k-means clustering on the input data matrix X.

      Args:
          X               (numpy.ndarray):  The input data matrix, where rows are observations and columns are variables.
          idxs            (list):           ids of lower and upper bounds in data.
          k               (int):            The number of clusters to be formed.
          max_repetitions (int):            max algorithm iterations.
          hyp_clusters    (list):           hypothetical clusters to initialize.
          beta_degree     (float):          weights degree.
          
      Returns:
          list: k found clusters.
      """
      N, m = X.shape
      self.K = k

      self.beta_degree = beta_degree
      self.idxs = idxs
      self.weights = np.full(m, 1.) if self.d == '2d' else np.full((k, m), 1.)

      # Data standardization
      if N == 1:
          print('Wrong data size; please do something about it.')
          return
      else:
          me = np.mean(X, axis=0)  # Grand mean
          range_ = np.max(X, axis=0) - np.min(X, axis=0)
          zz = []
          for i in range(m):
              if range_[i] == 0:
                  zz.append(i)
          if zz:
              print(f'{zz} features are constant -- should be removed.')
              return
          maver = np.repeat(me[np.newaxis, :], N, axis=0)
          mrange = np.repeat(range_[np.newaxis, :], N, axis=0)
          Y = (X - maver) / mrange  # Standardized data

      # K-means iterations
      flag = 0
      dd = np.sum(Y * Y)  # Standardized data scatter


      # Declaring a cluster membership label vector
      membership = np.zeros(N, dtype=int)
      # Clusters initialization:
      if hyp_clusters is None:
          # Choosing random centers
          p = np.random.permutation(N)
          inc = p[:k]
          cent = Y[inc, :]  # k x m matrix of initial cluster centers (standardized)
          membership[inc] = np.arange(k)
      else:
          cent = np.array([np.mean(Y[list(cluster)], axis=0) for cluster in hyp_clusters])

      # Iterations of updates of clusters and centroids
      while flag == 0:
          labelc, wc = self.clusterupdate(Y, cent)
          if np.array_equal(labelc, membership):
              flag = 1
              w = wc
          else:
              cent = self.ceupdate(Y, labelc)
              membership = labelc

      # Packing the output data
      wd = w * 100 / dd  # Unexplained data variance, percentage
      self.clusters = {
          'Clusters': membership,
          'wd': wd
      }

      
      return [set(np.where(labelc == k_id)[0]) for k_id in range(k)]
  



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

  method = KM(weights_func, opt.d)
  res = method.fit(dataset, idxs, int(opt.K), max_repetitions=int(opt.niter), hyp_clusters=hyp_clusters, beta_degree=float(opt.degree))
  for cl_id, cluster in enumerate(res):
    print(f"cluster [{cl_id}]: ", *cluster)

  print('Unexplained data variance, percentage: {}%'.format(round(method.clusters['wd'], 2)))

  if real_clusters is not None:
    r_labels = np.full(dataset.shape[0], 0)
    for i in range(len(real_clusters)):
      r_labels[list(real_clusters[i])] = i

    alg_labels = np.full(dataset.shape[0], 0)
    for i in range(len(res)):
      alg_labels[list(res[i])] = i

    print("ARI: ", round(adjusted_rand_score(r_labels, alg_labels), 2),
          "\tNMI: ", round(normalized_mutual_info_score(r_labels, alg_labels), 2))
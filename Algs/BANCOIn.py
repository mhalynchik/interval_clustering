import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np


class BANCOIn():
  def __init__(self, 
               weight_func, 
               d = '3d', 
               **kwargs):

    self.w_update_f = weight_func
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
      return cl_data
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

    return len_S * np.inner(w * c, c) -\
          2 * len_S * np.inner(w * c, obj) +\
          np.inner(w * obj, obj) > er

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

    return len_S * np.inner(w * c, c) -\
          2 * len_S * np.inner(w * c, obj) -\
          np.inner(w * obj, obj) < -er


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


    # 1. initialize
    # choose as the centroid the object farthest from zero
    c_idx = I_k[np.argmax([np.inner(data[i], data[i]) for i in I_k])]
    clusters.append({c_idx})
    centroids = np.vstack((centroids, data[c_idx]))
    #initialize weights
    w = np.full(self.V, 1., dtype=np.float32)

    while True:
      # 2. Anamalous cluster update
      old_c = centroids[-1]
      for i in I_k:
        w_b = np.power(w, self.beta_degree, out=np.zeros_like(w), where=w!=0)
        # Cluster Update Rule
        if i in clusters[-1] and self.CUR['remove'](data[i], len(clusters[-1]), centroids[-1], w_b, er):
          clusters[-1].remove(i)
        elif not (i in clusters[-1]) and self.CUR['add'](data[i], len(clusters[-1]), centroids[-1], w_b, er):
          clusters[-1].add(i)
        else:
          continue
        # 3. Anomalous center update
        new_c = self.cluster_center(data[list(clusters[-1])])
        if not np.allclose(new_c, centroids[-1]):
          centroids[-1] = new_c
          # 3.1. weights update
          w = self.w_update_f(data, clusters, b=self.beta_degree, idxs=self.idxs, d='3d')[-1]
          # go to step 2
          continue
      # 4. Stop criteria
      if np.allclose(centroids[-1], old_c):
        return clusters, centroids


  def fit(self,
          data:             np.ndarray,                                           # data
          idxs:             list,                                                 # ids of lower and upper bounds in data
          K:                int,                                                  # number of clusters
          max_repetitions:  int   = 1000,                                         # max algorithm iterations
          hyp_clusters:     list  = None,                                         # hypothetical clusters to initialize
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
    S = sorted(clusters, key=lambda x: len(x), reverse=True)[:K]
    # 4. Output K anamalous clusters
    return S

if __name__ == "__main__":
  from weights import A0, A1, A2
  
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

  method = BANCOIn(weights_func, opt.d)
  res = method.fit(dataset, idxs, int(opt.K), max_repetitions=int(opt.niter), hyp_clusters=hyp_clusters, beta_degree=float(opt.degree), er=float(opt.error))
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
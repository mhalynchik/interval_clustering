import argparse
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Algs.optimization import (preprocess, validate_options, validate_dimension,
                               validate_init, default_feature_groups)
from Algs.optimization import validate_input, initial_partition, lloyd
from Algs.partitions import read_clusters, as_clusters, labels_from_clusters, centers_for_labels, assign_nonempty


def _sample_kmeanspp_indices(Y: np.ndarray, k: int) -> np.ndarray:
    n_rows = Y.shape[0]
    chosen = np.empty(k, dtype=np.int64)
    chosen[0] = int(np.random.randint(n_rows))
    closest_sq = np.sum((Y - Y[chosen[0]]) ** 2, axis=1)
    for step in range(1, k):
        mass = closest_sq.copy()
        mass[chosen[:step]] = 0.0
        total = float(mass.sum())
        if total == 0.0:
            unused = np.flatnonzero(~np.isin(np.arange(n_rows), chosen[:step]))
            chosen[step] = int(unused[np.random.randint(len(unused))])
        else:
            chosen[step] = int(np.random.choice(n_rows, p=mass / total))
        new_sq = np.sum((Y - Y[chosen[step]]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return chosen


class KM():
  def __init__(self,
               weight_func,
               d = '3d',
               **kwargs):

    self.w_update_f = weight_func
    validate_dimension(d)
    self.d = d
    self.init = kwargs.get("init", "kmeans++")
    validate_init(self.init)
    self.verbose = kwargs.setdefault('verbose', False)
    self.FLAG_CENTER_DATA = kwargs.get('center_data', True)
    self.FLAG_MINMAX_NORAMILIZE_DATA = kwargs.get('minmax_normilize', True)
    self.FLAG_SCALE_BY_VAR = kwargs.get('scale_by_var', False)

  def normalize_data(self, data):
    return preprocess(data, self.FLAG_CENTER_DATA,
                      self.FLAG_MINMAX_NORAMILIZE_DATA, self.FLAG_SCALE_BY_VAR)

  def clusterupdate(self,Y,cent):
    from Algs.optimization import squared_distances
    distances=squared_distances(Y,cent,self.weights,self.beta_degree)
    labels=assign_nonempty(distances)
    return labels,float(np.sum((Y-cent[labels])**2))

  def ceupdate(self,X,labelc):
    k=getattr(self,'K',int(np.max(labelc))+1)
    self.weights=self.w_update_f(X,as_clusters(labelc,k),b=self.beta_degree,idxs=self.idxs,d=self.d)
    return centers_for_labels(X,labelc,k)

  def fit(self, X, idxs, k, max_repetitions=100000, hyp_clusters=None, beta_degree=1.):
    x=validate_input(X,k,max_repetitions)
    idxs=validate_options(idxs,x.shape[1],beta_degree,self.d)
    self.K=k;self.idxs=idxs;self.beta_degree=beta_degree
    x=self.normalize_data(x)
    labels=initial_partition(x,k,self.init,hyp_clusters,_sample_kmeanspp_indices)
    result=lloyd(x,labels,k,self.w_update_f,idxs,beta_degree,self.d,max_repetitions)
    self.weights=result['weights'];self.centers_=result['centers']
    self.converged_=result['converged'];self.n_iter_=result['iterations']
    self.stop_reason_=result['reason'];self.clusters={'Clusters':result['labels'],'wd':result['wd']}
    return as_clusters(result['labels'],k)


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
  parser.add_argument('--d', default='3d', choices=['2d', '3d'], required=False, help='way to calculate weights: 2d | 3d')
  parser.add_argument('--K', type=int, required=True, help='number of clusters')
  parser.add_argument('--init', choices=['random', 'kmeans++', 'ward'], default='kmeans++')
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

  idxs = default_feature_groups(dataset.shape[1], opt.weightsfunc, interval=False)

  method = KM(weights_func, opt.d, init=opt.init)
  res = method.fit(dataset, idxs, int(opt.K), max_repetitions=int(opt.niter), hyp_clusters=hyp_clusters, beta_degree=float(opt.degree))
  if res is None:
    raise ValueError("Clustering failed; check dataset size and constant features")
  for cl_id, cluster in enumerate(res):
    print(f"cluster [{cl_id}]: ", *cluster)

  print('Unexplained data variance, percentage: {}%'.format(round(method.clusters['wd'], 2)))

  if real_clusters is not None:
    r_labels = labels_from_clusters(real_clusters,dataset.shape[0])
    alg_labels = labels_from_clusters(res,dataset.shape[0])

    print("ARI: ", round(adjusted_rand_score(r_labels, alg_labels), 2),
          "\tNMI: ", round(normalized_mutual_info_score(r_labels, alg_labels, average_method='max'), 2))

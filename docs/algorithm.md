# Algorithm and implementation conventions

## Representations and weights

An interval matrix has shape `(N, 2V)`: all lower endpoints are followed by all
upper endpoints. Each column is centered and divided by its half-range. A
half-range below `1e-12` is replaced by 1. Columns are transformed independently;
endpoint order need not be preserved after preprocessing, and the paired
coordinates are not reordered.

| Manuscript notation | Function | Constraint |
| --- | --- | --- |
| n: no weighting | `A0` | All coefficients equal 1 |
| c: conventional weights | `A1` | Weights sum to 1 within each feature group |
| o: product-normalized weights | `A2` | Product of weights equals 1 |
| cs / os | `A1` / `A2` with `3d` mode | Separate weights for each cluster |

Let `SSE[k,j]` be the sum of squared deviations from cluster `k`'s mean in
coordinate `j`. For feature group `g`, `rho[g]` is the mean population variance
of its columns over the full preprocessed dataset, bounded below by `1e-12`.
The correction is fixed across partitions and keeps the dispersions positive.

IKM and KM use `D[k,j] = SSE[k,j] + rho[group(j)]` for cluster-specific weights,
or `D[j] = sum_k SSE[k,j] + rho[group(j)]` for global weights. BANCO instead
uses within-cluster variances, `D[k,j] = SSE[k,j]/n[k] + rho[group(j)]`.

- For conventional weights and `beta > 1`, weights are proportional to
  `D**(-1/(beta-1))` and normalized within each group. IKM and BANCO have separate
  lower- and upper-endpoint groups; doubled-coordinate KM has one group.
- For product-normalized weights, `w[j] = (geometric_mean(D)/D[j])**(1/beta)`.
  The product constraint spans both endpoint groups. Distances multiply by
  `w**beta`, so these distance coefficients do not depend on beta.
- The API also supports the conventional-weight boundary case `beta=1`, with
  equal weights on minimum-dispersion coordinates and zeros elsewhere. This
  case is excluded from the manuscript's conventional-weight grid.

For Lloyd optimization the implemented regularized objective is
`sum(D * w**beta)`. With fixed weights, the added regularization term does not
vary with assignments or centers. The supported exponent domain is
`beta=1` or `1.1 <= beta <= 5`.

## Initialization

- Uniform random initialization samples observations as initial centers.
- k-means++ draws the first center uniformly, then draws unused observations
  with probability proportional to squared distance from the nearest selected
  center. This is the single-candidate D-squared procedure.
- Ward merges the pair with the smallest increase in unweighted squared error
  on the preprocessed coordinates, until exactly K groups remain.
- BANCO extracts anomalous clusters and retains up to the K largest. Their
  centers initialize IKM or KM. The retained groups may leave some observations
  unassigned; the subsequent IKM/KM supplies the complete partition.

Initialization is followed by a nearest-center assignment with unit weights,
then updates of centers and weights. Equal-distance ties preserve existing
membership. An empty cluster receives the farthest observation from a donor
cluster containing at least two observations. This repair is part of the
implementation and is not covered by a general monotonicity guarantee.

## BANCO updates and termination

Within an extracted cluster, BANCO processes observations sequentially. The
membership test increases `n * sum(w**beta * c**2)` with the current weights
fixed. An accepted addition or removal is followed by recomputation of the
center and then the weights. Weights are also updated at the end of a pass.
The last observation cannot be removed from a cluster.

The fixed-weight membership calculation does not establish monotonicity of
the combined adaptive membership/weight procedure. Repeated states and the
iteration limit are therefore detected explicitly. Every extraction must
converge for the BANCO result to be eligible in the experiment.

IKM/KM report a fixed point, a repeated state, or an iteration limit. Only
converged results with K nonempty clusters are eligible for the reported
comparisons. All attempted candidates and starts, including failures, remain
in the JSON records.

## Python API

Use `Algs.BANCOIn.BANCOIn`, `Algs.IKM.IKM` and `Algs.KM.KM`. The
`experiments/revision_run.py` functions provide complete examples of arguments,
seeding, preprocessing and scoring.

The retained compatibility interfaces have the following conventions:

- `BANCOIn.fit(..., hyp_clusters=None)` requires `hyp_clusters=None`; BANCO
  always discovers its own anomalous groups. Its output contains up to K
  retained groups, not necessarily a complete partition.
- IKM/KM accept supplied groups for initialization. If fewer than K centers
  are supplied, the API can complete them randomly; such hybrid starts are
  recorded and excluded from the pure BANCO comparison.
- IKM distance helpers return scalar squared distances.
- The individual IKM/KM script argument `--error` is retained by the historical
  interface but does not control stopping. Use the experiment module interface
  and the returned `converged_`, `stop_reason_` and `n_iter_` attributes.

The calculation modules are preserved byte-for-byte as recorded in the
experiment manifest. Updated presentation commands are separate modules,
so checking the recorded source hashes remains possible.

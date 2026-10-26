# Anomalous Euclidean k-means clustering for interval data

Code and recorded experiments accompanying the revised manuscript by B. Mirkin,
M. Halynchyk, A. Parinov and Francisco de A. T. de Carvalho.

The implementation compares BANCO, Ward, uniform random and k-means++
initialization, followed by weighted interval k-means (IKM) or k-means on
paired endpoint coordinates (KM). Experiments use Fungi, Brazilian Scientific
Production (BSP), and interval-valued Iris data.

## Installation

Use Python 3.12 and the pinned dependencies:

```sh
python3.12 -m venv .venv-paper
. .venv-paper/bin/activate
python -m pip install -r requirements.txt
```

On NixOS, use the optional local environment described in
[docs/reproducibility.md](docs/reproducibility.md#nixos). `shell.nix` and virtual
environments are excluded from version control.

## Inspect the reported results

The seven JSON files in `experiments/results/ward_final/` contain every
individual random and k-means++ run, BANCO candidates, selected exponents,
reference labels, cluster assignments and convergence diagnostics. The
manifest records software versions and SHA-256 checksums of the calculation
code, data and result files. These records are the numerical source of the
revised manuscript tables.

```sh
python -m experiments.revision_verify --results experiments/results/ward_final --results-only
python -m experiments.manuscript_tables --results experiments/results/ward_final --out letter/manuscript_tables
```

The table exporter writes five LaTeX tables. Random and k-means++ entries report
**maximum / standard deviation / number of starts attaining the maximum**.
Unavailable results are shown as dashes. The exponent table reports the modal
selected value, with `n/a` for methods independent of the exponent.

Inspection of saved results does not rerun clustering. Verification checks
saved assignments, scores, parameter selection and source-data hashes.

## Reproduce the experiments

Run commands from the repository root. All seven input datasets are included.
Use a new output directory for each run:

```sh
python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.revision_run --workers 7 --out experiments/results/paper_rerun
python -m experiments.revision_verify --results experiments/results/paper_rerun --results-only
```

The full protocol uses 100 starts per stochastic initializer and 50 random
subsets of approximately 20% of the objects for exponent selection. `--workers`
parallelizes datasets. Existing completed result packages are not overwritten.

A short installation check uses a reduced grid and three starts per stochastic
initializer. Its output is not a manuscript result:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.revision_run --quick --datasets Fungi_3 --workers 1 --out experiments/results/smoke
```

## Data

Fungi and BSP inputs are distributed in `subdatasets/Fungi_genera_{3,4,5}` and
`subdatasets/BSP_GAP_{3,4,5}`. The interval Iris input in
`experiments/data/new/Iris/` has **30 interval-valued observations**, rather
than the 150 individual observations in point-valued Iris. Formats and
reference partitions are described in
[docs/reproducibility.md](docs/reproducibility.md#data-files).

All experiments center each endpoint column and divide it by its half-range.
The number of clusters is supplied by the reference partition. Reference
categories are used for parameter selection and evaluation; they are not
assumed to be the unique intrinsic cluster structure.

## Repository contents

| Path | Purpose |
| --- | --- |
| `Algs/` | BANCO, IKM, KM, weighting and initialization implementations |
| `experiments/revision_run.py` | Experiment protocol used for the reported results |
| `experiments/revision_verify.py` | Verification of saved runs and source provenance |
| `experiments/manuscript_tables.py` | Export of the revised manuscript tables |
| `experiments/results/ward_final/` | Original recorded runs and their manifest |
| `subdatasets/` | Fungi and BSP inputs and reference partitions |
| `experiments/data/new/Iris/` | Interval-valued Iris inputs and reference partition |
| `tests/` | Numerical, API and experiment-integrity tests |
| `docs/` | Algorithm conventions and reproduction details |
| `legacy/` | Superseded notebooks and tables from the original workflow |

See [docs/algorithm.md](docs/algorithm.md) for weight conventions, initialization,
termination and API details, and [docs/reproducibility.md](docs/reproducibility.md)
for parameter selection, metrics and result provenance. The experiment module
commands above are the supported reproduction interface; the original
notebooks and individual algorithm script interfaces are retained for historical
compatibility.

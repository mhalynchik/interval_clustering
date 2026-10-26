# Reproducing the revised manuscript experiments

## Data files

All seven manuscript datasets are included. Fungi and BSP use `data.npy` with
all lower endpoints followed by all upper endpoints. `data.csv` provides a
human-readable copy. `real_clusters.txt` contains one reference category per
line, expressed as comma-separated, zero-based observation indices.

Interval Iris is in `experiments/data/new/Iris/`. Its `data.csv` has `index` and
`label` columns followed by interval features written as quoted `[lower, upper]`
pairs. CSV row order defines observation indices. The loader reads this CSV and
`real_clusters.txt`. The supplied `data.npy` and `reference_clusters.txt` are
also retained because their checksums belong to the original experiment record.
This interval representation has 30 observations and is distinct from ordinary
point-valued Iris with 150 observations.

Fungi reference categories are genera; BSP reference categories are scientific
fields. The three versions of each dataset use three, four or five categories.
Reference partitions specify K and provide labels for partial-label parameter
selection and for the reported recovery measures.

## Experimental protocol

1. Center every endpoint column and divide by its half-range. Values below
   `1e-12` in the half-range denominator are replaced by 1. No division by the
   standard deviation or subsequent endpoint reordering is performed.
2. For pipelines containing conventional weights, test beta in
   `{1 + 0.1*j : j = 1, ..., 40}`. Pipelines containing only unit or
   product-normalized weights need one setting; beta=1 is their computational
   representative because their distance coefficients are beta-independent.
3. Fit each BANCO plus IKM/KM candidate on the full matrix. Exclude candidates
   unless every BANCO extraction and the subsequent optimizer converge and K
   nonempty clusters are obtained without random center completion.
4. Draw 50 subsets, each containing `max(1, round(0.2*N))` observations, uniformly
   without replacement within each draw, with selection seed 42. For each
   subset, choose the eligible beta with highest cluster purity: the fraction
   of sampled objects matching the most frequent reference label in their
   predicted cluster. Equal subset scores choose the smaller beta.
5. Use the most frequently selected beta. Tied modal counts choose the beta
   first selected in the fixed draw sequence. Hold this beta fixed across
   BANCO, Ward, random and k-means++ initialization for that method/dataset.
6. Run 100 uniform random starts and 100 k-means++ starts, with seeds 42 through
   141. Ward and BANCO each contribute one result. The iteration limit is 1000.

The complete package contains 11,200 stochastic starts across the seven datasets,
in addition to the BANCO candidate grid and Ward runs. Full-data ARI and NMI
include the objects whose labels were used in selection. They are descriptive
recovery scores, not estimates on an independent test set. The 50 subsets
can overlap. Per-draw held-out scores are recorded as diagnostics and do not
select beta.

## Metrics and tables

ARI is the adjusted Rand index. NMI is mutual information divided by the larger
of the two partition entropies (`average_method="max"` in scikit-learn).

For random and k-means++ starts, manuscript tables report the maximum, population
standard deviation (`ddof=0`), and count within absolute tolerance `1e-12` of
the maximum, with relative tolerance zero. Counts use unrounded values. Raw
records also retain the mean; its presence in JSON does not change the table
statistic. No failed or wrong-K start is silently removed before aggregation.

The beta table shows the modal beta without its frequency. A dash denotes an
unavailable result; `n/a` denotes an exponent-independent method. Frequencies,
excluded candidates, convergence information and every individual run remain
available in JSON.

`python -m experiments.manuscript_tables` exports the current presentation.
The original `revision_tables` module is retained as part of the recorded code
snapshot. It exports diagnostic and historical comparison material, including
its original maximum/mean layout, and is not the manuscript-table command.
The legacy comparison inputs and correspondence files are not required for
reproduction.

## Provenance and validation

`experiments/results/ward_final/manifest.json` records Python 3.12.12,
NumPy 2.2.5, SciPy 1.15.3 and scikit-learn 1.6.1, the protocol, and SHA-256
checksums of all calculation source files, input files and result JSON files.
The recorded package is kept unchanged. New export code is separate from
those calculation sources.

The verifier recomputes ARI/NMI from saved assignments, checks reference labels
against the input partitions, reconstructs the random label subsets and beta
selection, checks convergence and exact K, and validates aggregate statistics,
run seeds, file inventories and checksums. It does not rerun clustering.

Rerun the experiment to reproduce assignments themselves. Runtime measurements
and package identifiers vary between executions. A rerun records its own
source and input checksums; comparing numerical results is separate from
comparing entire JSON files byte-for-byte. Use the pinned software versions
and one BLAS/OpenMP thread for the closest reproducibility across machines.

All generated TeX, figures, PDFs, correspondence packages, local environments,
and additional experiment directories are ignored. Only the seven manuscript
result files and their manifest are included from the results directory.

## NixOS

Create a local `shell.nix` containing:

```nix
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = [ pkgs.python312 pkgs.python312Packages.pip ];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
  shellHook = ''
    test -d .venv-paper || python -m venv .venv-paper
    source .venv-paper/bin/activate
    python -m pip install -r requirements.txt
  '';
}
```

Run `nix-shell` from the repository root, then use the Python commands in the
README. `shell.nix`, `.venv-paper/`, `.venv/` and `venv/` remain ignored. The
Python dependencies are pinned in the distributed `requirements.txt`.

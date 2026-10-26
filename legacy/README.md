# Historical experiments

These notebooks and CSV tables belong to the original experiment workflow.
They are retained for provenance and are not the source of the revised
manuscript results. Use `experiments/revision_run.py` for the current protocol
and `experiments/results/ward_final/` for the reported individual runs.

The notebooks retain their original code and assume paths relative to the
repository root. Saved notebook output has been removed to avoid displaying
obsolete results as current results. They use pandas in addition to the
requirements of the supported experiment runner and are not part of its
automated reproduction workflow.

- `test.ipynb`: original parameter sweep and experiment driver.
- `dataset_analysis.ipynb`: original result aggregation and table generation.
- `generate_subdatasets.ipynb`: original Fungi and BSP data preparation.
- `*.csv`: original aggregate results and selected parameter tables.

The prepared datasets remain in `subdatasets/`; rerunning data preparation is
not necessary to reproduce the revised experiments.

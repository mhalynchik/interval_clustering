"""Diagnostic: historical beta settings applied to the CURRENT implementation.

This does not reproduce the historical algorithm or its published tables.
Its immutable packages have a separate kind and cannot enter manuscript tables.
"""
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import platform
import numpy as np
import scipy
import sklearn
from experiments.revision_run import run_optimizer, summarize
from experiments import revision_data as d
from experiments import revision_package as package


def one(c):
    folder = {short: f for f, short in d.PAPER_DATASETS}[c]
    x, y = d.load_dataset(d.dataset_folder(folder))
    overrides = {('Fungi_3', 'KMc'): 1.8, ('Fungi_3', 'KMcs'): 1.8,
                 ('Fungi_4', 'IKMo'): 1.5, ('Fungi_4', 'IKMos'): 1.5,
                 ('Fungi_4', 'KMc'): 1.8, ('Fungi_4', 'KMcs'): 1.8}
    result = {}
    for spec in d.MINUS_SPECS:
        beta = overrides.get((c, spec[0]), 1.)
        runs = [run_optimizer(x, y, spec, beta, 42+i) for i in range(100)]
        result[spec[0]] = dict(beta=beta, runs=runs, summary=summarize(runs))
    return c, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--out', type=Path, default=d.ROOT/'experiments/results/previous_beta_control_current')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    code = [*sorted((d.ROOT/'Algs').glob('*.py')), *sorted((d.ROOT/'experiments').glob('revision*.py'))]
    metadata = dict(kind='historical_beta_diagnostic', python=platform.python_version(),
                    numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                    nmi_average_method='max', datasets=d.PAPER_COLS, seeds=list(range(42,142)),
                    code_sha256={str(p.relative_to(d.ROOT)): package.sha256(p) for p in code},
                    data_sha256={str(p.relative_to(d.ROOT)): package.sha256(p)
                                 for f,_ in d.PAPER_DATASETS for p in sorted(d.dataset_folder(f).iterdir()) if p.is_file()})
    with package.staging(args.out, metadata) as stage:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for c, result in executor.map(one, d.PAPER_COLS):
                package.write_json(stage/f'{c}.json', result)
        package.verify_sources(metadata, d.ROOT)
        package.seal(stage)


if __name__ == '__main__':
    main()

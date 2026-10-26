"""Reproduce Ward/random/kmeans++ comparisons with partial-label beta selection.

Run from repository root: python -m experiments.revision_run --workers 7
No full-label metric is used to select beta or initialization runs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import platform
import time
import numpy as np
import scipy
import sklearn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from Algs.BANCOIn import BANCOIn
from Algs.IKM import IKM
from Algs.KM import KM
from experiments import revision_data as d
from experiments import revision_package as package

GRID = np.round(np.arange(1.1, 5.01, .1), 1).tolist()


def partition_labels(clusters, n):
    return d.labels_from_clusters(clusters, n)


def purity(y, pred):
    """Huang recovery accuracy: majority reference label within each cluster."""
    return sum(max(Counter(y[pred == j]).values()) for j in np.unique(pred)) / len(y)


def metrics(y, pred):
    return dict(ari=float(adjusted_rand_score(y, pred)),
                nmi=float(normalized_mutual_info_score(y, pred, average_method='max')))


def select_beta(rows, y, draws=50) -> dict[str, Any]:
    rows = list(rows)
    betas = [r['beta'] for r in rows]
    if not rows or any(isinstance(b, (bool, np.bool_)) or not isinstance(b, (int, float, np.integer, np.floating))
                       or not np.isfinite(b) or b < 1 for b in betas):
        raise ValueError('Beta candidates must be nonempty, numeric, finite and >=1')
    if len(set(betas)) != len(betas):
        raise ValueError('Duplicate beta candidates')
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError('Draw count must be a positive integer')
    y = np.asarray(y)
    if y.ndim != 1 or len(y) < 2:
        raise ValueError('Selection needs at least two reference labels')
    rows.sort(key=lambda r: r['beta'])
    rng = np.random.default_rng(42)
    records = []
    for draw in range(draws):
        sample = np.sort(rng.choice(len(y), max(1, round(.2 * len(y))), replace=False))
        holdout = np.setdiff1d(np.arange(len(y)), sample)
        scores = [purity(y[sample], np.asarray(r['labels'])[sample]) for r in rows]
        # Ascending grid: exact ties choose smaller beta, as in previous code.
        best = rows[int(np.argmax(scores))]
        records.append(dict(draw=draw, beta=best['beta'], sample=sample.tolist(),
                            sample_accuracy=max(scores),
                            heldout=metrics(y[holdout], np.asarray(best['labels'])[holdout])))
    counts = Counter(r['beta'] for r in records)
    # Historical tie rule: first selected beta among equally frequent modes.
    beta = counts.most_common(1)[0][0]
    chosen = next(r for r in rows if r['beta'] == beta)
    return dict(beta=beta, frequency=counts[beta], frequencies=dict(counts),
                draws=records, selected=chosen)


def run_optimizer(x, y, spec, beta, seed, init='kmeans++', hyp=None) -> dict[str, Any]:
    name, kind, weight, dim = spec
    d.seed_rng(seed)
    model = (IKM if kind == 'IKM' else KM)(d.WEIGHT_FUNCS[weight], dim,
                                          init=init, **d.PREPROCESS)
    t = time.perf_counter()
    clusters = model.fit(x.copy(), d.idxs_for_weights(weight, x.shape[1], kind),
                         len(np.unique(y)), hyp_clusters=hyp, beta_degree=beta,
                         max_repetitions=1000)
    pred = partition_labels(clusters, len(y))
    if not np.isfinite(model.weights).all():
        raise ValueError(f'Non-finite weights: {name}, {beta}')
    metadata = {} if hyp is None else dict(banco_retained_count=len(hyp),
        random_completion_centers=max(0, len(np.unique(y))-len(hyp)))
    return dict(name=name, beta=beta, seed=seed, init=('BANCO' if hyp is not None else init), **metadata,
                labels=pred.tolist(), clusters=len(np.unique(pred)),
                converged=model.converged_, stop_reason=model.stop_reason_, iterations=model.n_iter_,
                seconds=time.perf_counter()-t, **metrics(y, pred))


def availability(available, reason):
    return dict(status='available' if available else 'unavailable',
                reason=None if available else reason)


def eligible(row, k):
    return (row['converged'] and row['clusters'] == k and
            row.get('banco_converged', True) and row.get('banco_retained_count', k) == k)


def candidate_selection(rows, y):
    """Keep every diagnostic, but select only converged full-K pipelines."""
    valid = [r for r in rows if eligible(r, len(np.unique(y)))]
    selection: dict[str, Any] = (select_beta(valid, y) if valid else
                 dict(selected=None, beta=None, frequency=0, frequencies={}, draws=[]))
    selection.update(availability(bool(valid), 'no_eligible_candidate'), grid=rows,
                     excluded_betas=[r['beta'] for r in rows if r not in valid],
                     legacy_unfiltered=select_beta(rows, y))
    return selection


def summarize(rows, k=None, reason='nonconverged_or_wrong_k') -> dict[str, Any]:
    failed = sum(not r['converged'] if k is None else not eligible(r, k) for r in rows)
    result: dict[str, Any] = dict(n=len(rows), nonconverged=sum(not r['converged'] for r in rows),
                                  failed=failed, **availability(bool(rows) and failed == 0, reason))
    if result['status'] == 'unavailable':
        return result
    for m in ['ari', 'nmi']:
        vals = np.asarray([r[m] for r in rows])
        result[m] = dict(max=float(vals.max()), mean=float(vals.mean()),
                         std=float(vals.std()), hits=int(np.isclose(vals, vals.max(), rtol=0, atol=1e-12).sum()))
    return result


def dataset_run(folder, short, out, quick):
    x, y = d.load_dataset(d.dataset_folder(folder))
    k = len(np.unique(y))
    grid = [1.1, 1.5, 2.] if quick else GRID
    cache = {}
    plus: dict[str, Any] = {}
    path = Path(out)/f'{short}.json'
    result: dict[str, Any] = dict(dataset=short, folder=folder, n=len(x), features=x.shape[1], k=k,
                  reference_labels=y.tolist(), plus=plus, methods={})
    package.write_json(path, result)
    for name, bw, kind, kw, dim in d.PLUS_ALL:
        candidates = grid if 'A1' in (bw, kw) else [1.]
        rows = []
        for beta in candidates:
            key = (bw, beta if bw == 'A1' else 1.)
            if key not in cache:
                d.seed_rng(42)
                banco = BANCOIn(d.WEIGHT_FUNCS[bw], '3d', **d.PREPROCESS)
                hyp = banco.fit(
                    x.copy(), d.idxs_for_weights(bw, x.shape[1], 'BANCO'), k, beta_degree=key[1], er=1e-10)
                cache[key] = (hyp, banco.converged_, banco.extraction_status_)
            row = run_optimizer(x, y, (name, kind, kw, dim), beta, 42, hyp=cache[key][0])
            row['banco_converged'] = cache[key][1]
            row['banco_extractions'] = cache[key][2]
            rows.append(row)
        plus[name] = candidate_selection(rows, y)
        package.write_json(path, result)
        print(short, name, plus[name]['status'], 'beta', plus[name]['beta'], flush=True)
    for spec in d.MINUS_SPECS:
        name, _, weight, _ = spec
        matching = plus['B'+name]
        beta = matching['beta']
        beta_source = 'banco_selected'
        if beta is None and weight in ('A0', 'A2'):
            beta, beta_source = 1., 'beta_independent'
        elif beta is None:
            beta_source = None
        method: dict[str, Any] = dict(beta=beta, beta_source=beta_source,
            **availability(beta is not None, 'no_eligible_banco_beta'))
        result['methods'][name] = method
        if beta is None:
            method['ward'] = None
            for init in ('random', 'kmeans++'):
                method[init] = dict(runs=[], summary=summarize([], k, 'no_eligible_banco_beta'))
            package.write_json(path, result)
            continue
        ward = run_optimizer(x, y, spec, beta, 42, init='ward')
        ward.update(availability(eligible(ward, k), 'nonconverged_or_wrong_k'))
        method['ward'] = ward
        package.write_json(path, result)
        for init in ['random', 'kmeans++']:
            runs = [run_optimizer(x, y, spec, beta, 42+i, init=init)
                    for i in range(3 if quick else 100)]
            method[init] = dict(runs=runs, summary=summarize(runs, k))
            package.write_json(path, result)
        print(short, name, 'Ward', ward['status'], flush=True)
    path = Path(out)/f'{short}.json'
    package.write_json(path, result)
    return short


def run_package(out, jobs, quick=False, workers=7):
    if not jobs or workers < 1:
        raise ValueError('At least one dataset and a positive worker count are required')
    code = [*sorted((d.ROOT/'Algs').glob('*.py')),
            *sorted((d.ROOT/'experiments').glob('revision*.py'))]
    hashes = {str(p.relative_to(d.ROOT)): package.sha256(p) for p in code}
    data_hashes = {str(p.relative_to(d.ROOT)): package.sha256(p)
                   for folder, _ in jobs for p in sorted(d.dataset_folder(folder).iterdir())
                   if p.is_file()}
    metadata = dict(kind='results', python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, sklearn=sklearn.__version__, code_sha256=hashes,
                    data_sha256=data_hashes, datasets=[short for _, short in jobs],
                    grid=[1.1, 1.5, 2.] if quick else GRID, quick=quick,
                    seeds=list(range(42, 45 if quick else 142)), draws=50,
                    sample_fraction=.2, selection_seed=42, optimizer_seed=42,
                    nmi_average_method='max', preprocess=d.PREPROCESS,
                    beta_independent_grid=[1.], completed_datasets=[], failures={})
    with package.staging(out, metadata) as stage:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(dataset_run, folder, short, str(stage), quick): short
                       for folder, short in jobs}
            failures = {}
            completed = []
            for future in as_completed(futures):
                short = futures[future]
                try:
                    future.result()
                    completed.append(short)
                    print('DONE', short, flush=True)
                except Exception as error:
                    failures[short] = f'{type(error).__name__}: {error}'
                manifest = package.read_manifest(stage)
                manifest.update(completed_datasets=sorted(completed), failures=failures)
                package.write_json(stage/'manifest.json', manifest)
        if failures:
            raise RuntimeError(f'Dataset failures: {failures}')
        package.verify_sources(metadata, d.ROOT)
        package.seal(stage)
        from experiments.revision_verify import verify_results
        verify_results(stage)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=7)
    p.add_argument('--datasets', nargs='+', choices=d.PAPER_COLS)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--out', type=Path, default=d.ROOT/'experiments/results/ward_final')
    a = p.parse_args(argv)
    jobs = [(f, s) for f, s in d.PAPER_DATASETS if a.datasets is None or s in a.datasets]
    run_package(a.out, jobs, a.quick, a.workers)


if __name__ == '__main__':
    main()

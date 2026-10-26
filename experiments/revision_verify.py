"""Recompute scores/selection and verify complete, immutable result/artifact packages."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import zipfile
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from experiments import revision_data as d
from experiments import revision_package as package


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context):
    require(np.isfinite(actual) and abs(actual-expected) < 1e-14, context)


def rescore(y, labels):
    return dict(ari=float(adjusted_rand_score(y, labels)),
                nmi=float(normalized_mutual_info_score(y, labels, average_method='max')))


def verify_row(row, y, name, beta, init, seed, k, require_convergence=False):
    context = f'{name}/{init}/{beta}/{seed}'
    fields = {'name', 'beta', 'init', 'seed', 'labels', 'clusters', 'converged',
              'stop_reason', 'iterations', 'seconds', 'ari', 'nmi'}
    if init == 'BANCO':
        fields |= {'banco_converged', 'banco_extractions', 'banco_retained_count', 'random_completion_centers'}
    elif init == 'ward':
        fields |= {'status', 'reason'}
    require(set(row) == fields, f'Run schema/fake availability: {context}')
    require((row['name'], row['beta'], row['init'], row['seed']) == (name, beta, init, seed),
            f'Run identity mismatch: {context}')
    labels = np.asarray(row['labels'])
    require(labels.shape == y.shape and labels.dtype.kind in 'iu' and np.all(labels >= 0),
            f'Invalid labels: {context}')
    require(len(np.unique(labels)) == row['clusters'], f'Cluster count mismatch: {context}')
    require(isinstance(row['converged'], bool) and row['iterations'] > 0, f'Invalid status: {context}')
    require(row['stop_reason'] == 'fixed_point' if row['converged'] else row['stop_reason'] in ('cycle', 'iteration_limit'),
            f'Stop reason mismatch: {context}')
    require(np.isfinite(row['seconds']) and row['seconds'] >= 0, f'Invalid duration: {context}')
    if init == 'BANCO':
        require(row['banco_extractions'] and isinstance(row['banco_converged'], bool), f'Missing BANCO status: {context}')
        retained = row['banco_retained_count']
        require(isinstance(retained, int) and not isinstance(retained, bool) and
                retained == min(k, len(row['banco_extractions'])) and retained > 0,
                f'BANCO retained count/extractions mismatch: {context}')
        require(row['random_completion_centers'] == max(0, k-retained),
                f'BANCO random completion mismatch: {context}')
        if require_convergence:
            require(retained == k and row['random_completion_centers'] == 0,
                    f'Selected BANCO result uses random completion: {context}')
        for extraction in row['banco_extractions']:
            require(isinstance(extraction['converged'], bool) and
                    isinstance(extraction['iterations'], int) and not isinstance(extraction['iterations'], bool)
                    and extraction['iterations'] > 0, f'Invalid BANCO extraction: {context}')
            require(extraction['reason'] == 'fixed_point' if extraction['converged'] else
                    extraction['reason'] in ('cycle', 'iteration_limit'), f'Invalid BANCO stop reason: {context}')
        require(row['banco_converged'] == all(t['converged'] for t in row['banco_extractions']),
                f'BANCO extraction status mismatch: {context}')
    if require_convergence:
        require(row['converged'] and row['clusters'] == k and row.get('banco_converged', True),
                f'Unconverged/wrong-K selected result: {context}')
    for metric, expected in rescore(y, labels).items():
        close(row[metric], expected, f'{metric} mismatch: {context}')


def verify_selection(selection, rows, y, manifest):
    require(bool(rows), 'Empty candidate set')
    rows = sorted(rows, key=lambda row: row['beta'])
    draws = selection['draws']
    require(len(draws) == manifest['draws'], 'Selection draw count mismatch')
    rng = np.random.default_rng(manifest['selection_seed'])
    size = max(1, round(manifest['sample_fraction']*len(y)))
    selected_betas = []
    for index, record in enumerate(draws):
        sample = np.sort(rng.choice(len(y), size, replace=False))
        require(record['draw'] == index and record['sample'] == sample.tolist(), 'Selection sample/seed mismatch')
        scores = []
        for row in rows:
            pred = np.asarray(row['labels'])[sample]
            scores.append(sum(max(Counter(y[sample][pred == cluster]).values())
                              for cluster in np.unique(pred)) / len(sample))
        best_index = int(np.argmax(scores))
        best = rows[best_index]
        require(record['beta'] == best['beta'], 'Draw winner/tie rule mismatch')
        close(record['sample_accuracy'], scores[best_index], 'Sample accuracy mismatch')
        heldout = np.setdiff1d(np.arange(len(y)), sample)
        for metric, value in rescore(y[heldout], np.asarray(best['labels'])[heldout]).items():
            close(record['heldout'][metric], value, 'Heldout score mismatch')
        selected_betas.append(best['beta'])
    frequencies = Counter(selected_betas)
    beta, frequency = frequencies.most_common(1)[0]
    require(selection['beta'] == beta and selection['frequency'] == frequency, 'Modal beta/frequency mismatch')
    require({float(k): v for k, v in selection['frequencies'].items()} == dict(frequencies), 'Frequency histogram mismatch')
    expected = next(r for r in rows if r['beta'] == beta)
    require(selection['selected'] == expected, 'Selected copy differs from its grid row')


def verify_availability(value, available, reason):
    require(value.get('status') == ('available' if available else 'unavailable') and
            'reason' in value and value['reason'] == (None if available else reason),
            'Availability status/reason mismatch')


def verify_summary(summary, rows, k, reason='nonconverged_or_wrong_k'):
    failed = sum(not r['converged'] or r['clusters'] != k for r in rows)
    available = bool(rows) and failed == 0
    verify_availability(summary, available, reason)
    require(set(summary) == {'n', 'nonconverged', 'failed', 'status', 'reason'} |
            ({'ari', 'nmi'} if available else set()), 'Summary schema/fake metrics')
    require(summary['n'] == len(rows) and summary['nonconverged'] == sum(not r['converged'] for r in rows)
            and summary['failed'] == failed, 'Summary counts mismatch')
    if not available:
        return
    for metric in ('ari', 'nmi'):
        vals = np.asarray([r[metric] for r in rows])
        for key, value in [('max', vals.max()), ('mean', vals.mean()), ('std', vals.std())]:
            close(summary[metric][key], value, 'Summary '+metric+'/'+key+' mismatch')
        require(summary[metric]['hits'] == int(np.isclose(vals, vals.max(), rtol=0, atol=1e-12).sum()),
                'Maximum hit count mismatch')


def verify_results(root, source_root=d.ROOT):
    root = Path(root)
    manifest = package.validate_package(root, 'results')
    package.verify_sources(manifest, source_root)
    cols = manifest['datasets']
    require(bool(cols) and cols == [c for c in d.PAPER_COLS if c in cols], 'Invalid dataset inventory/order')
    require(manifest['completed_datasets'] == sorted(cols) and manifest['failures'] == {}, 'Incomplete dataset execution')
    require(set(manifest['files_sha256']) == {c+'.json' for c in cols}, 'Mixed or missing result files')
    require(isinstance(manifest['quick'], bool), 'Invalid quick flag')
    require(manifest['grid'] == ([1.1, 1.5, 2.] if manifest['quick'] else np.round(np.arange(1.1, 5.01, .1), 1).tolist()), 'Incorrect executed grid')
    require(manifest['seeds'] == list(range(42, 45 if manifest['quick'] else 142)), 'Incorrect executed seeds')
    require(manifest['draws'] == 50 and manifest['sample_fraction'] == .2 and manifest['selection_seed'] == 42
            and manifest['optimizer_seed'] == 42, 'Invalid selection protocol')
    require(manifest['nmi_average_method'] == 'max' and manifest['preprocess'] == d.PREPROCESS
            and manifest['beta_independent_grid'] == [1.], 'Invalid metric/preprocessing protocol')
    results = {}
    for c in cols:
        result = json.loads((root/(c+'.json')).read_text())
        y = np.asarray(result['reference_labels'])
        require(y.ndim == 1 and y.dtype.kind in 'iu' and len(y) == result['n'] and len(y) >= 2 and len(np.unique(y)) == result['k'], 'Invalid reference labels')
        require(result['dataset'] == c and result['folder'] == dict((short, folder) for folder, short in d.PAPER_DATASETS)[c], 'Dataset identity mismatch')
        relative_folder = d.dataset_folder(result['folder']).relative_to(d.ROOT)
        source_folder = Path(source_root) / relative_folder
        matrix_file = 'data.csv' if result['folder'] in d.CSV_DATASETS else 'data.npy'
        require(all(str(relative_folder/name) in manifest['data_sha256']
                    for name in (matrix_file, 'real_clusters.txt')),
                'Loaded dataset files are missing from input provenance')
        source_x, source_y = d.load_dataset(source_folder)
        require(source_x.ndim == 2 and np.isfinite(source_x).all() and
                source_x.shape == (result['n'], result['features']),
                'Recorded dimensions differ from source data')
        require(np.array_equal(y, source_y),
                'Recorded reference labels differ from source partition')
        require(set(result['plus']) == {s[0] for s in d.PLUS_ALL} and set(result['methods']) == {s[0] for s in d.MINUS_SPECS}, 'Missing/extra methods')
        for name, bw, kind, kw, dim in d.PLUS_ALL:
            selection = result['plus'][name]
            grid = selection['grid']
            expected_grid = manifest['grid'] if 'A1' in (bw, kw) else [1.]
            require([r['beta'] for r in grid] == expected_grid, 'Candidate grid mismatch: '+name)
            for row in grid:
                verify_row(row, y, name, row['beta'], 'BANCO', 42, result['k'])
            valid = [r for r in grid if r['converged'] and r['clusters'] == result['k'] and r['banco_converged'] and r['banco_retained_count'] == result['k']]
            require(selection['excluded_betas'] == [r['beta'] for r in grid if r not in valid], 'Candidate exclusion mismatch')
            require(set(selection) == {'status', 'reason', 'beta', 'selected', 'frequency',
                    'frequencies', 'draws', 'grid', 'excluded_betas', 'legacy_unfiltered'},
                    'Selection schema/fake metrics')
            verify_availability(selection, bool(valid), 'no_eligible_candidate')
            if valid:
                verify_selection(selection, valid, y, manifest)
                verify_row(selection['selected'], y, name, selection['beta'], 'BANCO', 42, result['k'], True)
            else:
                require(selection['selected'] is None and selection['beta'] is None and
                        selection['frequency'] == 0 and selection['frequencies'] == {} and
                        selection['draws'] == [], 'Unavailable selection has fake metrics/selection')
            # Historical unfiltered selection is diagnostic only, never a main-table result.
            verify_selection(selection['legacy_unfiltered'], grid, y, manifest)
        for name, _, weight, _ in d.MINUS_SPECS:
            method = result['methods'][name]
            beta = result['plus']['B'+name]['beta']
            source = 'banco_selected'
            if beta is None:
                beta = 1. if weight in ('A0', 'A2') else None
                source = 'beta_independent' if beta is not None else None
            require(set(method) == {'beta', 'beta_source', 'status', 'reason', 'ward', 'random', 'kmeans++'},
                    'Method schema/fake metrics')
            require(method['beta'] == beta and method['beta_source'] == source, 'Method beta/source differs from protocol')
            verify_availability(method, beta is not None, 'no_eligible_banco_beta')
            if beta is None:
                require(method['ward'] is None, 'Unavailable comparison has a Ward result')
            else:
                ward = method['ward']
                verify_row(ward, y, name, beta, 'ward', 42, result['k'])
                verify_availability(ward, ward['converged'] and ward['clusters'] == result['k'],
                                    'nonconverged_or_wrong_k')
            for init in ('random', 'kmeans++'):
                require(set(method[init]) == {'runs', 'summary'}, 'Start schema/fake metrics')
                rows = method[init]['runs']
                require([r['seed'] for r in rows] == (manifest['seeds'] if beta is not None else []), 'Start count/seeds mismatch')
                for row in rows:
                    verify_row(row, y, name, beta, init, row['seed'], result['k'])
                verify_summary(method[init]['summary'], rows, result['k'],
                               'no_eligible_banco_beta' if beta is None else 'nonconverged_or_wrong_k')
        results[c] = result
    return manifest, results


def verify_artifacts(results_root, out, source_root=d.ROOT):
    from experiments.revision_tables import table_documents, csv_rows
    manifest, results = verify_results(results_root, source_root)
    out = Path(out)
    artifacts = package.validate_package(out, 'artifacts')
    require(artifacts['results_package_id'] == manifest['package_id'] and
            artifacts['results_manifest_sha256'] == package.sha256(Path(results_root)/'manifest.json'),
            'Artifacts belong to a different results manifest')
    require(artifacts['datasets'] == manifest['datasets'] and artifacts['quick'] == manifest['quick'], 'Artifact protocol mismatch')
    documents = table_documents(results, manifest)
    actual_tex = {str(p.relative_to(out)) for p in out.rglob('*.tex') if p.parts[len(out.parts)] != 'baseline'}
    require(actual_tex == set(documents), 'Unexpected/missing TeX file')
    for filename, expected in documents.items():
        require((out/filename).read_text() == expected, 'TeX cells/caption mismatch: '+filename)
    reported_count = 0
    for filename, diagnostics in [('individual_runs.csv', False), ('candidate_diagnostics.csv', True)]:
        expected_rows = csv_rows(results, manifest['datasets'], diagnostics=diagnostics)
        if not diagnostics:
            reported_count = len(expected_rows)
        expected_fields = list(dict.fromkeys(key for row in expected_rows for key in row)) or ['dataset', 'name', 'record_type']
        with (out/filename).open(newline='') as handle:
            reader = csv.DictReader(handle)
            actual_rows = list(reader)
            require(reader.fieldnames == expected_fields, 'CSV columns mismatch: '+filename)
        expected_rows = [{key: str(row[key]) if key in row and row[key] is not None else ''
                          for key in expected_fields} for row in expected_rows]
        require(actual_rows == expected_rows, 'CSV rows/values mismatch: '+filename)
    with zipfile.ZipFile(out/'manuscript_tables_final.zip') as archive:
        expected_names = sorted(str(p.relative_to(out)) for p in (out/'tables').iterdir() if p.is_file())
        require(sorted(archive.namelist()) == expected_names, 'Archive inventory mismatch')
        for name in expected_names:
            require(archive.read(name) == (out/name).read_bytes(), 'Archive content mismatch: '+name)
    return reported_count, len(documents)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=d.ROOT/'experiments/results/ward_final')
    parser.add_argument('--out', type=Path, default=d.ROOT/'letter/ward_final')
    parser.add_argument('--source-root', type=Path, default=d.ROOT)
    parser.add_argument('--results-only', action='store_true')
    args = parser.parse_args(argv)
    if args.results_only:
        manifest, _ = verify_results(args.results, args.source_root)
        print(f"PASS: {len(manifest['datasets'])} datasets; metrics, selection, convergence, seeds, provenance and hashes")
    else:
        rows, tex = verify_artifacts(args.results, args.out, args.source_root)
        print(f'PASS: results, selection, convergence, provenance and hashes; {rows} CSV rows; all {tex} TeX files and archive')


if __name__ == '__main__':
    main()

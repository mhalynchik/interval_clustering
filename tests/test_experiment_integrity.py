"""Tiny, isolated integrity fixtures; never execute or overwrite paper experiments."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import normalized_mutual_info_score
from experiments import revision_data as d
from experiments import revision_package as package
from experiments import revision_run as run
from experiments import revision_tables as tables
from experiments import revision_verify as verify


@pytest.fixture
def specs(monkeypatch):
    monkeypatch.setattr(d, 'PAPER_DATASETS', (('tiny', 'Tiny'),))
    monkeypatch.setattr(d, 'PAPER_COLS', ['Tiny'])
    monkeypatch.setattr(d, 'PLUS_SAME', (('BIKMn', 'A0', 'IKM', 'A0', '2d'),
                                        ('BIKMc', 'A1', 'IKM', 'A1', '2d')))
    monkeypatch.setattr(d, 'PLUS_MIXED', (('BnIKMc', 'A0', 'IKM', 'A1', '2d'),
                                         ('BoIKMn', 'A2', 'IKM', 'A0', '2d')))
    monkeypatch.setattr(d, 'PLUS_ALL', d.PLUS_SAME + d.PLUS_MIXED)
    monkeypatch.setattr(d, 'MINUS_SPECS', (('IKMn', 'IKM', 'A0', '2d'), ('IKMc', 'IKM', 'A1', '2d')))


def make_result(quick=True) -> dict[str, Any]:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    def row(name, beta, init, seed=42):
        return dict(name=name, beta=beta, init=init, seed=seed, labels=pred.tolist(), clusters=2,
                    converged=True, stop_reason='fixed_point', iterations=2, seconds=.01, **run.metrics(y, pred))
    result: dict[str, Any] = dict(dataset='Tiny', folder='tiny', reference_labels=y.tolist(), n=len(y), k=2, features=4, plus={}, methods={})
    for name, bw, _, kw, _ in d.PLUS_ALL:
        grid = ([1.1, 1.5, 2.] if quick else run.GRID) if 'A1' in (bw, kw) else [1.]
        rows = [dict(row(name, beta, 'BANCO'), banco_converged=True,
                     banco_retained_count=2, random_completion_centers=0,
                     banco_extractions=[dict(converged=True, reason='fixed_point', iterations=1) for _ in range(2)]) for beta in grid]
        # Failed candidates remain recorded but cannot win the main selection.
        if len(rows) > 1:
            rows[0].update(converged=False, stop_reason='cycle')
        value = run.candidate_selection(rows, y)
        result['plus'][name] = value
    for name, *_ in d.MINUS_SPECS:
        beta = result['plus']['B'+name]['beta']
        method: dict[str, Any] = dict(beta=beta, beta_source='banco_selected',
            **run.availability(True, ''), ward=dict(row(name, beta, 'ward'), **run.availability(True, '')))
        for init in ('random', 'kmeans++'):
            rows = [row(name, beta, init, seed) for seed in range(42, 45 if quick else 142)]
            method[init] = dict(runs=rows, summary=run.summarize(rows))
        result['methods'][name] = method
    return result


@pytest.fixture
def recorded(tmp_path, specs):
    sources = tmp_path/'sources'
    sources.mkdir()
    (sources/'algorithm.py').write_text('fixture source')
    (sources/'data.txt').write_text('fixture data')
    result = make_result()
    folder = sources/'subdatasets/tiny'
    folder.mkdir(parents=True)
    np.save(folder/'data.npy', np.zeros((result['n'], result['features'])))
    labels = np.asarray(result['reference_labels'])
    (folder/'real_clusters.txt').write_text('\n'.join(
        ', '.join(map(str, np.flatnonzero(labels == k))) for k in np.unique(labels)))
    metadata = dict(kind='results', datasets=['Tiny'], completed_datasets=['Tiny'], failures={}, quick=True,
                    grid=[1.1, 1.5, 2.], seeds=[42, 43, 44], draws=50, sample_fraction=.2,
                    selection_seed=42, optimizer_seed=42, nmi_average_method='max', preprocess=d.PREPROCESS,
                    beta_independent_grid=[1.], code_sha256={'algorithm.py': package.sha256(sources/'algorithm.py')},
                    data_sha256={str(p.relative_to(sources)): package.sha256(p)
                                 for p in [sources/'data.txt', folder/'data.npy', folder/'real_clusters.txt']})
    root = tmp_path/'results'
    with package.staging(root, metadata) as stage:
        package.write_json(stage/'Tiny.json', result)
        package.seal(stage)
    return root, sources, result


def rewrite_result(recorded, mutate):
    root, _, result = recorded
    mutate(result)
    package.write_json(root/'Tiny.json', result)
    package.seal(root)  # Simulate internally consistent hashes but scientifically incorrect content.


def test_max_nmi_and_signed_parser():
    y, labels = [0, 0, 1, 1, 1], [0, 0, 0, 1, 2]
    assert run.metrics(y, labels)['nmi'] == normalized_mutual_info_score(y, labels, average_method='max')
    assert run.metrics(y, labels)['nmi'] != normalized_mutual_info_score(y, labels, average_method='arithmetic')
    assert tables.values('-0.13 / +0.25') == [-.13, .25]
    assert tables.values('-1e-2 / .25') == [-.01, .25]
    assert tables.color('-.1 / .5', '-.2 / .6', 'ari') == '#d7eedb'
    assert tables.color('.9 / .9', '.1 / .1', 'none') == 'white'


def test_verified_quick_package_and_protocol(recorded):
    root, source, _ = recorded
    manifest, result = verify.verify_results(root, source)
    docs = tables.table_documents(result, manifest)
    text = '\n'.join(docs.values())
    assert '3 starts' in text and '100 starts' not in text
    assert 'Quick diagnostic' in text
    assert '1.1, 1.5, 2.0' in text
    assert 'not a fitted parameter' in text
    assert not any('sensitivity' in filename.lower() for filename in docs)
    assert 'population variance' in text and 'SSE' in text and 'rho' in text


@pytest.mark.parametrize('mutate', [
    lambda r: r['plus']['BIKMc']['selected'].update(nmi=.123),
    lambda r: r['plus']['BIKMc']['draws'][0].update(beta=2.),
    lambda r: r['plus']['BIKMc']['draws'][0].update(sample=[0, 0]),
    lambda r: r['plus']['BIKMc']['draws'][0].update(sample_accuracy=.123),
    lambda r: r['plus']['BIKMc']['draws'][0]['heldout'].update(nmi=.123),
    lambda r: r['plus']['BIKMc'].update(frequency=1),
    lambda r: r['plus']['BIKMc'].update(excluded_betas=[]),
    lambda r: r['plus']['BIKMc']['grid'][1].update(banco_converged=False),
    lambda r: r['plus']['BIKMc']['grid'][1].update(status='unavailable', reason='fake'),
    lambda r: r['methods']['IKMn']['ward'].update(converged=False, stop_reason='cycle'),
    lambda r: r['methods']['IKMn']['ward'].update(beta=2.),
    lambda r: r['methods']['IKMn']['ward'].update(nmi=.123),
    lambda r: r['methods']['IKMn']['random']['runs'][0].update(seed=99),
    lambda r: r['methods']['IKMn']['random']['summary']['nmi'].update(std=.1),
    lambda r: r['methods']['IKMn']['kmeans++']['runs'].pop(),
])
def test_scientific_tampering_rejected(recorded, mutate):
    rewrite_result(recorded, mutate)
    with pytest.raises(ValueError):
        verify.verify_results(recorded[0], recorded[1])


@pytest.mark.parametrize('target', ['result', 'extra', 'missing', 'code', 'data', 'manifest', 'seeds'])
def test_provenance_rejected(recorded, target):
    root, source, _ = recorded
    if target == 'result':
        (root/'Tiny.json').write_text('{}')
    elif target == 'extra':
        (root/'Stale.json').write_text('{}')
    elif target == 'missing':
        (root/'Tiny.json').unlink()
    elif target in ('code', 'data'):
        (source/('algorithm.py' if target == 'code' else 'data.txt')).write_text('changed')
    else:
        manifest = package.read_manifest(root)
        manifest.update(status='running') if target == 'manifest' else manifest.update(seeds=list(range(42, 142)))
        package.write_json(root/'manifest.json', manifest)
    with pytest.raises(ValueError):
        verify.verify_results(root, source)


def test_staging_immutable_and_failure_inventory(tmp_path):
    out = tmp_path/'run'
    out.mkdir()
    (out/'old').write_text('preserve')
    with pytest.raises(FileExistsError):
        with package.staging(out, {}):
            pytest.fail('Must not enter')
    assert (out/'old').read_text() == 'preserve'
    failed_out = tmp_path/'failed'
    with pytest.raises(RuntimeError):
        with package.staging(failed_out, {'kind': 'results'}) as stage:
            (stage/'done.json').write_text('{}')
            raise RuntimeError('worker failed')
    assert not failed_out.exists()
    failed_stage = next(tmp_path.glob('.failed.staging-*'))
    manifest = package.read_manifest(failed_stage)
    assert manifest['status'] == 'failed' and 'done.json' in manifest['files_sha256']
    with pytest.raises(ValueError, match='Incomplete'):
        package.validate_package(failed_stage)
    empty = tmp_path/'empty'
    empty.mkdir()
    with package.staging(empty, {'kind': 'fixture'}) as stage:
        (stage/'new.txt').write_text('new')
        package.seal(stage)
    assert (empty/'new.txt').read_text() == 'new'


def test_all_mixed_csv_rows_in_full_inventory():
    result = {}
    for c in d.PAPER_COLS:
        result[c] = dict(methods={n: {'random': {'runs': []}, 'kmeans++': {'runs': []}, 'ward': {'name': n, 'init': 'ward'}} for n, *_ in d.MINUS_SPECS},
                         plus={n: {'selected': {'name': n, 'init': 'BANCO'}} for n, *_ in d.PLUS_ALL})
    rows = tables.csv_rows(result, d.PAPER_COLS)
    mixed_names = {s[0] for s in d.PLUS_MIXED}
    assert sum(row['name'] in mixed_names for row in rows) == 112
    assert sum(row['init'] == 'BANCO' for row in rows) == 168


def test_old_nmi_rescored_without_mutation(specs):
    old = make_result()
    labels = old['reference_labels']
    runrow = old['plus']['BIKMn']['selected']
    runrow['nmi'] = normalized_mutual_info_score(labels, runrow['labels'], average_method='arithmetic')
    original = copy.deepcopy(old)
    rescored = tables.rescore_previous(old, labels)
    assert old == original
    assert rescored['plus']['BIKMn']['selected']['nmi'] != runrow['nmi']
    assert rescored['plus']['BIKMn']['selected']['nmi'] == run.metrics(np.asarray(labels), np.asarray(runrow['labels']))['nmi']
    with pytest.raises(ValueError, match='reference'):
        tables.rescore_previous(old, list(reversed(labels)))
    del old['plus']['BIKMn']['selected']['labels']
    with pytest.raises(ValueError, match='labels required'):
        tables.rescore_previous(old, labels)


def test_optimizer_passes_kind_and_records_banco(monkeypatch):
    calls = []
    class Model:
        weights = np.ones(4)
        converged_ = True
        stop_reason_ = 'fixed_point'
        n_iter_ = 1
        def __init__(self, *args, **kwargs): pass
        def fit(self, *args, **kwargs): return [{0, 1}, {2, 3}]
    monkeypatch.setattr(run, 'KM', Model)
    monkeypatch.setattr(d, 'idxs_for_weights', lambda w, n, kind: calls.append(kind) or [np.arange(n)])
    result = run.run_optimizer(np.zeros((4, 4)), np.array([0, 0, 1, 1]), ('BKMn', 'KM', 'A0', '2d'), 1., 42, hyp=[{0, 1}, {2, 3}])
    assert calls == ['KM'] and result['init'] == 'BANCO'


def test_runner_failure_records_each_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ProcessPoolExecutor', ThreadPoolExecutor)
    monkeypatch.setattr(d, 'ROOT', tmp_path)
    monkeypatch.setattr(d, 'dataset_folder', lambda f: tmp_path)
    def worker(folder, short, out, quick):
        if short == 'bad': raise RuntimeError('cycle fixture')
        package.write_json(Path(out)/(short+'.json'), {'done': True})
        return short
    monkeypatch.setattr(run, 'dataset_run', worker)
    with pytest.raises(RuntimeError, match='Dataset failures'):
        run.run_package(tmp_path/'output', [('a', 'good'), ('b', 'bad')], True, 2)
    assert not (tmp_path/'output').exists()
    stage = next(tmp_path.glob('.output.staging-*'))
    manifest = package.read_manifest(stage)
    assert manifest['status'] == 'failed'
    assert manifest['completed_datasets'] == ['good']
    assert manifest['failures'] == {'bad': 'RuntimeError: cycle fixture'}
    assert 'good.json' in manifest['files_sha256']


@pytest.fixture
def artifacts(recorded, tmp_path, monkeypatch):
    root, source, result = recorded
    baseline, previous = tmp_path/'baseline', tmp_path/'old'
    baseline.mkdir()
    previous.mkdir()
    (baseline/'historic.tex').write_text('historic only')
    package.write_json(previous/'Tiny.json', result)
    # Plot rendering itself is checked separately; no full-sized figures for corruption tests.
    monkeypatch.setattr(tables, 'render_comparisons', lambda *a: None)
    out = tmp_path/'artifacts'
    tables.generate(root, out, baseline, previous, source)
    return root, out, source


def test_artifact_roundtrip_portable_args(artifacts):
    root, out, source = artifacts
    assert verify.verify_artifacts(root, out, source) == (18, 7)
    verify.main(['--results', str(root), '--out', str(out), '--source-root', str(source)])
    verify.main(['--results', str(root), '--results-only', '--source-root', str(source)])


@pytest.mark.parametrize('target', ['table3', 'table4', 'ward', 'banco', 'beta', 'protocol', 'preview', 'csv', 'diagnostics', 'zip', 'binding'])
def test_artifact_corruption_rejected(artifacts, target):
    root, out, source = artifacts
    if target in ('table3', 'table4', 'ward', 'banco', 'beta', 'protocol', 'preview'):
        names = {'table3': 'tables/Table_3_B_plus_p1_ARI_and_NMI.tex',
                 'table4': 'tables/Table_4_B_plus_p2_ARI_and_NMI.tex',
                 'ward': 'tables/Table_5_B_minus_ARI_max_and_mean.tex',
                 'banco': 'tables/Table_6_B_minus_NMI_max_and_mean.tex',
                 'beta': 'tables/Selected_beta_and_frequency.tex',
                 'protocol': 'tables/Experiment_protocol.tex', 'preview': 'preview.tex'}
        path = out/names[target]
        path.write_text(path.read_text().replace('1.5', '9.9') + '\ncorruption')
    elif target in ('csv', 'diagnostics'):
        path = out/('individual_runs.csv' if target == 'csv' else 'candidate_diagnostics.csv')
        lines = path.read_text().splitlines()
        path.write_text('\n'.join(lines[:-1])+'\n')
    elif target == 'zip':
        (out/'manuscript_tables_final.zip').unlink()
    else:
        manifest = package.read_manifest(out)
        manifest['results_package_id'] = 'other-run'
        package.write_json(out/'manifest.json', manifest)
    package.seal(out)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify.verify_artifacts(root, out, source)


def test_select_beta_order_validation_and_strict_partition():
    y = np.array([0, 0, 1, 1, 1])
    rows = [dict(beta=2., labels=y.tolist()), dict(beta=1.1, labels=y.tolist())]
    assert run.select_beta(rows, y, draws=2)['beta'] == 1.1
    assert rows[0]['beta'] == 2.  # Caller order is untouched.
    for invalid in ([], [rows[0], rows[0]], [dict(beta=float('nan'))],
                    [dict(beta=float('inf'))], [dict(beta=True)], [dict(beta=.9)]):
        with pytest.raises(ValueError):
            run.select_beta(invalid, y)
    for partition in ([[0.1], [1]], [[True], [1]], [[0, 0], [1]], [[0], []], [[0], [0, 1]]):
        with pytest.raises(ValueError):
            run.partition_labels(partition, 2)
    assert run.partition_labels([{0}, {1}], 2).tolist() == [0, 1]


@pytest.mark.parametrize('field,value', [('banco_retained_count', 1), ('random_completion_centers', 1)])
def test_retained_center_metadata_tampering_rejected(recorded, field, value):
    rewrite_result(recorded, lambda r: r['plus']['BIKMn']['grid'][0].update({field: value}))
    with pytest.raises(ValueError, match='BANCO'):
        verify.verify_results(recorded[0], recorded[1])


def test_random_completion_diagnostic_excluded_even_if_lloyd_converges(recorded):
    root, source, result = recorded
    selection = result['plus']['BIKMc']
    diagnostic = selection['grid'][0]
    diagnostic.update(converged=True, stop_reason='fixed_point', banco_retained_count=1,
                      random_completion_centers=1, banco_extractions=diagnostic['banco_extractions'][:1])
    # The selected beta remains 1.5: pure BANCO eligibility excludes the 1.1 row.
    selection['legacy_unfiltered'] = run.select_beta(selection['grid'], np.array(result['reference_labels']))
    package.write_json(root/'Tiny.json', result)
    package.seal(root)
    verify.verify_results(root, source)
    # A plausible legacy selection is still forbidden as the main selection.
    selection.update(selection['legacy_unfiltered'])
    selection['excluded_betas'] = []
    package.write_json(root/'Tiny.json', result)
    package.seal(root)
    with pytest.raises(ValueError):
        verify.verify_results(root, source)


def test_dataset_run_counts_centers_and_records_no_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'PLUS_ALL', (('BIKMn', 'A0', 'IKM', 'A0', '2d'),))
    monkeypatch.setattr(d, 'MINUS_SPECS', ())
    monkeypatch.setattr(d, 'load_dataset', lambda _: (np.zeros((4, 4)), np.array([0, 0, 1, 1])))
    kinds = []
    monkeypatch.setattr(d, 'idxs_for_weights', lambda w, n, kind: kinds.append(kind) or [np.arange(n)])
    class Banco:
        converged_ = True
        extraction_status_ = [dict(converged=True, reason='fixed_point', iterations=1)]
        def __init__(self, *a, **kw): pass
        def fit(self, *a, **kw): return [{0, 1}]
    class Model:
        converged_, stop_reason_, n_iter_, weights = True, 'fixed_point', 1, np.ones(4)
        def __init__(self, *a, **kw): pass
        def fit(self, *a, **kw): return [{0, 1}, {2, 3}]
    monkeypatch.setattr(run, 'BANCOIn', Banco)
    monkeypatch.setattr(run, 'IKM', Model)
    assert run.dataset_run('tiny', 'Tiny', tmp_path, True) == 'Tiny'
    import json
    result = json.loads((tmp_path/'Tiny.json').read_text())
    row = result['plus']['BIKMn']['grid'][0]
    assert row['banco_retained_count'] == 1 and row['random_completion_centers'] == 1
    assert row['converged'] and result['plus']['BIKMn']['selected'] is None
    assert result['plus']['BIKMn']['status'] == 'unavailable'
    assert kinds == ['BANCO', 'IKM']


@pytest.mark.parametrize('kind', ['IKM', 'KM'])
def test_real_tiny_optimizer_initializers(kind):
    x = np.array([[0., .1, 1., 1.1], [.1, .2, 1.1, 1.2], [.2, .3, 1.2, 1.3],
                  [9., 9.1, 10., 10.1], [9.1, 9.2, 10.1, 10.2], [9.2, 9.3, 10.2, 10.3]])
    y = np.array([0, 0, 0, 1, 1, 1])
    for init in ('random', 'kmeans++', 'ward', 'BANCO'):
        hyp = [{0, 1, 2}, {3, 4, 5}] if init == 'BANCO' else None
        row = run.run_optimizer(x, y, ('fixture', kind, 'A1', '2d'), 1.5, 42,
                                init='kmeans++' if hyp else init, hyp=hyp)
        assert row['converged'] and row['clusters'] == 2 and row['beta'] == 1.5
        assert row['nmi'] == 1. and row['init'] == init
        if hyp:
            assert row['banco_retained_count'] == 2 and row['random_completion_centers'] == 0


def test_comparison_normalization_modes(specs, tmp_path, monkeypatch):
    result = make_result()
    calls = []
    monkeypatch.setattr(tables, 'comparison', lambda *args: calls.append(args))
    monkeypatch.setattr(tables, 'csv_old', lambda *args: {})
    monkeypatch.setattr(tables, 'tex_old', lambda *args: {})
    tables.render_comparisons(tmp_path, {'Tiny': result}, {'datasets': ['Tiny'], 'seeds': [42, 43, 44]}, tmp_path, {'Tiny': result})
    for call in calls:
        filename, mode = call[0].name, call[-1]
        if filename.startswith(('Table_3_', 'Table_4_')) and 'before_fixes' not in filename:
            assert mode == 'ari'
        if filename.startswith('Table_6_') and 'before_fixes' not in filename:
            assert mode == 'none'
        if 'before_fixes' in filename:
            assert mode == 'both' and 'max-normalized' in call[5]
        assert '100' not in call[1]


def test_partial_caption(recorded, monkeypatch):
    root, _, result = recorded
    monkeypatch.setattr(d, 'PAPER_COLS', ['Tiny', 'Other'])
    docs = tables.table_documents({'Tiny': result}, package.read_manifest(root))
    assert 'Partial dataset run.' in docs['tables/Table_3_B_plus_p1_ARI_and_NMI.tex']
    assert 'Other' not in docs['tables/Table_3_B_plus_p1_ARI_and_NMI.tex']


def make_unavailable(result, name):
    """A converged optimizer cannot rescue a cycling BANCO extraction."""
    rows = result['plus'][name]['grid']
    for row in rows:
        row['banco_converged'] = False
        row['banco_extractions'][-1].update(converged=False, reason='cycle', iterations=5)
    result['plus'][name] = run.candidate_selection(rows, np.array(result['reference_labels']))
    minus = name[1:]
    if minus not in result['methods']:
        return
    method = result['methods'][minus]
    weight = next(s[2] for s in d.MINUS_SPECS if s[0] == minus)
    if weight in ('A0', 'A2'):
        method['beta_source'] = 'beta_independent'
    else:
        method.update(beta=None, beta_source=None, ward=None,
                      **run.availability(False, 'no_eligible_banco_beta'))
        for init in ('random', 'kmeans++'):
            method[init] = dict(runs=[], summary=run.summarize([], result['k'], 'no_eligible_banco_beta'))


@pytest.fixture
def unavailable(recorded):
    for name in recorded[2]['plus']:
        make_unavailable(recorded[2], name)
    rewrite_result(recorded, lambda r: None)
    return recorded


def test_unavailable_complete_package_tables_csv_and_baselines(unavailable, tmp_path, monkeypatch):
    root, source, result = unavailable
    manifest, results = verify.verify_results(root, source)
    assert package.read_manifest(root)['status'] == 'complete'
    assert result['methods']['IKMn']['beta_source'] == 'beta_independent'
    assert result['methods']['IKMn']['beta'] == 1.
    assert result['methods']['IKMc']['beta'] is None
    docs = tables.table_documents(results, manifest)
    assert all('NA (' in docs[f] for f in docs if f.startswith('tables/') and 'protocol' not in f.lower())
    assert 'never zero' in docs['tables/Table_3_B_plus_p1_ARI_and_NMI.tex']
    rows = tables.csv_rows(results, ['Tiny'])
    for row in rows:
        if row.get('status') == 'unavailable':
            assert 'ari' not in row and 'nmi' not in row
    diagnostics = tables.csv_rows(results, ['Tiny'], diagnostics=True)
    assert len(diagnostics) == sum(len(s['grid']) for s in result['plus'].values())
    assert all(r['record_type'] == 'excluded_candidate' for r in diagnostics)
    assert all('"reason": "cycle"' in r['banco_extractions'] for r in diagnostics)
    assert tables.color('NA (failed 100 / 100)', '0.1 / 0.2') == '#e8eef5'
    previous = tables.rescore_previous(result, result['reference_labels'])
    assert previous['plus']['BIKMn']['selected'] is None
    assert 'nmi' not in previous['methods']['IKMc']['random']['summary']
    baseline, old = tmp_path/'baseline', tmp_path/'old'
    baseline.mkdir(); old.mkdir()
    package.write_json(old/'Tiny.json', result)
    monkeypatch.setattr(tables, 'render_comparisons', lambda *args: None)
    tables.generate(root, tmp_path/'unavailable-artifacts', baseline, old, source)
    verify.verify_artifacts(root, tmp_path/'unavailable-artifacts', source)


@pytest.mark.parametrize('mutate', [
    lambda r: r['plus']['BIKMn'].update(status='available'),
    lambda r: r['plus']['BIKMn'].update(reason=None),
    lambda r: r['plus']['BIKMn'].update(beta=1.),
    lambda r: r['plus']['BIKMn'].update(selected=r['plus']['BIKMn']['grid'][0]),
    lambda r: r['plus']['BIKMn'].update(ari=0., nmi=0.),
    lambda r: r['plus']['BIKMn'].update(frequency=50),
    lambda r: r['plus']['BIKMn'].update(frequencies={'1.0': 50}),
    lambda r: r['plus']['BIKMn'].update(draws=[{}]),
    lambda r: r['plus']['BIKMn'].update(excluded_betas=[]),
    lambda r: r['plus']['BIKMn']['grid'][0].update(ari=0.),
    lambda r: r['plus']['BIKMn']['grid'][0]['banco_extractions'][-1].update(reason='fixed_point'),
    lambda r: r['methods']['IKMn'].update(beta_source='banco_selected'),
    lambda r: r['methods']['IKMn'].update(beta=1.1),
    lambda r: r['methods']['IKMc'].update(beta=1.1, beta_source='beta_independent'),
    lambda r: r['methods']['IKMc'].update(ward=r['methods']['IKMn']['ward']),
    lambda r: r['methods']['IKMc']['random']['summary'].update(ari={'mean': 0.}),
    lambda r: r['methods']['IKMc']['random']['runs'].append(r['methods']['IKMn']['random']['runs'][0]),
    lambda r: r['methods']['IKMc'].update(status='available'),
])
def test_unavailable_schema_and_fake_metrics_rejected(unavailable, mutate):
    rewrite_result(unavailable, mutate)
    with pytest.raises(ValueError):
        verify.verify_results(unavailable[0], unavailable[1])


def test_cannot_hide_eligible_candidate_as_unavailable(recorded):
    selection = recorded[2]['plus']['BIKMn']
    selection.update(selected=None, beta=None, frequency=0, frequencies={}, draws=[],
                     **run.availability(False, 'no_eligible_candidate'))
    rewrite_result(recorded, lambda r: None)
    with pytest.raises(ValueError, match='Availability'):
        verify.verify_results(recorded[0], recorded[1])


def test_failed_starts_preserved_without_aggregate(tmp_path, specs, monkeypatch):
    monkeypatch.setattr(d, 'load_dataset', lambda _: (np.zeros((10, 4)), np.array([0]*4 + [1]*6)))
    class Banco:
        converged_ = False
        extraction_status_ = [dict(converged=False, reason='cycle', iterations=5)]
        def __init__(self, *args, **kwargs): pass
        def fit(self, *args, **kwargs): return [{0, 1}]
    monkeypatch.setattr(run, 'BANCOIn', Banco)
    calls = []
    def optimizer(x, y, spec, beta, seed, init='kmeans++', hyp=None):
        calls.append((spec[0], seed, init))
        success = hyp is not None or (init != 'ward' and seed != 43)
        return dict(name=spec[0], beta=beta, seed=seed, init='BANCO' if hyp else init,
                    labels=y.tolist(), clusters=2, converged=success,
                    stop_reason='fixed_point' if success else 'cycle', iterations=5, seconds=0.,
                    **(dict(banco_retained_count=1, random_completion_centers=1) if hyp else {}),
                    **run.metrics(y, y))
    monkeypatch.setattr(run, 'run_optimizer', optimizer)
    run.dataset_run('tiny', 'Tiny', tmp_path, False)
    import json
    result = json.loads((tmp_path/'Tiny.json').read_text())
    method = result['methods']['IKMn']
    assert method['ward']['status'] == 'unavailable'
    for init in ('random', 'kmeans++'):
        summary = method[init]['summary']
        assert summary['n'] == 100 and summary['failed'] == summary['nonconverged'] == 1
        assert summary['status'] == 'unavailable' and 'ari' not in summary and 'nmi' not in summary
        assert len(method[init]['runs']) == 100
        verify.verify_summary(summary, method[init]['runs'], 2)
    assert not any(c[0] == 'IKMc' for c in calls)
    assert result['methods']['IKMc']['ward'] is None
    diagnostics = tables.csv_rows({'Tiny': result}, ['Tiny'], diagnostics=True)
    assert sum(r['record_type'] == 'failed_run' for r in diagnostics) == 3


def test_failed_ward_and_summary_raw_rescore(recorded):
    method = recorded[2]['methods']['IKMn']
    method['ward'].update(converged=False, stop_reason='cycle',
                         **run.availability(False, 'nonconverged_or_wrong_k'))
    runs = method['random']['runs']
    runs[0].update(converged=False, stop_reason='iteration_limit')
    method['random']['summary'] = run.summarize(runs, 2)
    rewrite_result(recorded, lambda r: None)
    verify.verify_results(recorded[0], recorded[1])
    assert tables.pair(method['ward']).startswith('NA')
    assert tables.triple(method['random']['summary'], 'ari').startswith('NA')
    runs[0]['ari'] = -.99
    rewrite_result(recorded, lambda r: None)
    with pytest.raises(ValueError, match='ari mismatch'):
        verify.verify_results(recorded[0], recorded[1])


def test_actual_fungi4_a2_unavailable_bounded(tmp_path, monkeypatch):
    # One real pipeline, one beta, three starts per initializer; no full experiment.
    monkeypatch.setattr(d, 'PLUS_ALL', (('BIKMo', 'A2', 'IKM', 'A2', '2d'),))
    monkeypatch.setattr(d, 'MINUS_SPECS', (('IKMo', 'IKM', 'A2', '2d'),))
    monkeypatch.setattr(run, 'ProcessPoolExecutor', ThreadPoolExecutor)
    out = tmp_path/'actual'
    run.run_package(out, [('Fungi_genera_4', 'Fungi_4')], quick=True, workers=1)
    manifest, results = verify.verify_results(out)
    assert manifest['status'] == 'complete' and manifest['failures'] == {}
    result = results['Fungi_4']
    selection = result['plus']['BIKMo']
    assert selection['selected'] is None and selection['beta'] is None
    assert selection['status'] == 'unavailable' and selection['excluded_betas'] == [1.]
    extraction = selection['grid'][0]['banco_extractions'][2]
    assert extraction == dict(converged=False, reason='cycle', iterations=5)
    method = result['methods']['IKMo']
    assert method['beta'] == 1. and method['beta_source'] == 'beta_independent'
    assert len(method['random']['runs']) == len(method['kmeans++']['runs']) == 3
    y = np.asarray(result['reference_labels'])
    for row in selection['grid']:
        verify.verify_row(row, y, 'BIKMo', 1., 'BANCO', 42, result['k'])
    verify.verify_row(method['ward'], y, 'IKMo', 1., 'ward', 42, result['k'])
    for init in ('random', 'kmeans++'):
        for row in method[init]['runs']:
            verify.verify_row(row, y, 'IKMo', 1., init, row['seed'], result['k'])
        verify.verify_summary(method[init]['summary'], method[init]['runs'], result['k'])


def test_all_cells_unavailable_and_comparison_guard(unavailable, tmp_path, monkeypatch):
    root, source, result = unavailable
    method = result['methods']['IKMn']
    method['ward'].update(converged=False, stop_reason='cycle',
                         **run.availability(False, 'nonconverged_or_wrong_k'))
    for init in ('random', 'kmeans++'):
        for row in method[init]['runs']:
            row.update(converged=False, stop_reason='cycle')
        method[init]['summary'] = run.summarize(method[init]['runs'], 2)
    rewrite_result(unavailable, lambda r: None)
    manifest, results = verify.verify_results(root, source)
    docs = tables.table_documents(results, manifest)
    for name, document in docs.items():
        if 'protocol' in name.lower() or name == 'preview.tex':
            continue
        cells = [line for line in document.splitlines() if ' & ' in line and not line.startswith('Method')]
        assert cells and all(' & NA (' in line for line in cells)
    rows = tables.csv_rows(results, ['Tiny'])
    assert rows and all('ari' not in row and 'nmi' not in row for row in rows)
    calls = []
    monkeypatch.setattr(tables, 'comparison', lambda *args: calls.append(args))
    monkeypatch.setattr(tables, 'csv_old', lambda *args: {})
    monkeypatch.setattr(tables, 'tex_old', lambda *args: {})
    previous = {'Tiny': tables.rescore_previous(result, result['reference_labels'])}
    tables.render_comparisons(tmp_path, results, manifest, tmp_path, previous)
    assert calls
    for call in calls:
        assert all(cell.startswith('NA (') for columns in call[4].values() for cell in columns.values())


def test_summary_compatibility_and_wrong_k():
    metric_only = [dict(converged=True, ari=.5, nmi=.4)]
    assert run.summarize(metric_only)['ari']['mean'] == .5
    wrong_k = [dict(converged=True, clusters=1, ari=.5, nmi=.4)]
    assert run.summarize(wrong_k)['status'] == 'available'
    summary = run.summarize(wrong_k, 2)
    assert summary['status'] == 'unavailable' and summary['failed'] == 1
    assert summary['nonconverged'] == 0 and 'ari' not in summary
    verify.verify_summary(summary, wrong_k, 2)
    summary['nmi'] = dict(mean=0.)
    with pytest.raises(ValueError, match='fake metrics'):
        verify.verify_summary(summary, wrong_k, 2)


def test_reference_binding_rejects_changed_source_with_updated_hash(recorded):
    root, source, _ = recorded
    path=source/'subdatasets/tiny/real_clusters.txt'
    path.write_text('1, 2, 3, 4\n0, 5, 6, 7, 8, 9')
    manifest=package.read_manifest(root)
    manifest['data_sha256'][str(path.relative_to(source))]=package.sha256(path)
    package.write_json(root/'manifest.json',manifest)
    with pytest.raises(ValueError,match='reference labels differ'):
        verify.verify_results(root,source)


@pytest.mark.parametrize('field,value',[('features',99),('features',0)])
def test_source_dimensions_binding(recorded,field,value):
    rewrite_result(recorded,lambda r:r.update({field:value}))
    with pytest.raises(ValueError,match='dimensions differ'):
        verify.verify_results(recorded[0],recorded[1])


def test_loaded_reference_must_be_covered_by_manifest(recorded):
    root,source,_=recorded
    manifest=package.read_manifest(root)
    del manifest['data_sha256']['subdatasets/tiny/real_clusters.txt']
    package.write_json(root/'manifest.json',manifest)
    with pytest.raises(ValueError,match='missing from input provenance'):
        verify.verify_results(root,source)

"""API and command-line regressions for validation and preprocessing."""
import itertools
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from Algs.BANCOIn import BANCOIn
from Algs.IKM import IKM
from Algs.KM import KM
from Algs.optimization import default_feature_groups
from Algs.partitions import labels_from_clusters, read_clusters
from Algs.weights import A0, A1


METHODS = (IKM, KM, BANCOIn)
ROOT = Path(__file__).resolve().parents[1]
X = np.array([[0., 2., 7.], [2., 3., 7.], [5., 9., 7.], [9., 10., 7.]])


@pytest.mark.parametrize('cls', METHODS)
@pytest.mark.parametrize('flags', itertools.product((False, True), repeat=3))
def test_preprocessing_flags_reach_weight_update_without_mutating_input(cls, flags):
    observed = []

    def weights(data, clusters, **kwargs):
        observed.append(data.copy())
        return A0(data, clusters, **kwargs)

    center, half_range, variance = flags
    model = cls(weights, center_data=center, minmax_normilize=half_range,
                scale_by_var=variance)
    data = X.copy()
    model.fit(data, [np.arange(3)], 2, max_repetitions=3)
    expected = X.copy()
    if center:
        expected -= expected.mean(0)
    if half_range:
        half = np.ptp(expected, axis=0) / 2
        expected /= np.where(half < 1e-12, 1., half)
    if variance:
        deviation = expected.std(0)
        expected = (expected - expected.mean(0)) / np.where(deviation == 0, 1, deviation)
    assert observed
    for actual in observed:
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_array_equal(data, X)


@pytest.mark.parametrize('cls', METHODS)
@pytest.mark.parametrize('limit', [0, -1, 1.5, 2., np.nan, np.inf, True, '2', None])
def test_iteration_limit_requires_positive_integer(cls, limit):
    with pytest.raises(ValueError, match='positive integer'):
        cls(A0).fit(X, [np.arange(3)], 2, max_repetitions=limit)


@pytest.mark.parametrize('cls', METHODS)
@pytest.mark.parametrize('k', [0, 5, 1.5, True, None])
def test_invalid_cluster_count(cls, k):
    with pytest.raises(ValueError, match='K must'):
        cls(A0).fit(X, [np.arange(3)], k)


@pytest.mark.parametrize('cls', METHODS)
@pytest.mark.parametrize('data', [[], [1., 2.], [[np.nan]], [[np.inf]], np.empty((2, 0))])
def test_invalid_data(cls, data):
    with pytest.raises(ValueError):
        cls(A0).fit(data, [np.arange(3)], 1)


@pytest.mark.parametrize('cls', METHODS)
@pytest.mark.parametrize('groups', [
    [], [[]], [[0, 1]], [[0, 1, 1, 2]], [[0, 1], [1, 2]], [[-1, 0, 1]],
    [[0, 1, 3]], [[0., 1., 2.]], [[True, False, True]], [[[0, 1, 2]]],
    [[0, 1, 2], []], [[0, True, 2]], None,
])
def test_invalid_feature_groups_even_for_a0(cls, groups):
    with pytest.raises(ValueError):
        cls(A0).fit(X, groups, 2)


@pytest.mark.parametrize('cls', METHODS)
@pytest.mark.parametrize('beta', [0., .9, -1., np.nan, np.inf, True, None, '2'])
def test_invalid_beta_even_for_a0(cls, beta):
    with pytest.raises(ValueError, match='beta'):
        cls(A0).fit(X, [np.arange(3)], 2, beta_degree=beta)


@pytest.mark.parametrize('cls', METHODS)
@pytest.mark.parametrize('dim', ['bad', '2D', None, 2])
def test_invalid_dimension(cls, dim):
    message = 'cluster-specific' if cls is BANCOIn else '2d or 3d'
    with pytest.raises(ValueError, match=message):
        cls(A0, dim)


@pytest.mark.parametrize('cls', [IKM, KM])
def test_invalid_init_is_not_kmeanspp_fallback(cls):
    with pytest.raises(ValueError, match='init must'):
        cls(A0, init='typo')
    model = cls(A0)
    model.init = 'typo'
    with pytest.raises(ValueError, match='init must'):
        model.fit(X, [np.arange(3)], 2)


@pytest.mark.parametrize('er', [-1., np.nan, np.inf, -np.inf, None, '0', True])
def test_banco_invalid_error(er):
    with pytest.raises(ValueError, match='er must'):
        BANCOIn(A0).fit(X, [np.arange(3)], 2, er=er)


@pytest.mark.parametrize('cls', METHODS)
@pytest.mark.parametrize('weight_func', [A0, A1])
def test_odd_features_and_single_group_remain_supported(cls, weight_func):
    model = cls(weight_func)
    groups = model.fit(X, [np.arange(3)], np.int64(2),
                       max_repetitions=np.int64(5), beta_degree=1.)
    assert groups and all(groups)
    if cls is not BANCOIn:
        labels_from_clusters(groups, len(X))


@pytest.mark.parametrize('cls', METHODS)
def test_dispersion_contract(cls):
    modes = []

    def weights(data, clusters, **kwargs):
        modes.append(kwargs.get('dispersion', 'sse'))
        return A0(data, clusters, **kwargs)

    cls(weights).fit(X, [np.arange(3)], 2, max_repetitions=5)
    assert modes
    assert set(modes) == ({'variance'} if cls is BANCOIn else {'sse'})


@pytest.mark.parametrize('groups', [[], [[]], [[0, 0]], [[0], [0]], [[True]], [[1.]], [[-1]], [[4]], [[[0]]]])
def test_object_partitions_reject_invalid_groups_even_if_partial(groups):
    with pytest.raises(ValueError):
        labels_from_clusters(groups, 4, require_complete=False)


@pytest.mark.parametrize('text', ['0, 1\n2, 3', '0,1\r\n2,3\r\n', ' 0 , 1 \n 2 , 3 '])
def test_reader_preserves_final_index_and_accepts_whitespace(tmp_path, text):
    path = tmp_path / 'clusters.txt'
    path.write_text(text)
    assert read_clusters(path, 4) == [[0, 1], [2, 3]]


@pytest.mark.parametrize('text', ['0,0\n1,2,3', '0,1\n1,2,3', '', '\n', '0,1\n\n2,3', '0,1,\n2,3', '0,1\n2'])
def test_reader_rejects_duplicates_empty_and_incomplete_partitions(tmp_path, text):
    path = tmp_path / 'clusters.txt'
    path.write_text(text)
    with pytest.raises(ValueError):
        read_clusters(path, 4)


def test_reader_allows_valid_partial_seeds(tmp_path):
    path = tmp_path / 'seeds.txt'
    path.write_text('0\n3')
    assert read_clusters(path, 4, require_complete=False) == [[0], [3]]


@pytest.mark.parametrize('cls', [IKM, KM])
@pytest.mark.parametrize('seeds', [[], [[]], [[0, 0]], [[0], [0]], [[0.]], [[0], [1], [2]]])
def test_invalid_initial_clusters(cls, seeds):
    with pytest.raises(ValueError):
        cls(A0).fit(X, [np.arange(3)], 2, hyp_clusters=seeds)


@pytest.mark.parametrize('interval', [False, True])
@pytest.mark.parametrize('name', ['A0', 'A1', 'A2'])
def test_default_cli_feature_groups(interval, name):
    groups = default_feature_groups(4, name, interval)
    expected = [[0, 1], [2, 3]] if name == 'A2' or (interval and name == 'A1') else [[0, 1, 2, 3]]
    assert [g.tolist() for g in groups] == expected
    if name == 'A2' or (interval and name == 'A1'):
        with pytest.raises(ValueError, match='even'):
            default_feature_groups(3, name, interval)
    else:
        assert default_feature_groups(3, name, interval)[0].tolist() == [0, 1, 2]


def run_cli(name, style, args):
    entry = ['-m', f'Algs.{name}'] if style == 'module' else [str(ROOT / 'Algs' / f'{name}.py')]
    return subprocess.run([sys.executable, *entry, *args], cwd=ROOT,
                          text=True, capture_output=True, timeout=30)


@pytest.mark.parametrize('name', ['IKM', 'KM', 'BANCOIn'])
@pytest.mark.parametrize('style', ['module', 'script'])
def test_cli_newline_free_reference_and_partial_banco(tmp_path, name, style):
    dataset = tmp_path / 'data.npy'
    np.save(dataset, np.array([[-5., -5.], [-4., -4.], [4., 4.], [5., 5.]]))
    reference = tmp_path / 'reference.txt'
    reference.write_text('0, 1\n2, 3')
    result = run_cli(name, style, ['--dataroot', str(dataset), '--weightsfunc', 'A0',
                                 '--K', '1', '--manualSeed', '7', '--realclroot', str(reference)])
    assert result.returncode == 0, result.stderr
    if name == 'BANCOIn':
        assert 'ARI/NMI not computed for partial initialization' in result.stdout
        assert 'ARI: ' not in result.stdout
    else:
        assert 'ARI: ' in result.stdout


@pytest.mark.parametrize('name', ['IKM', 'KM'])
@pytest.mark.parametrize('duplicate', [False, True])
def test_cli_seed_reader_keeps_eof_and_duplicates(tmp_path, name, duplicate):
    dataset = tmp_path / 'data.npy'
    np.save(dataset, X)
    seeds = tmp_path / 'seeds.txt'
    seeds.write_text('0,0\n3' if duplicate else '0\n3')
    result = run_cli(name, 'module', ['--dataroot', str(dataset), '--weightsfunc', 'A0',
                                    '--K', '2', '--manualSeed', '7', '--hypclroot', str(seeds)])
    if duplicate:
        assert result.returncode != 0 and 'duplicate' in result.stderr
    else:
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('name', ['IKM', 'KM', 'BANCOIn'])
def test_cli_nmi_uses_max_entropy_normalization(tmp_path, name):
    dataset = tmp_path / 'data.npy'
    np.save(dataset, np.array([[-5., -5.], [-4., -4.], [4., 4.], [5., 5.]]))
    reference = tmp_path / 'reference.txt'
    reference.write_text('0, 1\n2\n3')
    result = run_cli(name, 'module', ['--dataroot', str(dataset), '--weightsfunc', 'A0',
                                    '--K', '2', '--manualSeed', '7', '--realclroot', str(reference)])
    assert result.returncode == 0, result.stderr
    # H(predicted)=log(2), H(reference)=1.5*log(2): max-normalized NMI=2/3.
    # Arithmetic normalization would incorrectly print 0.8 instead.
    assert float(result.stdout.split('NMI:')[-1]) == .67


def test_banco_rejects_global_weights_in_api_and_cli(tmp_path):
    with pytest.raises(ValueError, match='cluster-specific'):
        BANCOIn(A0, '2d')
    result = run_cli('BANCOIn', 'module', ['--dataroot', str(tmp_path / 'unused.npy'),
                                         '--weightsfunc', 'A0', '--K', '2', '--d', '2d'])
    assert result.returncode != 0
    assert "invalid choice: '2d'" in result.stderr


@pytest.mark.parametrize('er', [0., 1e-5])
def test_banco_accepts_finite_nonnegative_error(er):
    assert BANCOIn(A0).fit(X, [np.arange(3)], 2, er=er)


@pytest.mark.parametrize('cls', [IKM, KM])
def test_initial_cluster_index_arrays_remain_supported(cls):
    model = cls(A0)
    groups = model.fit(X, [np.arange(3)], 2, hyp_clusters=np.array([[0], [3]]))
    labels_from_clusters(groups, len(X))

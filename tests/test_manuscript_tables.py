"""Formatting and CLI checks using small saved records, without private inputs."""

import copy
import csv
import io
import re

import numpy as np
import pytest

from experiments import manuscript_tables as tables
from experiments import revision_data as d
from experiments.revision_run import summarize
from experiments.revision_verify import verify_summary


@pytest.fixture
def recorded():
    # Deliberately distinct maximum (.70), population std (.28), and mean (.50).
    runs = [
        dict(ari=ari, nmi=nmi, converged=True, clusters=2, seed=seed)
        for seed, ari, nmi in [(42, 0.1, 0.2), (43, 0.7, 0.8), (44, 0.7, 0.8)]
    ]
    summary = summarize(runs, 2)
    verify_summary(summary, runs, 2)
    result = {
        "plus": {
            spec[0]: {
                "status": "available",
                "reason": None,
                "selected": {"ari": -0.126, "nmi": 0.876},
                "beta": 1.7 if "A1" in (spec[1], spec[3]) else 1.0,
                "frequency": 31,
                "frequencies": {"1.7": 31, "2.0": 19},
                "grid": [{"ari": 0.99, "nmi": 0.99}],
                "legacy_unfiltered": {"selected": {"ari": 0.98, "nmi": 0.98}},
            }
            for spec in d.PLUS_ALL
        },
        "methods": {
            name: {
                "ward": {"status": "available", "reason": None, "ari": 0.456, "nmi": 0.654},
                "random": {"runs": copy.deepcopy(runs), "summary": copy.deepcopy(summary)},
                "kmeans++": {"runs": copy.deepcopy(runs), "summary": copy.deepcopy(summary)},
            }
            for name, *_ in d.MINUS_SPECS
        },
    }
    manifest = {
        "datasets": list(d.PAPER_COLS),
        "quick": False,
        "seeds": [42, 43, 44],
        "draws": 50,
        "sample_fraction": 0.2,
    }
    return manifest, {c: copy.deepcopy(result) for c in d.PAPER_COLS}


def csv_cells(documents, stem):
    rows = list(csv.reader(io.StringIO(documents[stem + ".csv"])))
    assert rows[0] == ["Method", "Fungi_3", "Fungi_4", "Fungi_5", "BSP_3", "BSP_4", "BSP_5", "Iris"]
    return {row[0]: row[1:] for row in rows[1:]}


def mock_verifier(monkeypatch, function):
    monkeypatch.setattr(tables, "verify_results", function)


def test_five_tables_preserve_all_methods_datasets_and_match_csv(recorded):
    manifest, results = recorded
    before = copy.deepcopy(recorded)
    documents = tables.table_documents(results, manifest)
    assert len(documents) == 10
    expected_rows = {
        "Table_3_B_plus_p1_ARI_and_NMI": [spec[0] for spec in d.PLUS_SAME],
        "Table_4_B_plus_p2_ARI_and_NMI": [spec[0] for spec in d.PLUS_MIXED],
        "Table_5_B_minus_ARI_max_std": [
            f"{name}: {init}" for name, *_ in d.MINUS_SPECS
            for init in ("random", "k-means++", "Ward", "BANCOi")
        ],
        "Table_6_B_minus_NMI_max_std": [
            f"{name}: {init}" for name, *_ in d.MINUS_SPECS
            for init in ("random", "k-means++", "Ward", "BANCOi")
        ],
        "Selected_beta": [spec[0] for spec in d.PLUS_ALL],
    }
    for stem, expected in expected_rows.items():
        cells = csv_cells(documents, stem)
        assert list(cells) == expected
        tex = documents[stem + ".tex"]
        assert "Fungi\\_3 & Fungi\\_4 & Fungi\\_5 & BSP\\_3 & BSP\\_4 & BSP\\_5 & Iris" in tex
        for name, values in cells.items():
            assert len(values) == 7
            assert " & ".join([name, *values]) + r" \\" in tex
        assert tex.count(r"\caption{") == 1
        assert r"\resizebox{\textwidth}{!}" in tex
    assert recorded == before
    for table_id, stem in enumerate(list(expected_rows)[:4], start=1):
        assert r"\label{table:" + str(table_id) + "}" in documents[stem + ".tex"]
    assert r"\label{tab:revision-beta}" in documents["Selected_beta.tex"]
    all_tex = "\n".join(text for name, text in documents.items() if name.endswith(".tex"))
    # Cross-reference labels are manuscript identifiers, not displayed prose.
    visible_tex = re.sub(r"\\label\{[^}]*\}", "", all_tex)
    for word in ("revision", "review", "letter", "workflow", "diagnostic", "legacy", "mean", "frequency"):
        assert re.search(r"\b" + word + r"\b", visible_tex.lower()) is None


@pytest.mark.parametrize("metric,maximum,ward", [("ARI", "0.70", "0.46"), ("NMI", "0.80", "0.65")])
def test_initialization_uses_population_std_not_mean(recorded, metric, maximum, ward):
    manifest, results = recorded
    documents = tables.table_documents(results, manifest)
    stem = f"Table_{5 if metric == 'ARI' else 6}_B_minus_{metric}_max_std"
    cells = csv_cells(documents, stem)
    for init in ("random", "k-means++"):
        assert cells[f"IKMn: {init}"] == [f"{maximum} / 0.28 / 2"] * 7
    assert cells["IKMn: Ward"] == [ward] * 7
    assert cells["IKMn: BANCOi"] == ["-0.13" if metric == "ARI" else "0.88"] * 7
    assert "population standard deviation" in documents[stem + ".tex"]
    assert "over 3 starts" in documents[stem + ".tex"]
    assert f"{np.std([0.1, 0.7, 0.7], ddof=1):.2f}" == "0.35"  # Not the displayed .28.


def test_scores_use_selected_candidate_not_grid_or_historical_best(recorded):
    manifest, results = recorded
    documents = tables.table_documents(results, manifest)
    for stem in ("Table_3_B_plus_p1_ARI_and_NMI", "Table_4_B_plus_p2_ARI_and_NMI"):
        assert all(values == ["-0.13 / 0.88"] * 7 for values in csv_cells(documents, stem).values())


def test_hit_count_is_not_recomputed_from_rounded_scores():
    summary = summarize([dict(ari=v, nmi=v, converged=True) for v in (0.801, 0.804, 0.804)])
    assert tables.summary_cell(summary, "ari") == "0.80 / 0.00 / 2"


def test_unavailable_scores_are_dashes_even_with_failed_raw_metrics(recorded):
    manifest, results = recorded
    result = results["Fungi_4"]
    for name in ("BIKMn", "BIKMc", "BnIKMc"):
        result["plus"][name].update(status="unavailable", reason="no_eligible_candidate", selected=None, beta=None)
    method = result["methods"]["IKMn"]
    method["ward"].update(status="unavailable", reason="nonconverged_or_wrong_k")
    method["random"]["runs"][0]["converged"] = False
    method["random"]["summary"] = summarize(method["random"]["runs"], 2)
    method["kmeans++"]["summary"] = summarize([], 2, "no_eligible_banco_beta")
    results["Fungi_4"]["methods"]["IKMc"]["ward"] = None
    documents = tables.table_documents(results, manifest)
    assert csv_cells(documents, "Table_3_B_plus_p1_ARI_and_NMI")["BIKMn"][1] == "--"
    assert csv_cells(documents, "Table_4_B_plus_p2_ARI_and_NMI")["BnIKMc"][1] == "--"
    for table_id, metric in ((5, "ARI"), (6, "NMI")):
        cells = csv_cells(documents, f"Table_{table_id}_B_minus_{metric}_max_std")
        assert all(cells[f"IKMn: {init}"][1] == "--" for init in ("random", "k-means++", "Ward", "BANCOi"))
        assert cells["IKMc: Ward"][1] == "--"
    betas = csv_cells(documents, "Selected_beta")
    assert betas["BIKMn"][1] == "n/a"
    assert betas["BIKMc"][1] == "--"
    assert "no eligible" not in "\n".join(documents.values())


def test_beta_is_modal_only_and_independence_checks_both_stages(recorded):
    manifest, results = recorded
    documents = tables.table_documents(results, manifest)
    cells = csv_cells(documents, "Selected_beta")
    for spec in d.PLUS_ALL:
        expected = "1.7" if "A1" in (spec[1], spec[3]) else "n/a"
        assert cells[spec[0]] == [expected] * 7
    assert cells["BcIKMn"] == cells["BoIKMc"] == ["1.7"] * 7
    assert cells["BoIKMn"] == ["n/a"] * 7
    assert "(31/50)" not in documents["Selected_beta.tex"]
    assert "Modal values" in documents["Selected_beta.tex"]


@pytest.mark.parametrize("kind", ["missing", "reordered", "quick"])
def test_incomplete_tables_are_rejected(recorded, kind):
    manifest, results = recorded
    if kind == "missing":
        manifest["datasets"].remove("Iris")
        del results["Iris"]
    elif kind == "reordered":
        manifest["datasets"].reverse()
    else:
        manifest["quick"] = True
    with pytest.raises(ValueError, match="Reduced-grid" if kind == "quick" else "dataset inventory"):
        tables.table_documents(results, manifest)


def test_cli_calls_strict_verifier_and_exports_its_records(tmp_path, monkeypatch, recorded, capsys):
    calls = []
    source = tmp_path / "published"
    out = tmp_path / "export"
    mock_verifier(monkeypatch, lambda root: calls.append(root) or recorded)
    monkeypatch.setattr(d, "load_dataset", lambda _: pytest.fail("No raw data may be loaded"))
    tables.main(["--results", str(source), "--out", str(out)])
    assert calls == [source]
    assert len(list(out.glob("*.tex"))) == len(list(out.glob("*.csv"))) == 5
    assert "all 7 datasets" in capsys.readouterr().out
    expected = tables.table_documents(recorded[1], recorded[0])
    assert {p.name: p.read_text() for p in out.iterdir()} == expected


@pytest.mark.parametrize("kind", ["empty", "nonempty", "file", "symlink"])
def test_output_must_be_new_even_before_validation(tmp_path, monkeypatch, kind):
    out = tmp_path / "export"
    if kind in ("empty", "nonempty"):
        out.mkdir()
        if kind == "nonempty":
            (out / "keep").write_text("original")
    elif kind == "file":
        out.write_text("original")
    else:
        out.symlink_to(tmp_path / "missing")
    mock_verifier(monkeypatch, lambda _: pytest.fail("Existing output must be rejected first"))
    with pytest.raises(FileExistsError, match="Refusing existing"):
        tables.generate(tmp_path / "results", out)
    if kind == "nonempty":
        assert (out / "keep").read_text() == "original"
    if kind == "file":
        assert out.read_text() == "original"


def test_cli_validation_failure_creates_no_output(tmp_path, monkeypatch, capsys):
    def reject(_):
        raise ValueError("Package file inventory/hash mismatch")

    mock_verifier(monkeypatch, reject)
    out = tmp_path / "export"
    with pytest.raises(SystemExit) as error:
        tables.main(["--out", str(out)])
    assert error.value.code == 2
    assert not out.exists()
    assert "hash mismatch" in capsys.readouterr().err


def test_cli_defaults_to_accepted_results_and_ignored_output(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(tables, "generate", lambda root, out: calls.append((root, out)))
    tables.main([])
    assert calls == [(d.ROOT / "experiments/results/ward_final", d.ROOT / "letter/manuscript_tables")]
    assert "5 manuscript tables" in capsys.readouterr().out

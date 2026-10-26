"""Export five manuscript tables from accepted, recorded results.

Run ``python -m experiments.manuscript_tables --out <new directory>``.
The TeX fragments require booktabs and graphicx; CSVs contain the same cells.
Scientific validation belongs to revision_verify, not to this renderer.
"""

import argparse
import csv
import io
from pathlib import Path

from experiments import revision_data as d
from experiments.revision_verify import verify_results


DASH = "--"


def score_cell(record, metric=None):
    """Selected BANCO rows lack status; unavailable Ward rows retain scores."""
    if record is None or record.get("status") == "unavailable":
        return DASH
    if metric is None:
        return f"{record['ari']:.2f} / {record['nmi']:.2f}"
    return f"{record[metric]:.2f}"


def summary_cell(summary, metric):
    """Use the validated population standard deviation and unrounded hit count."""
    if summary["status"] == "unavailable":
        return DASH
    values = summary[metric]
    return f"{values['max']:.2f} / {values['std']:.2f} / {values['hits']}"


def beta_cell(selection, spec):
    # Either stage can depend on beta, including mixed BANCO/optimizer weights.
    if "A1" not in (spec[1], spec[3]):
        return "n/a"
    if selection["status"] == "unavailable" or selection["selected"] is None:
        return DASH
    return f"{selection['beta']:.1f}"


def _tex_table(caption, label, headers, rows):
    # Only fixed method/dataset labels and formatted numeric cells enter here.
    def row_text(row):
        return " & ".join(str(cell).replace("_", r"\_") for cell in row) + r" \\"

    lines = [
        r"\begin{table*}[tbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + "}",
        r"\toprule",
        row_text(headers),
        r"\midrule",
        *(row_text(row) for row in rows),
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def table_documents(results, manifest):
    """Render validated records without modifying or recomputing any result."""
    cols = manifest["datasets"]
    if cols != d.PAPER_COLS or set(results) != set(d.PAPER_COLS):
        raise ValueError(
            "Partial or unexpected dataset inventory: manuscript tables require "
            + ", ".join(d.PAPER_COLS)
            + "; received "
            + ", ".join(cols)
        )
    if manifest["quick"]:
        raise ValueError("Reduced-grid results cannot be exported as manuscript tables")

    documents = {}
    headers = ["Method", *cols]
    unavailable = "A dash denotes an unavailable result."
    nmi_note = "NMI is normalized by the larger of the two partition entropies."

    def add(stem, caption, label, rows):
        documents[stem + ".tex"] = _tex_table(caption, label, headers, rows)
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerows([headers, *rows])
        documents[stem + ".csv"] = buffer.getvalue()

    def banco_score(dataset, name, metric=None):
        selection = results[dataset]["plus"][name]
        selected = selection["selected"] if selection["status"] == "available" else None
        return score_cell(selected, metric)

    for table_id, specs, stem, weighting in [
        (3, d.PLUS_SAME, "B_plus_p1_ARI_and_NMI", "the same weighting scheme"),
        (4, d.PLUS_MIXED, "B_plus_p2_ARI_and_NMI", "different weighting schemes"),
    ]:
        add(
            f"Table_{table_id}_{stem}",
            f"Clustering performance with {weighting} in BANCOi and the subsequent "
            "optimizer. Entries report adjusted Rand index (ARI) / normalized "
            f"mutual information (NMI). {nmi_note} {unavailable}",
            f"table:{table_id - 2}",
            [[spec[0], *(banco_score(c, spec[0]) for c in cols)] for spec in specs],
        )

    for table_id, metric, title in [
        (5, "ari", "Adjusted Rand index (ARI)"),
        (6, "nmi", "Normalized mutual information (NMI)"),
    ]:
        rows = []
        for name, *_ in d.MINUS_SPECS:
            for init, display in [("random", "random"), ("kmeans++", "k-means++")]:
                rows.append(
                    [f"{name}: {display}"]
                    + [
                        summary_cell(results[c]["methods"][name][init]["summary"], metric)
                        for c in cols
                    ]
                )
            rows.append(
                [f"{name}: Ward"]
                + [score_cell(results[c]["methods"][name]["ward"], metric) for c in cols]
            )
            rows.append(
                [f"{name}: BANCOi"] + [banco_score(c, "B" + name, metric) for c in cols]
            )
        add(
            f"Table_{table_id}_B_minus_{metric.upper()}_max_std",
            f"{title} for four initialization methods. Random and k-means++ entries "
            "report maximum / population standard deviation / number of runs attaining "
            f"the maximum over {len(manifest['seeds'])} starts "
            r"(absolute tolerance $10^{-12}$). Ward and BANCOi entries report a single "
            r"score. For beta-dependent methods, all initializations use the modal $\beta$ "
            "selected with BANCOi. "
            + (nmi_note + " " if metric == "nmi" else "")
            + unavailable,
            f"table:{table_id - 2}",
            rows,
        )

    add(
        "Selected_beta",
        r"Modal values of $\beta$ selected over "
        f"{manifest['draws']} random subsets, each containing approximately "
        f"{manifest['sample_fraction'] * 100:g}"
        r"\% of the reference labels. "
        "n/a denotes beta-independent methods; a dash denotes an unavailable selection.",
        "tab:revision-beta",
        [
            [spec[0], *(beta_cell(results[c]["plus"][spec[0]], spec) for c in cols)]
            for spec in d.PLUS_ALL
        ],
    )
    return documents


def generate(results_root, out):
    """Verify saved records, then create a new directory containing all ten files."""
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Refusing existing output path: {out}")

    manifest, results = verify_results(Path(results_root))
    documents = table_documents(results, manifest)
    # mkdir also rejects a destination created while validation was running.
    out.mkdir(parents=True, exist_ok=False)
    for filename, text in documents.items():
        (out / filename).write_text(text, encoding="utf-8")
    return documents


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=d.ROOT / "experiments/results/ward_final"
    )
    parser.add_argument("--out", type=Path, default=d.ROOT / "letter/manuscript_tables")
    args = parser.parse_args(argv)
    try:
        generate(args.results, args.out)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Exported 5 manuscript tables and 5 matching CSVs for all 7 datasets to {args.out}")


if __name__ == "__main__":
    main()

"""Render review tables from recorded results; never run/select experiments."""
from pathlib import Path
import argparse, csv, json, re, zipfile
import shutil
import copy
from sklearn.metrics import normalized_mutual_info_score
from experiments import revision_package as package
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import revision_data as d


NA_NOTE = ('NA means unavailable, never zero: no eligible candidate requires BANCO and optimizer '
           'convergence with exactly K centers/clusters; no selected beta prevents A1 comparisons. '
           'Any nonconverged or wrong-K start suppresses its aggregate; all attempted runs and failure counts '
           'remain in JSON, with excluded candidates/failed runs in candidate_diagnostics.csv. '
           'When matching BANCO is unavailable, A0/A2 comparisons use fixed beta=1 (beta-independent). ')


def na(reason):
    return 'NA (' + reason.replace('_', ' ') + ')'


def number(x): return f'{x:.2f}'
def pair(r, reason='no_eligible_candidate'):
    if r is None or r.get('status') == 'unavailable':
        return na(reason if r is None else r['reason'])
    return f"{number(r['ari'])} / {number(r['nmi'])}"


def scalar(r, metric, reason):
    if r is None or r.get('status') == 'unavailable':
        return na(reason if r is None else r['reason'])
    return number(r[metric])


def triple(r,m):
    if r.get('status') == 'unavailable':
        return na(r['reason'])
    v=r[m]
    return f"{number(v['max'])} / {number(v['mean'])} / {v['hits']}"


def beta_cell(selection, draws):
    if selection['selected'] is None:
        return na(selection['reason'])
    return f"{selection['beta']:.1f} ({selection['frequency']}/{draws})"


def escape(s): return str(s).replace('_',r'\_').replace('%',r'\%')

def tex_text(caption, label, rows, headers=None, column_spec=None):
    headers = headers or ['Method'] + d.PAPER_COLS
    lines=[r'\begingroup',r'\small',r'\setlength{\LTcapwidth}{\textwidth}',r'\setlength{\tabcolsep}{3pt}',
           r'\begin{longtable}{'+(column_spec or 'l'+'r'*(len(headers)-1))+'}',
           r'\caption{'+caption+'}'+r'\label{'+label+r'}\\',r'\toprule',
           ' & '.join(map(escape,headers))+r' \\',r'\midrule',r'\endfirsthead',
           r'\toprule',' & '.join(map(escape,headers))+r' \\',r'\midrule',r'\endhead',
           r'\bottomrule',r'\endfoot']
    for row in rows:
        lines.append(' & '.join(map(escape,row))+r' \\')
        if ': BANCO' in row[0]:
            lines.append(r'\addlinespace[2pt]')
    lines += [r'\end{longtable}',r'\endgroup']
    return '\n'.join(lines)+'\n'


def csv_old(path, expected_names=None):
    with path.open() as h: rows = list(csv.DictReader(h))
    result = {}
    for i, row in enumerate(rows):
        name = row['name']
        if name in result:
            # Table 4 source prints BnKMcs twice. Preserve both printed rows,
            # but do not assert the second row belongs to BoKMcs.
            if expected_names is None:
                raise ValueError(f'Duplicate source row: {name}')
            key = expected_names[i]
            row['__label__'] = name + ' (second row)'
            row['__unmatched__'] = True
            result[key] = row
        else:
            result[name] = row
    return result


def tex_old(path):
    result={}
    for line in path.read_text().splitlines():
        if '&' not in line or 'dataset' in line or line.startswith('Name'):continue
        parts=[x.strip() for x in line.split('&')]
        if parts[0] not in [s[0] for s in d.PLUS_ALL+d.MINUS_SPECS]:continue
        result[parts[0]]={col: re.sub(r'\\.*$','',cell).strip() for col,cell in zip(d.PAPER_COLS,parts[1:])}
    return result


def values(cell):
    return [float(x) for x in re.findall(r'(?<![A-Za-z0-9.])[+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?',cell)[:2]]


def color(new,old,mode='both'):
    if mode == 'none': return 'white'
    if new.startswith('NA') or old.startswith('NA'): return '#e8eef5'
    a,b=values(new),values(old)
    if len(a)<2 or len(b)<2:return '#e8eef5'
    count = 1 if mode == 'ari' else 2
    dif=np.round(np.array(a[:count])-b[:count],4)
    if (dif>=0).all() and (dif>0).any():return '#d7eedb'
    if (dif<=0).all() and (dif<0).any():return '#f5d7d7'
    if (dif!=0).any():return '#fff0c7'
    return 'white'


def comparison(path,title,names,old,new,oldnote,newnote,cols=None,color_mode='both'):
    cols = d.PAPER_COLS if cols is None else cols
    n=len(names)
    fig,axes=plt.subplots(2,1,figsize=(18, max(8,n*.64+3)))
    fig.suptitle(title,fontsize=15,y=.986)
    for idx,(ax,lookup,note) in enumerate(zip(axes,[old,new],[oldnote,newnote])):
        ax.axis('off');ax.set_title(note,fontsize=11,pad=12)
        cells=[[lookup.get(name,{}).get('__label__',name)]+[lookup.get(name,{}).get(c,'--') for c in cols] for name in names]
        tbl=ax.table(cellText=cells,colLabels=['Method']+cols,cellLoc='center',bbox=[0,0,1,.96])
        tbl.auto_set_font_size(False);tbl.set_fontsize(9 if n>10 else 10)
        for (i,j),cell in tbl.get_celld().items():
            cell.set_linewidth(.4)
            if i==0:cell.set_facecolor('#dce3ec');cell.set_text_props(weight='bold')
            elif j==0:cell.set_facecolor('#f1f3f5')
            elif idx==1:cell.set_facecolor(color(cells[i-1][j],('--' if old.get(names[i-1],{}).get('__unmatched__') else old.get(names[i-1],{}).get(cols[j-1],'--')),color_mode))
    legend = ('Historical NMI normalization is unverified or arithmetic; comparison with max-normalized NMI is not equivalent.\n' + ('Colors compare ARI only.' if color_mode == 'ari' else 'Metric coloring disabled.') if color_mode != 'both' else 'Green: both scores no worse; red: both no better; yellow: mixed; blue: no old value.\nMulti-start colors compare maximum and mean only; hit counts are not compared with standard deviations.')
    fig.text(.5,.016,legend,ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.96),h_pad=2)
    fig.savefig(path,dpi=170);plt.close(fig)


def table_documents(result, manifest):
    """Every generated TeX file, including preview, from validated recorded data."""
    cols = manifest['datasets']
    draws, starts = manifest['draws'], len(manifest['seeds'])
    mode = 'Quick diagnostic run. ' if manifest['quick'] else ''
    if cols != d.PAPER_COLS:
        mode += 'Partial dataset run. '
    grid = ', '.join(f'{v:.1f}' for v in manifest['grid'])
    protocol = (f'{draws} random 20\\% label subsets; modal beta; '
                'eligible BANCO runs retain exactly K centers without random completion; all-object scores are descriptive. NMI uses maximum entropy normalization. '
                'Main A1 grid starts at 1.1 (beta above one); pure A0/A2 use beta=1 as a placeholder, not a fitted parameter. ')
    docs = {}

    def add(filename, caption, label, rows):
        docs['tables/' + filename] = tex_text(mode + caption + ' ' + escape(NA_NOTE), label, rows, ['Method'] + cols)

    for table_id, specs, stem in [(3, d.PLUS_SAME, 'B_plus_p1_ARI_and_NMI'),
                                  (4, d.PLUS_MIXED, 'B_plus_p2_ARI_and_NMI')]:
        add(f'Table_{table_id}_{stem}.tex', 'BANCO followed by the indicated optimizer. ' + protocol,
            f'tab:revision-{table_id}', [[s[0]] + [pair(result[c]['plus'][s[0]]['selected'])
                                                  for c in cols] for s in specs])
    for table_id, metric in [(5, 'ari'), (6, 'nmi')]:
        rows = []
        for name, *_ in d.MINUS_SPECS:
            for init in ['random', 'kmeans++']:
                rows.append([f'{name}: {init}'] + [triple(result[c]['methods'][name][init]['summary'], metric) for c in cols])
            rows.append([f'{name}: Ward'] + [scalar(result[c]['methods'][name]['ward'], metric, 'no_eligible_banco_beta') for c in cols])
            rows.append([f'{name}: BANCO'] + [scalar(result[c]['plus']['B'+name]['selected'], metric, 'no_eligible_candidate') for c in cols])
        add(f'Table_{table_id}_B_minus_{metric.upper()}_max_and_mean.tex',
            f'{metric.upper()}. Random and k-means++: {starts} starts, maximum / mean / number attaining the maximum '
            r'(absolute tolerance $10^{-12}$). Ward and BANCO: one run. When available, all four initializations use the same BANCO-selected $\beta$. '
            'Data are centered and divided by the endpoint half-range. ' + protocol,
            f'tab:revision-{table_id}', rows)
    add('Selected_beta_and_frequency.tex',
        f'Selected modal beta and frequency among {draws} random 20\\% label subsets. '
        f'A1 grid: {grid}. Pure A0/A2 pipelines use beta=1 because their effective objective is beta-independent. '
        'Tied sample scores choose the smaller beta; tied modal frequencies retain the first encountered value.',
        'tab:revision-beta', [[s[0]] + [beta_cell(result[c]['plus'][s[0]], draws)
                                      for c in cols] for s in d.PLUS_ALL])
    docs['tables/Experiment_protocol.tex'] = tex_text(mode + 'Weighting and objective conventions.',
        'tab:revision-protocol', [
            ['A1 groups', 'IKM and BANCO: lower and upper endpoints separately; KM: one group.'],
            ['A2 groups', 'Lower and upper endpoints; product constraint across both groups.'],
            ['BANCO dispersion', 'Within-cluster population variance plus rho; adaptive BANCO remains heuristic.'],
            ['IKM/KM dispersion', 'Within-cluster SSE plus rho; global weights pool SSE.'],
            ['Regularization', 'rho is mean input-column population variance within each group, floored at 1e-12.'],
            ['Objective', r'$J=\sum D w^{\beta}$, where D is the regularized dispersion (SSE plus rho for IKM/KM).'],
            ['Main A1 grid', grid],
            ['Pure A0/A2', 'beta=1 is a placeholder, not a fitted parameter; effective distance weights are beta-independent.'],
        ], ['Convention', 'Definition'], r'p{.16\textwidth}p{.78\textwidth}')
    docs['preview.tex'] = (r'\documentclass[10pt]{article}'+'\n'+
        r'\usepackage[a4paper,landscape,margin=12mm]{geometry}'+'\n'+
        r'\usepackage{booktabs,longtable}'+'\n'+r'\begin{document}'+'\n'+
        r'\setcounter{table}{2}'+'\n'+'\n'.join(r'\input{'+f+'}'+r'\clearpage' for f in docs)+
        '\n'+r'\end{document}'+'\n')
    return docs


def csv_rows(result, cols, diagnostics=False):
    """Reported rows and failed diagnostics have separate CSVs and record types."""
    rows = []
    def append(dataset, run, record_type, **extra):
        values = {key: value for key, value in run.items() if key != 'labels'}
        if 'banco_extractions' in values:
            values['banco_extractions'] = json.dumps(values['banco_extractions'], sort_keys=True)
        rows.append(dict(dataset=dataset, **values, record_type=record_type, **extra))
    for c in cols:
        for name, *_ in d.MINUS_SPECS:
            method = result[c]['methods'][name]
            source = method.get('beta_source', 'banco_selected')
            for init in ['random', 'kmeans++']:
                for run in method[init]['runs']:
                    failed = not run['converged'] or run['clusters'] != result[c]['k']
                    if failed == diagnostics:
                        append(c, run, 'failed_run' if failed else 'run', beta_source=source)
                summary = method[init].get('summary', {})
                if not diagnostics and summary.get('status') == 'unavailable':
                    append(c, dict(name=name, init=init, beta=method['beta'], **summary),
                           'unavailable_summary', beta_source=source)
            ward = method['ward']
            if ward is None or ward.get('status') == 'unavailable':
                if diagnostics and ward is not None:
                    append(c, ward, 'failed_run', beta_source=source)
                if not diagnostics:
                    append(c, dict(name=name, init='ward', beta=method['beta'], status='unavailable',
                                   reason=method['reason'] if ward is None else ward['reason']),
                           'unavailable', beta_source=source)
            elif not diagnostics:
                append(c, ward, 'run', beta_source=source)
        for name, *_ in d.PLUS_ALL:
            selection = result[c]['plus'][name]
            if diagnostics:
                for candidate in selection.get('grid', []):
                    if candidate['beta'] in selection['excluded_betas']:
                        append(c, candidate, 'excluded_candidate')
            elif selection['selected'] is None:
                append(c, dict(name=name, init='BANCO', beta=None, status=selection['status'],
                               reason=selection['reason']), 'unavailable')
            else:
                append(c, selection['selected'], 'selected')
    return rows


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)) or ['dataset', 'name', 'record_type'])
        writer.writeheader()
        writer.writerows(rows)


def render_comparisons(out, result, manifest, baseline, previous):
    cols, starts = manifest['datasets'], len(manifest['seeds'])
    plots = out/'comparisons'
    plots.mkdir()
    def plot(filename, title, names, old, new, oldnote, newnote, color_mode='both'):
        comparison(plots/filename, title, names, old, new, oldnote, newnote, cols, color_mode)
    for table_id, specs, stem in [(3, d.PLUS_SAME, 'B_plus_p1_ARI_and_NMI'),
                                  (4, d.PLUS_MIXED, 'B_plus_p2_ARI_and_NMI')]:
        names = [s[0] for s in specs]
        new = {n: {c: pair(result[c]['plus'][n]['selected']) for c in cols} for n in names}
        for label, old, note in [
            ('paper', csv_old(baseline/(stem+'.csv'), names), 'Manuscript PDF, 20 September 2026; duplicate labels preserved'),
            ('previous_amorim', tex_old(baseline/(stem+'.tex')), 'Previously delivered Amorim archive, 14 September 2026'),
            ('before_fixes', {n: {c: pair(previous[c]['plus'][n]['selected']) for c in cols} for n in names}, 'Previous Ward partitions rescored with max-normalized NMI')]:
            plot(f'Table_{table_id}_{label}_vs_new.png', f'Table {table_id}: ARI / NMI', names, old, new,
                 'Old: '+note, 'New: max-normalized NMI; beta reselected; BANCO and optimizer convergence required',
                 'both' if label == 'before_fixes' else 'ari')
    names = [s[0] for s in d.MINUS_SPECS]
    for table_id, metric in [(5, 'ari'), (6, 'nmi')]:
        stem = f'B_minus_{metric.upper()}_max_and_mean'
        for init, tag in [('random', 'random'), ('kmeans++', 'kmeanspp')]:
            new = {n: {c: triple(result[c]['methods'][n][init]['summary'], metric) for c in cols} for n in names}
            plot(f'Table_{table_id}_paper_vs_{tag}.png', f'Table {table_id}: {metric.upper()}, {starts} new starts', names,
                 csv_old(baseline/(stem+'.csv')), new, 'Old manuscript: maximum / mean / standard deviation',
                 f'New {init}: maximum / mean / hits', 'none' if metric == 'nmi' else 'both')
            if init == 'kmeans++':
                plot(f'Table_{table_id}_previous_amorim_vs_new.png', f'Table {table_id}: {starts} new starts', names,
                     tex_old(baseline/(stem+'.tex')), new, 'Old Amorim: maximum / mean / standard deviation / hits',
                     'New: maximum / mean / hits; beta matched to BANCO', 'none' if metric == 'nmi' else 'both')
                old = {n: {c: triple(previous[c]['methods'][n][init]['summary'], metric) for c in cols} for n in names}
                plot(f'Table_{table_id}_before_fixes_vs_new.png', f'Table {table_id}: {starts} new starts', names,
                     old, new, 'Before fixes: saved partitions rescored with max-normalized NMI; maximum / mean / hits',
                     'After fixes and reselection: maximum / mean / hits')
    old = {n: {c: pair(result[c]['plus']['B'+n]['selected']) for c in cols} for n in names}
    new = {n: {c: pair(result[c]['methods'][n]['ward'], 'no_eligible_banco_beta') for c in cols} for n in names}
    plot('Ward_vs_BANCO_matched_beta.png', 'Ward versus BANCO initialization; ARI / NMI', names, old, new,
         'BANCO initialization + optimizer', 'Ward initialization + same optimizer at the same beta')


def rescore_previous(result, reference_labels):
    """Never compare old arithmetic-NMI summaries with new max-NMI summaries."""
    result = copy.deepcopy(result)
    if result['reference_labels'] != reference_labels:
        raise ValueError('Historical reference labels/order differ; cannot compare partitions')
    def score(run):
        labels = run.get('labels')
        if labels is None or len(labels) != len(reference_labels):
            raise ValueError('Historical labels required for max-NMI comparison')
        run['nmi'] = float(normalized_mutual_info_score(reference_labels, labels, average_method='max'))
    for selection in result['plus'].values():
        if selection['selected'] is not None:
            score(selection['selected'])
    for method in result['methods'].values():
        if method['ward'] is not None:
            score(method['ward'])
        for init in ('random', 'kmeans++'):
            for run in method[init]['runs']:
                score(run)
            if method[init]['summary'].get('status') == 'unavailable':
                continue
            vals = np.asarray([run['nmi'] for run in method[init]['runs']])
            method[init]['summary']['nmi'] = dict(max=float(vals.max()), mean=float(vals.mean()),
                std=float(vals.std()), hits=int(np.isclose(vals, vals.max(), rtol=0, atol=1e-12).sum()))
    return result


def generate(results, out, baseline, previous_results, source_root=d.ROOT):
    from experiments.revision_verify import verify_results, verify_artifacts
    manifest, result = verify_results(results, source_root)
    metadata = dict(kind='artifacts', results_package_id=manifest['package_id'],
                    results_manifest_sha256=package.sha256(Path(results)/'manifest.json'),
                    datasets=manifest['datasets'], quick=manifest['quick'])
    with package.staging(out, metadata) as stage:
        # Historical data are copied only as comparison sources, never reused as new results.
        shutil.copytree(baseline, stage/'baseline')
        (stage/'previous_results').mkdir()
        previous = {}
        for c in manifest['datasets']:
            path = Path(previous_results)/f'{c}.json'
            shutil.copy2(path, stage/'previous_results'/path.name)
            previous[c] = rescore_previous(json.loads(path.read_text()), result[c]['reference_labels'])
        (stage/'tables').mkdir()
        for filename, text in table_documents(result, manifest).items():
            (stage/filename).write_text(text)
        write_csv(stage/'individual_runs.csv', csv_rows(result, manifest['datasets']))
        write_csv(stage/'candidate_diagnostics.csv', csv_rows(result, manifest['datasets'], diagnostics=True))
        render_comparisons(stage, result, manifest, stage/'baseline', previous)
        (stage/'tables/README.txt').write_text('Review draft; not inserted into the manuscript.\n'
            'Tables 3–6 retain manuscript numbering. Require booktabs and longtable.\n'
            'Supplementary beta tables describe the recorded protocol. Baselines are historical comparison sources only.\n'
            + NA_NOTE + '\n')
        with zipfile.ZipFile(stage/'manuscript_tables_final.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((stage/'tables').iterdir()):
                archive.write(path, 'tables/'+path.name)
        package.seal(stage)
        verify_artifacts(results, stage, source_root)
    print(f'Created validated artifact package: {out}')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=d.ROOT/'experiments/results/ward_final')
    parser.add_argument('--out', type=Path, default=d.ROOT/'letter/ward_final')
    parser.add_argument('--baseline', type=Path, default=d.ROOT/'letter/ward_bugfix/baseline')
    parser.add_argument('--previous-results', type=Path, default=d.ROOT/'experiments/results/ward_bugfix')
    parser.add_argument('--source-root', type=Path, default=d.ROOT)
    args = parser.parse_args(argv)
    generate(args.results, args.out, args.baseline, args.previous_results, args.source_root)


if __name__ == '__main__':
    main()

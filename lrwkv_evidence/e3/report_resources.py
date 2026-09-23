"""Render complete E3 resource and parsing diagnostics from portable receipt CSVs.

Reads two compressed exports, not the 105600 individual receipt files. Existing
collector resource summaries are independently checked against the CSV values.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

MODELS=('F2','R0')
LENGTHS=(16384,32768,65536)
CAVEAT=('Descriptive request costs under asymmetric warmup: F2 performs cache-replay qualification outside item timing; R0 first requests may include JIT compilation. R0 recomputes its full visible prefix for each output token with use_cache=False, so these are not optimized autoregressive latency, KV-cache, speedup, or goodput comparisons. Peak GiB values are maxima of per-request CUDA allocation/reservation readings, not process RSS.')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def percentile(values,p):
    ordered=sorted(values)
    if not ordered:raise ValueError('empty resource group')
    x=(len(ordered)-1)*p;lo=math.floor(x);hi=math.ceil(x)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(x-lo)


def summarize(records,summary):
    """Summarize already validated portable records and cross-check collector totals."""
    if summary.get('complete') is not True or summary.get('report_kind')!='complete_conditional_comparison':
        raise ValueError('complete confirmation summary required')
    cells={r['cell_id']:r for r in summary['cell_scores']}
    if len(cells)!=len(summary['cell_scores']):raise ValueError('duplicate summary cells')
    groups=defaultdict(list);parses=defaultdict(lambda:[0,0]);seen=set();counts=defaultdict(int);correct=defaultdict(int)
    for row in records:
        model=row['model'];cell=cells.get(row['cell_id'])
        if model not in MODELS or cell is None:raise ValueError('unknown model or cell in CSV')
        key=(model,row['cell_id'],int(row['data_seed']),int(row['instance_index']))
        if key in seen:raise ValueError('duplicate portable receipt')
        seen.add(key)
        if int(row['correct']) not in (0,1) or int(row['parse_ok']) not in (0,1):raise ValueError('nonbinary portable outcome')
        nfe=int(row['actual_nfe']);wall=float(row['wall_seconds'])
        allocated=float(row['peak_allocated_bytes']);reserved=float(row['peak_reserved_bytes'])
        if nfe<1 or any(not math.isfinite(v) or v<0 for v in (wall,allocated,reserved)):
            raise ValueError('invalid portable resource value')
        groups[model,cell['length']].append((nfe,wall,allocated,reserved))
        parses[model,cell['length'],cell['family']][0]+=int(row['parse_ok'])
        parses[model,cell['length'],cell['family']][1]+=1
        counts[model,row['cell_id']]+=1;correct[model,row['cell_id']]+=int(row['correct'])
    for model in MODELS:
        if sum(n for (m,_),n in counts.items() if m==model)!=summary['observed_items'][model]:
            raise ValueError('portable model count differs from complete summary')
        for cell_id,cell in cells.items():
            if counts[model,cell_id]!=cell['n'] or not math.isclose(correct[model,cell_id]/cell['n'],cell[model],rel_tol=0,abs_tol=1e-12):
                raise ValueError('portable cell count/outcome differs from summary')
    if set(groups)!={(m,length) for m in MODELS for length in LENGTHS}:
        raise ValueError('six complete resource groups required')
    rows=[]
    for model,length in sorted(groups):
        group=groups[model,length];nfe,wall,allocated,reserved=map(list,zip(*group))
        values=dict(n=len(group),nfe_mean=statistics.mean(nfe),nfe_median=statistics.median(nfe),
            nfe_p95=percentile(nfe,.95),wall_seconds_mean=statistics.mean(wall),
            wall_seconds_median=statistics.median(wall),wall_seconds_p95=percentile(wall,.95),
            wall_seconds_sum=math.fsum(wall),peak_allocated_bytes_max=max(allocated),peak_reserved_bytes_max=max(reserved))
        original=summary['resources'][model][str(length)]
        for key,value in values.items():
            if key not in original:
                if key=='nfe_p95':continue
                raise ValueError('collector summary lacks resource field '+key)
            if not math.isclose(value,original[key],rel_tol=1e-9,abs_tol=1e-8):
                raise ValueError('portable resource disagrees with collector: '+key)
        rows.append(dict(model=model,length=length,**values,
                         peak_allocated_gib=max(allocated)/2**30,peak_reserved_gib=max(reserved)/2**30))
    parse_rows=[dict(model=model,length=length,family=family,parsed=parsed,n=n,parse_rate=parsed/n)
                for (model,length,family),(parsed,n) in sorted(parses.items())]
    return dict(schema='lrwkv_e3_resource_appendix_v1',complete=True,resources=rows,parsing=parse_rows,
                quantile_method='inclusive linear interpolation at sorted index (n-1)*p',resource_scope=CAVEAT,
                parsing_scope='Task-grammar parsing only; a parseable answer need not be correct. Counts concern the supported confirmation cells, not post-development controls.')


def build_report(summary_path,manifest_path):
    summary_path=Path(summary_path);manifest_path=Path(manifest_path)
    summary=json.loads(summary_path.read_text());manifest=json.loads(manifest_path.read_text())
    if manifest.get('summary_sha256')!=sha(summary_path) or manifest.get('summary_cell_counts_verified') is not True or manifest.get('csv_roundtrip_counts_verified') is not True:
        raise ValueError('export is not bound to this completed summary')
    if set(manifest['csv'])!=set(MODELS):raise ValueError('both portable model exports required')
    verified={}
    def records():
        for model in MODELS:
            item=manifest['csv'][model];path=Path(item['path'])
            actual=sha(path)
            if actual!=item['sha256']:raise ValueError('portable CSV hash mismatch')
            verified[model]=dict(path=str(path),sha256=actual)
            n=0
            with gzip.open(path,'rt',newline='') as stream:
                for row in csv.DictReader(stream):
                    if row['model']!=model:raise ValueError('misfiled portable model')
                    n+=1;yield row
            if n!=item['records'] or n!=manifest['record_count_per_model'][model]:raise ValueError('portable manifest count mismatch')
    result=summarize(records(),summary)
    result.update(summary_path=str(summary_path),summary_sha256=sha(summary_path),
                  export_manifest_path=str(manifest_path),export_manifest_sha256=sha(manifest_path),csv=verified)
    return result


def latex(result):
    lines=[r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',
        r'Model & Length & $n$ & NFE mean/med./p95 & Seconds med./p95 & Peak GiB alloc./res. \\',r'\midrule']
    for r in result['resources']:
        lines.append(f"{r['model']} & {r['length']//1024}K & {r['n']:,} & {r['nfe_mean']:.2f}/{r['nfe_median']:.1f}/{r['nfe_p95']:.1f} & {r['wall_seconds_median']:.3f}/{r['wall_seconds_p95']:.3f} & {r['peak_allocated_gib']:.2f}/{r['peak_reserved_gib']:.2f}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',
        r'\caption{Descriptive per-request resources on completed supported confirmation cells. NFE counts actual model calls. Medians and p95 use inclusive linear interpolation; GiB values are maxima of request-level CUDA allocated/reserved bytes. F2 cache-replay qualification is outside item timing, whereas R0 first requests may include JIT compilation. R0 recomputes the full prefix with no KV cache. These measurements do not establish an optimized autoregressive speed comparison, speedup, or goodput.}',
        r'\label{tab:e3-confirmation-resources}',r'\end{table}','',
        r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{@{}llrr@{}}',r'\toprule',
        r'Length & Family & F2 parseable/$n$ & R0 parseable/$n$ \\',r'\midrule']
    names=dict(associative_recall='Recall',overwrite_delayed_query='Overwrite',code_dataflow='Dataflow')
    lookup={(r['model'],r['length'],r['family']):r for r in result['parsing']}
    for length,family in sorted({(r['length'],r['family']) for r in result['parsing']}):
        f,r=(lookup[m,length,family] for m in MODELS)
        lines.append(f"{length//1024}K & {names[family]} & {f['parsed']:,}/{f['n']:,} & {r['parsed']:,}/{r['n']:,}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Task-grammar parsing on completed supported confirmation cells. Parseability is a diagnostic of output form and does not imply correctness. Unsupported cells are excluded, not zero-filled.}',r'\label{tab:e3-confirmation-parsing}',r'\end{table}','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=Path('results/e3_confirmation_summary.json'))
    parser.add_argument('--export-manifest',type=Path,default=Path('results/e3_receipt_export/manifest.json'))
    parser.add_argument('--out',type=Path,default=Path('results/e3_resources_appendix.json'))
    args=parser.parse_args();report=build_report(args.summary,args.export_manifest)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    args.out.with_suffix('.tex').write_text(latex(report))
    print(f'wrote {args.out} and {args.out.with_suffix(".tex")}')


if __name__=='__main__':main()

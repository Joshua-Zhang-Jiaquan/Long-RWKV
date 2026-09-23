"""Portable, source-hashed CSV export of a completed confirmation comparison.

Only accepted partition cells are read. This is a numerical evidence export of
an already validated collector summary, not a substitute for decoder validation.
No model imports, GPU calls, or writes to source output directories occur.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

from lrwkv_evidence.parallel_io import read_json_parallel

MODELS = ('F2', 'R0')
FIELDS = ('model', 'cell_id', 'data_seed', 'instance_index', 'correct', 'parse_ok',
          'actual_nfe', 'wall_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes',
          'provenance_sha256', 'source_file_sha256')
DEFAULT_PANEL = Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/capability_eval_data/lc_grid324')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _key(row):
    return row['cell_id'],row['data_seed'],row['instance_index']


def _partitions(plan, cells):
    covered=set(); outputs=set(); parts=[]
    lengths={c['history_length'] for c in cells.values()}
    jobs={(j['model'].upper(),j['length']):j for j in plan['jobs']}
    if len(jobs)!=len(plan['jobs']) or set(jobs)!={(m,l) for m in MODELS for l in lengths}:
        raise ValueError('plan does not uniquely cover models and lengths')
    for (model,length),job in sorted(jobs.items()):
        eligible={c for c,v in cells.items() if v['history_length']==length}
        entries=job.get('partitions',[dict(out=job['out'],cell_ids=sorted(eligible))])
        if not any(Path(p['out']).resolve()==Path(job['out']).resolve() for p in entries):
            raise ValueError('partition plan omits primary output')
        for part in entries:
            assigned=set(part['cell_ids']); out=str(Path(part['out']).resolve())
            pairs={(model,c) for c in assigned}
            if not assigned or len(assigned)!=len(part['cell_ids']) or not assigned<=eligible or pairs&covered or (model,out) in outputs:
                raise ValueError('invalid, duplicate, or overlapping accepted partition')
            covered|=pairs; outputs.add((model,out))
            parts.append(dict(model=model,length=length,out=out,cell_ids=sorted(assigned)))
    if covered!={(m,c) for m in MODELS for c in cells}:
        raise ValueError('accepted partition coverage is incomplete')
    return parts


def export(plan_path, summary_path, out_dir, *, expected_total=105600):
    plan_path,summary_path,out_dir=map(Path,(plan_path,summary_path,out_dir))
    plan_hash,summary_hash=sha(plan_path),sha(summary_path)
    plan,summary=read(plan_path),read(summary_path)
    if summary.get('complete') is not True or summary.get('report_kind')!='complete_conditional_comparison':
        raise ValueError('completed confirmation summary required')
    if set(summary.get('observed_items',{}))!=set(MODELS) or sum(summary['observed_items'].values())!=expected_total:
        raise ValueError('summary does not contain required complete record count')
    panel=Path(plan.get('panel',DEFAULT_PANEL))
    grid_path=panel/'grid_manifest.json'; grid=read(grid_path)
    if grid.get('split')!='iclr2027_grid' or grid.get('partial_run',False):
        raise ValueError('canonical grid is not complete confirmation')
    canonical={}; unsupported=[]; grid_hashes={}
    grid_files = (p for p in sorted(panel.glob('*.json')) if p != grid_path)
    for path, payload, cell in read_json_parallel(grid_files):
        if 'cell_id' not in cell: continue
        grid_hashes[path.name]=hashlib.sha256(payload).hexdigest()
        if cell['status']!='ok': unsupported.append(cell['cell_id']); continue
        if cell['cell_id'] in canonical: raise ValueError('duplicate canonical cell')
        canonical[cell['cell_id']]=cell
    if len(canonical)!=summary['supported_cells'] or len(unsupported)!=summary['excluded_cells']:
        raise ValueError('summary and canonical support differ')
    expected={}
    for cell_id,cell in canonical.items():
        if cell['total_instances']!=len(cell['instances']): raise ValueError('incomplete canonical cell')
        for inst in cell['instances']:
            key=_key(inst)
            if key[0]!=cell_id or key in expected: raise ValueError('duplicate or misfiled canonical item')
            expected[key]=inst
    if len(expected)*2!=expected_total or any(summary['observed_items'][m]!=len(expected) for m in MODELS):
        raise ValueError('canonical item count disagrees with required export')
    parts=_partitions(plan,canonical)
    summary_parts={(r['model'],r['length'],str(Path(r['out']).resolve())):r for r in summary['partitions']}
    if len(summary_parts)!=len(parts): raise ValueError('summary does not match current accepted plan')
    scores={m:{} for m in MODELS}; evidence=[]; counts=Counter()
    out_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.receipt-export-',dir=out_dir) as temporary:
        temp=Path(temporary); streams={}; writers={}
        try:
            for model in MODELS:
                raw=(temp/f'{model.lower()}_confirmation_receipts.csv.gz').open('wb')
                gz=gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0)
                stream=io.TextIOWrapper(gz,encoding='utf-8',newline='')
                streams[model]=(stream,raw)
                writers[model]=csv.DictWriter(stream,fieldnames=FIELDS);writers[model].writeheader()
            for part in parts:
                model=part['model'];out=Path(part['out']);assigned=set(part['cell_ids'])
                status=summary_parts.get((model,part['length'],part['out']))
                wanted=sum(len(canonical[c]['instances']) for c in assigned)
                if not status or status['assigned_cells']!=len(assigned) or status['used_receipts']!=wanted:
                    raise ValueError('summary and accepted partition counts differ')
                provenance_files={str(p):sha(p) for p in sorted(out.glob('provenance_rank*.json'))}
                provenance_hashes={digest(read(p)) for p in map(Path,provenance_files)}
                if len(provenance_hashes)!=1: raise ValueError('missing or mixed output provenance')
                inventory={}; n=0
                paths=out.glob('*/*__*__*.json') if model=='F2' else out.glob('*__*__*.json')
                # Exclude superseded/outside-prefix receipts before scheduling I/O.
                # Ordered, bounded prefetch preserves deterministic gzip row order.
                accepted_paths = (p for p in sorted(paths) if p.name.split('__')[0] in assigned)
                for path, payload, row in read_json_parallel(accepted_paths):
                    key=_key(row)
                    if key not in expected or key[0] not in assigned or path.stem!='__'.join(map(str,key)):
                        raise ValueError('receipt identity outside accepted canonical partition')
                    if key in scores[model]: raise ValueError('duplicate accepted receipt')
                    inst=expected[key]
                    if (row.get('status')!='ok' or row.get('split')!='iclr2027_grid'
                        or row.get('prompt_hash_verified') is not True
                        or row.get('input_ids_sha256')!=inst['input_ids_sha256']
                        or row.get('provenance_sha256') not in provenance_hashes
                        or type(row.get('correct')) is not bool or type(row.get('parse_ok')) is not bool
                        or type(row.get('actual_nfe')) is not int or row['actual_nfe']<1):
                        raise ValueError('invalid accepted receipt')
                    for k in ('wall_seconds','peak_allocated_bytes','peak_reserved_bytes'):
                        if type(row.get(k)) not in (int,float) or not math.isfinite(row[k]) or row[k]<0:
                            raise ValueError('invalid resource value')
                    receipt_hash=hashlib.sha256(payload).hexdigest()
                    inventory[str(path.relative_to(out))]=receipt_hash
                    values={k:row[k] for k in FIELDS if k not in ('model','source_file_sha256')}
                    values.update(model=model,source_file_sha256=receipt_hash,
                                  correct=int(row['correct']),parse_ok=int(row['parse_ok']))
                    writers[model].writerow(values)
                    scores[model][key]=row['correct'];counts[model]+=1;n+=1
                if n!=wanted: raise ValueError('missing accepted receipts')
                evidence.append(dict(**part,accepted_records=n,accepted_inventory_sha256=digest(inventory),
                    provenance_files_sha256=provenance_files,
                    auxiliary_files_sha256={str(p):sha(p) for folder in ('helper_schedule','cache_parity')
                                            for p in sorted((out/folder).glob('*.json'))}))
        finally:
            for stream,raw in streams.values(): stream.close();raw.close()
        if any(set(scores[m])!=set(expected) for m in MODELS): raise ValueError('missing semantic pairs')
        cell_summary={c['cell_id']:c for c in summary['cell_scores']}
        if len(cell_summary)!=len(summary['cell_scores']) or set(cell_summary)!=set(canonical):
            raise ValueError('summary cell scores do not match canonical cells')
        for cell_id,cell in canonical.items():
            keys=[_key(i) for i in cell['instances']];s=cell_summary[cell_id];n=len(keys)
            if s['n']!=n: raise ValueError('summary cell denominator differs')
            for model in MODELS:
                correct=sum(scores[model][key] for key in keys)
                if not math.isclose(correct/n,s[model],rel_tol=0,abs_tol=1e-12):
                    raise ValueError('export correct counts disagree with summary')
            paired=dict(F2_only=sum(scores['F2'][k] and not scores['R0'][k] for k in keys),
                        R0_only=sum(scores['R0'][k] and not scores['F2'][k] for k in keys),
                        both_correct=sum(scores['F2'][k] and scores['R0'][k] for k in keys))
            if any(s[k]!=v for k,v in paired.items()): raise ValueError('paired discordance differs from summary')
        # Read emitted CSV bytes back and independently compare exported cell totals.
        csv_evidence={}
        for model in MODELS:
            path=temp/f'{model.lower()}_confirmation_receipts.csv.gz'
            totals=Counter(); n=0
            with gzip.open(path,'rt',newline='') as stream:
                for row in csv.DictReader(stream):totals[row['cell_id']]+=int(row['correct']);n+=1
            if n!=len(expected) or any(not math.isclose(totals[c]/cell_summary[c]['n'],cell_summary[c][model],abs_tol=1e-12,rel_tol=0) for c in canonical):
                raise ValueError('CSV roundtrip count mismatch')
            csv_evidence[model]=dict(path=str(out_dir/path.name),sha256=sha(path),records=n)
        if sha(plan_path)!=plan_hash or sha(summary_path)!=summary_hash: raise ValueError('plan or summary changed during export')
        manifest=dict(schema='lrwkv_confirmation_receipt_export_v1',plan_path=str(plan_path),plan_sha256=plan_hash,
            summary_path=str(summary_path),summary_sha256=summary_hash,canonical_grid_manifest_path=str(grid_path),
            canonical_grid_manifest_sha256=sha(grid_path),canonical_cell_files_sha256=grid_hashes,
            record_count_per_model=dict(counts),total_records=sum(counts.values()),csv=csv_evidence,outputs=evidence,
            semantic_pairing='Exactly one F2 and one R0 receipt per canonical (cell_id,data_seed,instance_index); equal canonical prompt hash. Accepted partition assignments only; outside-assignment extras excluded. Correct counts and paired discordances match the completed summary. Unsupported cells are not zero-filled.',
            summary_cell_counts_verified=True,csv_roundtrip_counts_verified=True)
        for path in temp.glob('*.csv.gz'):os.replace(path,out_dir/path.name)
        (out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=Path('results/e3_confirmation_plan.json'))
    parser.add_argument('--summary',type=Path,default=Path('results/e3_confirmation_summary.json'))
    parser.add_argument('--out',type=Path,default=Path('results/e3_receipt_export'))
    args=parser.parse_args()
    manifest=export(args.plan,args.summary,args.out)
    print(json.dumps(dict(total_records=manifest['total_records'],csv=manifest['csv']),indent=2))


if __name__=='__main__':main()

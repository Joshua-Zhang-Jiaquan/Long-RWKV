"""Validate exact confirmation endpoints and retain structure-level contrasts."""
from collections import defaultdict
import json
import statistics
from lrwkv_evidence.train04.evaluate import support
from lrwkv_evidence.train04dev.collect_evaluation import validate_endpoint

METHODS=('one','information_set','random_halves','sequential')
FIELDS=('joint_kl_nats','exact_valid_mass','dependence_cost_nats','estimation_error_nats','conditional_valid_kl_nats')


def summarize(rows,panel):
    expected={(record['index'],method) for record,ex in panel for method in METHODS}
    by={(row['index'],row['method']):row for row in rows}
    if len(rows)!=len(by) or set(by)!=expected:raise ValueError('missing or duplicate endpoint rows')
    by_structure=defaultdict(list);conditions=[]
    for record,ex in panel:
        for method in METHODS:
            row=by[record['index'],method];validate_endpoint(row)
            if row['n']!=8 or row['family']!=record['family'] or row['instance_id']!=record['instance_id'] or row['structure']!=record['structure'] or row['valid_endpoints']!=support(ex):
                raise ValueError('endpoint does not match frozen public condition')
            by_structure[(row['family'],tuple(row['structure']),method)].append(row)
        a=by[record['index'],'information_set'];b=by[record['index'],'random_halves']
        conditions.append(dict(index=record['index'],family=record['family'],structure=record['structure'],
                               random_minus_information_set_kl=b['joint_kl_nats']-a['joint_kl_nats'],
                               information_set_minus_random_valid_mass=a['exact_valid_mass']-b['exact_valid_mass']))
    structures=[]
    for (family,structure,method),items in sorted(by_structure.items()):
        structures.append(dict(family=family,structure=list(structure),method=method,conditions=len(items),
                               **{field:statistics.mean(row[field] for row in items) for field in FIELDS}))
    grouped=defaultdict(list)
    for row in structures:grouped[row['family'],row['method']].append(row)
    families=[dict(family=family,method=method,structures=len(items),conditions=sum(r['conditions'] for r in items),
                   **{field:statistics.mean(row[field] for row in items) for field in FIELDS})
              for (family,method),items in sorted(grouped.items())]
    contrasts=defaultdict(list)
    for row in conditions:contrasts[row['family'],tuple(row['structure'])].append(row)
    structure_contrasts=[dict(family=family,structure=list(structure),conditions=len(items),
                              **{field:statistics.mean(row[field] for row in items) for field in ('random_minus_information_set_kl','information_set_minus_random_valid_mass')})
                         for (family,structure),items in sorted(contrasts.items())]
    return dict(family_summary=families,per_structure=structures,condition_contrasts=conditions,structure_contrasts=structure_contrasts)


def collect(out,stage):
    from .evaluate import load_panel
    from . import contract as K
    manifest,manifest_sha=K.validate(stage/'manifest.json',stage,stage/'model_source')
    panel=load_panel(stage,manifest);provenance=[];rows=[];calls=0
    common=('checkpoint_sha256','manifest_sha256','panel_sha256','partitions_sha256','seed','phase','step','triton_f32_default','torch_tf32')
    for rank in range(8):
        shard=[json.loads(line) for line in (out/f'rank{rank}.jsonl').read_text().splitlines()]
        if not shard or shard[0].get('kind')!='provenance' or shard[-1].get('kind')!='complete':raise ValueError('incomplete shard')
        p=shard[0]
        if p['rank']!=rank or p['world']!=8 or p['manifest_sha256']!=manifest_sha or p['step']!=2500 or p['triton_f32_default']!='ieee' or p['torch_tf32'] or p['seed'] not in (53,71,89) or p['phase'] not in ('independent','complementary'):
            raise ValueError('shard provenance differs from frozen experiment')
        if p['panel_sha256']!=manifest['panel_sha256'] or p['partitions_sha256']!=manifest['partitions_sha256']:raise ValueError('wrong panel provenance')
        if provenance and any(p[field]!=provenance[0][field] for field in common):raise ValueError('mixed checkpoints or training seeds')
        provenance.append(p)
        expected={(panel[position][0]['index'],method) for position in range(rank,len(panel),8) for method in METHODS}
        actual=[r for r in shard if r['kind']=='endpoint']
        if {(r['index'],r['method']) for r in actual}!=expected:raise ValueError('wrong shard condition assignment')
        if len(actual)!=len(shard)-2:raise ValueError('unexpected shard record')
        rows.extend(actual);calls+=shard[-1]['actual_audit_calls']
    return dict(provenance=provenance[0],actual_audit_calls=calls,
                scope='One fresh training seed on fixed held-out condition pairs; exact endpoints, no endpoint Monte Carlo error; no population significance claim',
                **summarize(rows,panel))


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=collect(args.root,args.stage);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['family_summary'],indent=2))

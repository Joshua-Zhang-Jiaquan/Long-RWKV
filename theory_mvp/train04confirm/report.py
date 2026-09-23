"""Report all six fixed-terminal replicates; no checkpoint or seed selection.

Run as python -m theory_mvp.train04confirm.report after the campaign finishes.
Raw endpoint shards are revalidated before any final report is written.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from .collect import collect,FIELDS,METHODS
from lrwkv_evidence.train04confirm import contract as K

SEEDS=(53,71,89)
ARMS=('independent','complementary')
FAMILIES=('systematic','paired_parity')
ROOT=Path(__file__).resolve().parents[2]


def across_seeds(rows,keys,fields):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    for group,items in sorted(groups.items()):
        if len(items)!=3 or {r['seed'] for r in items}!=set(SEEDS):raise ValueError('incomplete or duplicate seeds in summary')
        out=dict(zip(keys,group))
        for field in fields:
            values=[row[field] for row in sorted(items,key=lambda r:r['seed'])]
            if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite summary values')
            out[field]=dict(seeds=list(SEEDS),values=values,mean=statistics.mean(values),minimum=min(values),maximum=max(values))
        result.append(out)
    return result


def aggregate(reports):
    expected={(s,a) for s in SEEDS for a in ARMS}
    by={(r['provenance']['seed'],r['provenance']['phase']):r for r in reports}
    if len(reports)!=6 or len(by)!=6 or set(by)!=expected:raise ValueError('all six unique fixed replicates required')
    common=('manifest_sha256','panel_sha256','partitions_sha256','step','triton_f32_default','torch_tf32')
    first=reports[0]['provenance']
    if first['step']!=2500 or first['triton_f32_default']!='ieee' or first['torch_tf32']:raise ValueError('wrong terminal or precision')
    if any(any(r['provenance'][k]!=first[k] for k in common) for r in reports):raise ValueError('mixed experiment provenance')
    rows=[];scheduling=[]
    for (seed,arm),report in sorted(by.items()):
        family_keys={(r['family'],r['method']) for r in report['family_summary']}
        if len(report['family_summary'])!=8 or family_keys!={(f,m) for f in FAMILIES for m in METHODS}:raise ValueError('missing family/policy summary')
        for row in report['family_summary']:
            count=16 if row['family']=='paired_parity' else 11
            if row['structures']!=count or row['conditions']!=2*count:raise ValueError('wrong structure weighting')
            if not all(math.isfinite(row[k]) for k in FIELDS):raise ValueError('nonfinite endpoint summary')
            rows.append(dict(seed=seed,history_coupling=arm,**row))
        for family in FAMILIES:
            items=[r for r in report['structure_contrasts'] if r['family']==family]
            count=16 if family=='paired_parity' else 11
            if len(items)!=count or len({tuple(r['structure']) for r in items})!=count:raise ValueError('missing or duplicate structure contrast')
            scheduling.append(dict(seed=seed,history_coupling=arm,family=family,
                random_minus_information_set_kl=statistics.mean(r['random_minus_information_set_kl'] for r in items),
                information_set_minus_random_valid_mass=statistics.mean(r['information_set_minus_random_valid_mass'] for r in items)))
    lookup={(r['seed'],r['history_coupling'],r['family'],r['method']):r for r in rows}
    training=[]
    for seed in SEEDS:
        for family in FAMILIES:
            for method in METHODS:
                a=lookup[seed,'independent',family,method];b=lookup[seed,'complementary',family,method]
                training.append(dict(seed=seed,family=family,method=method,
                    independent_minus_complementary_kl=a['joint_kl_nats']-b['joint_kl_nats'],
                    complementary_minus_independent_valid_mass=b['exact_valid_mass']-a['exact_valid_mass']))
    return dict(provenance={k:first[k] for k in common},per_seed=rows,
        across_seeds=across_seeds(rows,('history_coupling','family','method'),FIELDS),
        training_contrasts_by_seed=training,
        training_contrasts=across_seeds(training,('family','method'),('independent_minus_complementary_kl','complementary_minus_independent_valid_mass')),
        scheduling_contrasts_by_seed=scheduling,
        scheduling_contrasts=across_seeds(scheduling,('history_coupling','family'),('random_minus_information_set_kl','information_set_minus_random_valid_mass')),
        scope='Three data/corruption seeds from the same pretrained initialization; all fixed terminal models retained. Exact fixed-condition endpoints; means and ranges are descriptive, not population significance tests.')


def main():
    folder=ROOT/'results/train04confirm';manifest_path=ROOT/'theory_mvp/train04confirm/FROZEN_RECIPE.json';digest=K.sha(manifest_path)
    reports=[];receipts=[]
    for seed in SEEDS:
        for arm in ARMS:
            receipt=folder/f'plan_eval_{digest[:16]}_{arm}_seed{seed}.json'
            record=json.loads(receipt.read_text());training=json.loads(Path(record['training_plan']).read_text())
            if K.sha(Path(training['out'])/'resume.pt')!=record['checkpoint_sha256']:raise ValueError('terminal checkpoint checksum changed')
            report=collect(Path(record['out']),Path(record['stage']))
            p=report['provenance']
            if p['seed']!=seed or p['phase']!=arm or p['manifest_sha256']!=digest or p['checkpoint_sha256']!=record['checkpoint_sha256']:raise ValueError('receipt and raw evidence differ')
            reports.append(report);receipts.append(dict(path=str(receipt.relative_to(ROOT)),sha256=K.sha(receipt),checkpoint_sha256=p['checkpoint_sha256']))
    result=aggregate(reports);result['receipts']=receipts
    # Preserve complete structure-level results in the final report as well.
    result['replicates']=reports
    dest=folder/'final_report.json';dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Frozen three-seed confirmation results','',result['scope'],'',
           '| Seed | Histories | Family | Policy | KL (nats) | Valid mass | Within-valid KL |',
           '|---|---|---|---|---:|---:|---:|']
    for r in result['per_seed']:
        lines.append(f"| {r['seed']} | {r['history_coupling']} | {r['family']} | {r['method']} | {r['joint_kl_nats']:.8g} | {r['exact_valid_mass']:.8g} | {r['conditional_valid_kl_nats']:.8g} |")
    lines+=['','All per-seed/structure contrasts, D/E components, and across-seed values/ranges are retained in final_report.json. No seed or checkpoint is omitted.']
    (folder/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(dest),replicates=6,per_seed_family_policy_rows=len(result['per_seed']))))


if __name__=='__main__':main()

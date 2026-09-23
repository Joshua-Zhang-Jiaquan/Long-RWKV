"""Fast coverage counts only; final correctness validation uses the collector."""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
plan = json.loads((ROOT/'results/e3_confirmation_plan.json').read_text())
rows=[]
for job in plan['jobs']:
    counts=[]
    for part in job.get('partitions',[{'out':job['out']}]):
        out=Path(part['out'])
        allowed=set(part.get('cell_ids',[]))
        files=out.glob('*/*__*__*.json') if job['model']=='f2' else out.glob('*__*__*.json')
        count=sum(1 for p in files if not allowed or p.name.split('__')[0] in allowed)
        expected=200*len(allowed) if allowed else (16800 if job['length']==16384 else 18000)
        counts.append({'out':str(out),'observed':count,'expected':expected})
    row={'model':job['model'],'length':job['length'], 'observed':sum(p['observed'] for p in counts),
         'expected':sum(p['expected'] for p in counts),'partitions':counts}
    rows.append(row)
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'meaning':'filename coverage only; no accuracy or validation claim','jobs':rows}
(ROOT/'results/e3_progress.json').write_text(json.dumps(result,indent=2)+'\n')
for r in rows:
    print(f"{r['model']:2} {r['length']:5}: {r['observed']:5}/{r['expected']} ({100*r['observed']/r['expected']:.1f}%)")

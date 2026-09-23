"""Publish only fully covered individual model/length panels during execution."""
import json
from collections import defaultdict
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lrwkv_evidence.e3.collect_confirmation import collect
from lrwkv_evidence.parallel_io import read_json_parallel
from lrwkv_evidence.e3 import grid324 as G

plan=json.loads((ROOT/'results/e3_confirmation_plan.json').read_text())
coverage=collect(plan,status_only=True)
completed=[]
for job in plan['jobs']:
    length=job['length']
    validated = [p for p in coverage['partitions'] if p['model'].lower() == job['model'].lower() and p['length'] == length]
    if any(p['used_receipts'] != 200 * p['assigned_cells'] for p in validated):
        continue
    expected_cells={c.cell_id for c in G.all_cells() if c.length==length
                    and json.loads((G.OUT_ROOT/f'{c.cell_id}.json').read_text())['status']=='ok'}
    rows=[]
    for part in job.get('partitions',[{'out':job['out']}]):
        allowed=set(part.get('cell_ids',expected_cells))
        out=Path(part['out'])
        paths=out.glob('*/*__*__*.json') if job['model']=='f2' else out.glob('*__*__*.json')
        accepted = (p for p in paths if p.name.split('__')[0] in allowed)
        rows.extend(row for _, _, row in read_json_parallel(accepted))
    groups=defaultdict(list)
    for row in rows:
        groups[row['cell_id']].append(row)
    if set(groups)!=expected_cells or any(len(rs)!=200 for rs in groups.values()):
        continue
    keys={(r['cell_id'],r['data_seed'],r['instance_index']) for r in rows}
    if len(keys)!=len(rows):
        raise ValueError('duplicate completed-panel receipt')
    family_scores=[]
    family_rows=[]
    for family in G.PAPER_FAMILIES:
        cells=[rs for cell,rs in groups.items() if cell.startswith(family+'_L')]
        selected=[r for rs in cells for r in rs]
        score=statistics.mean(statistics.mean(r['correct'] for r in rs) for rs in cells)
        family_scores.append(score)
        family_rows.append({'family':family,'cells':len(cells),'n':len(selected),
                            'correct':sum(r['correct'] for r in selected),
                            'parsed':sum(r['parse_ok'] for r in selected),'accuracy':score})
    completed.append({'model':job['model'],'length':length,'n':len(rows),
                      'family_macro_accuracy':statistics.mean(family_scores),'families':family_rows})
result={'scope':'Complete individual model/length panels only; not the completed paired three-length endpoint.',
        'complete_overall':coverage['complete'],'panels':completed}
(ROOT/'results/e3_completed_panels.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

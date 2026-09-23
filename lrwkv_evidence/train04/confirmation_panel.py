"""Predeclare fresh condition pairs; no model access or outcome-based selection."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from . import tasks

SEED=92673193
PER_STRUCTURE=2


def condition_key(ex):
    return ex['family'],tuple(sorted(zip(ex['matrix_rows'],ex['syndrome'])))


def build():
    # Includes primary sampling/joint panels, calibration, exact follow-up and runtime probe.
    prior=[tasks.make_example('test',73191,i) for i in range(32)]
    prior += [tasks.make_example('test',73192,i) for i in range(128)]
    excluded={condition_key(e) for e in prior}
    selected=defaultdict(list);seen=set();records=[]
    catalogs={family:set(tasks.split_catalog('test',family)) for family in tasks.FAMILIES}
    total=sum(len(v) for v in catalogs.values())*PER_STRUCTURE
    for index in range(100000):
        ex=tasks.make_example('test',SEED,index)
        structure=tuple(sorted(ex['matrix_rows']));key=condition_key(ex);group=(ex['family'],structure)
        if key in excluded or key in seen or len(selected[group])>=PER_STRUCTURE:continue
        assert structure in catalogs[ex['family']]
        selected[group].append(ex);seen.add(key)
        records.append(dict(index=index,family=ex['family'],structure=list(structure),condition_key=[list(x) for x in key[1]],
                            instance_id=ex['instance_id'],prompt_sha256=hashlib.sha256(ex['prompt'].encode()).hexdigest()))
        if len(records)==total:break
    assert len(records)==54 and len(selected)==27
    assert all(len(v)==PER_STRUCTURE for v in selected.values())
    assert not (seen & excluded)
    return dict(version=1,split='test',seed=SEED,per_structure=PER_STRUCTURE,
                scope='Fresh syndrome/structure pairs within existing held-out canonical structure catalog; not new algebraic topologies',
                selection='First eligible indices in fixed generator stream; exclude both primary test namespaces; no model evaluation',
                previous_unique_condition_pairs=len(excluded),records=records,
                counts={f:sum(r['family']==f for r in records) for f in tasks.FAMILIES})


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=build();payload=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if a.out.exists() and a.out.read_text()!=payload:raise ValueError('frozen panel differs')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(payload)
    print(json.dumps(dict(counts=result['counts'],previous_unique_condition_pairs=result['previous_unique_condition_pairs'],sha256=hashlib.sha256(payload.encode()).hexdigest())))
if __name__=='__main__':main()

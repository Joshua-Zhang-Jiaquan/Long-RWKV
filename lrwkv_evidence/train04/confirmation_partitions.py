"""Freeze reveal groups from the fixed confirmation inputs, without model access."""
import argparse
import hashlib
import json
from pathlib import Path
from . import tasks
from .evaluate import fixed_groups

PANEL_SHA='146737f87dba9abe14a4d6cab77cb191cc2507fb132a36276405975d221f6766'
METHODS=('one','information_set','random_halves','sequential')


def build(panel_path):
    raw=panel_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PANEL_SHA:raise ValueError('confirmation panel changed')
    panel=json.loads(raw);records=[]
    for record in panel['records']:
        ex=tasks.make_example(panel['split'],panel['seed'],record['index'])
        if ex['instance_id']!=record['instance_id'] or hashlib.sha256(ex['prompt'].encode()).hexdigest()!=record['prompt_sha256']:
            raise ValueError('confirmation generator changed')
        groups={method:fixed_groups(ex,method) for method in METHODS}
        for method,partition in groups.items():
            flat=[i for group in partition for i in group]
            if sorted(flat)!=list(range(8)) or len(flat)!=8:raise ValueError('invalid reveal partition')
            sizes=[len(group) for group in partition]
            expected={'one':[8],'information_set':[4,4],'random_halves':[4,4],'sequential':[1]*8}[method]
            if sizes!=expected:raise ValueError('wrong call budget')
        records.append(dict(index=record['index'],instance_id=record['instance_id'],family=record['family'],structure=record['structure'],groups=groups))
    return dict(version=1,panel_sha256=PANEL_SHA,seed=panel['seed'],records=records,
                scope='Input-derived reveal groups shared across all confirmation checkpoints; no neural outputs or solved bit values')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--panel',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();payload=json.dumps(build(args.panel),sort_keys=True,indent=2)+'\n'
    if args.out.exists() and args.out.read_text()!=payload:raise ValueError('frozen partitions differ')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(payload)
    print(json.dumps(dict(conditions=54,methods=list(METHODS),sha256=hashlib.sha256(payload.encode()).hexdigest())))


if __name__=='__main__':main()

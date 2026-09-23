"""Calibration first; target execution requires a complete prior prediction seal."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import campaign as C
import emit_jobs as E
from confirm_capacity import live_usage,route_body,PRIMARY
from submission_lock import campaign_submission_lock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.train04confirm.contract import sha
from lrwkv_evidence.train04.worker import canonical_sha
COMMON=ROOT.parent/'DiffRWKV-RELAY/long_rwkv_iclr2027'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--split',choices=('calibration','heldout'),required=True);ap.add_argument('--submit',action='store_true');args=ap.parse_args()
    folder=ROOT/'results/predictive_transfer';train=json.loads((folder/f'plan_seed{args.seed}.json').read_text())
    done=json.loads((Path(train['out'])/'completion.json').read_text())
    checkpoint=Path(train['out'])/'resume.pt'
    if not done.get('execution_complete') or done.get('qualification') or done['step']!=3100 or sha(checkpoint)!=done['checkpoint_sha256']:
        raise ValueError('terminal checkpoint required')
    protocol=ROOT/'revision/transfer/FROZEN.json';frozen=json.loads(protocol.read_text())
    sources={rel:Path(train['stage'])/rel for rel in train['sources']}
    for rel,p in sources.items():
        if sha(p)!=train['sources'][rel]:raise ValueError('training stage changed')
    for name in ('prediction.py','evaluate.py'):
        rel='lrwkv_evidence/predictive_transfer/'+name;sources[rel]=ROOT/rel
    for rel,digest in frozen['scientific_sources'].items():
        if sha(sources[rel])!=digest:raise ValueError('frozen science changed')
    sources['qz/launch_transfer_eval.sh']=ROOT/'qz/launch_transfer_eval.sh'
    hashes={rel:sha(p) for rel,p in sources.items()};digest=canonical_sha(hashes)[:16]
    stage=Path(E.G)/'long_rwkv_transfer_eval_stages'/digest
    out=Path(E.OUTPUTS)/f'predictive_transfer_eval_{digest}_{args.split}_seed{args.seed}'
    shared_protocol=Path(E.G)/'long_rwkv_predictive_transfer_protocols'/f'{sha(protocol)}.json'
    if sha(shared_protocol)!=sha(protocol):raise ValueError('shared protocol mismatch')
    env=dict(TRANSFER_ROOT=str(stage),TRANSFER_OUT=str(out),TRANSFER_SPLIT=args.split,TRANSFER_CHECKPOINT=str(checkpoint),
             TRANSFER_CHECKPOINT_SHA256=done['checkpoint_sha256'],TRANSFER_PROTOCOL=str(shared_protocol),TRITON_F32_DEFAULT='ieee',
             TRITON_CACHE_DIR=str(Path(E.G)/'long_rwkv_distance_eval_cache_seed/7d78deca71289430'))
    if args.split=='heldout':
        seal=json.loads((folder/'PREDICTION_SEAL.json').read_text())
        if seal['protocol_sha256']!=sha(protocol) or set(seal['predictions'])!=set(map(str,frozen['training_seeds'])):raise ValueError('incomplete seal')
        for rec in seal['predictions'].values():
            if sha(ROOT/rec['path'])!=rec['sha256']:raise ValueError('prediction changed after sealing')
        rec=seal['predictions'][str(args.seed)];pred=ROOT/rec['path']
        target=Path(E.G)/'long_rwkv_transfer_predictions'/f"{rec['sha256']}.json"
        if args.submit:
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(pred,target)
            if sha(target)!=rec['sha256']:raise ValueError('shared prediction conflict')
        env.update(TRANSFER_PREDICTIONS=str(target),TRANSFER_PREDICTIONS_SHA256=rec['sha256'])
    body=E.make_body(f'lrwkv-transfer-eval-{digest[:8]}-{args.split}-{args.seed}',E.wrapped(env,str(stage/'qz/launch_transfer_eval.sh')),
                    'Exact two-basis posterior evaluation; source calibration or previously sealed target',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='14400000')
    record=dict(stage=str(stage),out=str(out),sources=hashes,body=body,seed=args.seed,split=args.split,checkpoint_sha256=done['checkpoint_sha256'])
    dest=folder/f'plan_{args.split}_seed{args.seed}.json'
    if dest.exists() and json.loads(dest.read_text()).get('submission'):raise ValueError('duplicate plan')
    if args.submit:
        with campaign_submission_lock(COMMON):
            record['capacity']=live_usage(C.list_all_jobs());record['budget_project_id']=route_body(body,record['capacity'],PRIMARY)
            if C.already_submitted(body['name']):raise ValueError('duplicate name')
            for rel,p in sources.items():
                target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and sha(target)!=hashes[rel]:raise ValueError('stage conflict')
                if not target.exists():shutil.copyfile(p,target)
            manifest=stage/'transfer_eval_sources.json';data=json.dumps(hashes,indent=2)+'\n'
            if manifest.exists() and manifest.read_text()!=data:raise ValueError('manifest conflict')
            if not manifest.exists():manifest.write_text(data)
            record['submission']=C.submit_one(body,dry_run=False)
    dest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))
if __name__=='__main__':main()

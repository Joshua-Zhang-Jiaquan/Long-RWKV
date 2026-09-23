"""Stage immutable source and admit one eight-H100 revision job."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--qualification',action='store_true');ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--submit',action='store_true');args=ap.parse_args()
    prior=json.loads((ROOT/'results/distance_intervention/plan_qualification.json').read_text())
    sources={rel:Path(prior['stage'])/rel for rel in prior['sources']}
    for rel,p in sources.items():
        if sha(p)!=prior['sources'][rel]:raise ValueError('old source changed '+rel)
    sources.update({str(p.relative_to(ROOT)):p for p in (ROOT/'lrwkv_evidence/predictive_transfer').glob('*.py') if p.name in ('__init__.py','algebra.py','tasks.py','training.py','worker.py')})
    sources['qz/launch_predictive_transfer.sh']=ROOT/'qz/launch_predictive_transfer.sh'
    hashes={rel:sha(p) for rel,p in sources.items()};source_sha=canonical_sha(hashes);digest=source_sha[:16]
    stage=Path(E.G)/'long_rwkv_predictive_transfer_stages'/digest
    role='qualification' if args.qualification else f'seed{args.seed}'
    out=Path(E.OUTPUTS)/f'predictive_transfer_{digest}_{role}'
    env=dict(TRANSFER_ROOT=str(stage),TRANSFER_OUT=str(out),TRANSFER_SEED=str(args.seed),
             TRANSFER_QUALIFICATION='1' if args.qualification else '0',TRITON_F32_DEFAULT='ieee',
             TRITON_CACHE_DIR=str(Path(E.G)/'long_rwkv_distance_eval_cache_seed/7d78deca71289430'))
    folder=ROOT/'results/predictive_transfer';folder.mkdir(parents=True,exist_ok=True)
    if not args.qualification:
        q=json.loads((folder/'plan_qualification.json').read_text())
        if q['source_sha256']!=source_sha:raise ValueError('qualification source changed')
        done=json.loads((Path(q['out'])/'completion.json').read_text())
        if not done.get('execution_complete') or not done.get('qualification'):raise ValueError('qualification incomplete')
        protocol=ROOT/'revision/transfer/FROZEN.json'
        target=Path(E.G)/'long_rwkv_predictive_transfer_protocols'/f'{sha(protocol)}.json'
        if args.submit:
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha(target)!=sha(protocol):raise ValueError('protocol collision')
            if not target.exists():shutil.copyfile(protocol,target)
        env.update(TRANSFER_PROTOCOL=str(target),TRANSFER_QUALIFICATION_OUT=q['out'])
    body=E.make_body(f'lrwkv-transfer-{digest[:8]}-{role}',E.wrapped(env,str(stage/'qz/launch_predictive_transfer.sh')),
                    'Prospective code-class transfer; fresh public-base lineage; eight H100 under 32+16 cap',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='86400000')
    record=dict(stage=str(stage),out=str(out),source_sha256=source_sha,sources=hashes,body=body,seed=args.seed,qualification=args.qualification)
    dest=folder/f'plan_{role}.json'
    if dest.exists() and json.loads(dest.read_text()).get('submission'):raise ValueError('duplicate plan')
    if args.submit:
        with campaign_submission_lock(COMMON):
            record['capacity']=live_usage(C.list_all_jobs());record['budget_project_id']=route_body(body,record['capacity'],PRIMARY)
            if C.already_submitted(body['name']):raise ValueError('duplicate scheduler name')
            for rel,p in sources.items():
                if sha(p)!=hashes[rel]:raise ValueError('source changed during staging')
                target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and sha(target)!=hashes[rel]:raise ValueError('immutable stage collision')
                if not target.exists():shutil.copyfile(p,target)
            target=stage/'transfer_sources.json';text=json.dumps(hashes,indent=2)+'\n'
            if target.exists() and target.read_text()!=text:raise ValueError('source manifest collision')
            if not target.exists():target.write_text(text)
            record['submission']=C.submit_one(body,dry_run=False)
    dest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))

if __name__=='__main__':main()

"""Stage a development audit against the exact sources of a completed training job."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import campaign as C
import emit_jobs as E
from submit_mvp_lookup import live_usage
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training-plan',type=Path,required=True);ap.add_argument('--triton-precision',choices=['default','ieee'],default='default');ap.add_argument('--numerical-qualification',action='store_true');ap.add_argument('--submit',action='store_true');a=ap.parse_args()
    plan=json.loads(a.training_plan.read_text());source=Path(plan['stage']);train=Path(plan['out']);checkpoint=train/'resume.pt'
    done=json.loads((train/'completion.json').read_text())
    if not done['execution_complete'] or sha(checkpoint)!=done['checkpoint_sha256']:raise ValueError('checkpoint incomplete/mismatched')
    files={rel:source/rel for rel in plan['sources']}
    for rel,p in files.items():
        if sha(p)!=plan['sources'][rel]:raise ValueError(f'training source changed:{rel}')
    for rel in ('lrwkv_evidence/train04dev/evaluate.py','lrwkv_evidence/train04dev/history_response.py','lrwkv_evidence/train04dev/numerical_qualification.py','lrwkv_evidence/train04/evaluate.py','qz/launch_train04dev_eval.sh'):
        files[rel]=ROOT/rel
    hashes={rel:sha(p) for rel,p in files.items()};digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    stage=Path(E.G)/'long_rwkv_train04dev_eval_stages'/digest
    tag=f"{plan['objective']}_s{done['step']}_{done['checkpoint_sha256'][:8]}"
    if a.triton_precision!='default':tag+='_'+a.triton_precision
    if a.numerical_qualification:tag+='_numerical'
    out=Path(E.OUTPUTS)/f'train04dev_eval_{digest}_{tag}'
    env={'DEV_EVAL_ROOT':str(stage),'DEV_EVAL_OUT':str(out),'DEV_CHECKPOINT':str(checkpoint)}
    if a.numerical_qualification:env['DEV_NUMERICAL_QUALIFICATION']='1'
    if a.triton_precision=='ieee':env.update(TRITON_F32_DEFAULT='ieee',TRITON_CACHE_DIR=str(out/'triton_ieee_cache'))
    body=E.make_body(f"lrwkv-dev04-eval-{digest[:8]}-{tag}",E.wrapped(env,str(stage/'qz/launch_train04dev_eval.sh')),'Development-only exact endpoint accounting N2/N4/N8;8H100 under32cap',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='7200000')
    record=dict(numerical_qualification=a.numerical_qualification,triton_precision=a.triton_precision,stage=str(stage),out=str(out),training_plan=str(a.training_plan.resolve()),checkpoint_sha256=done['checkpoint_sha256'],sources=hashes,body=body)
    if a.submit:
        record['capacity']=live_usage(C.list_all_jobs())
        if record['capacity']['live_reserved_gpus']+8>32:raise SystemExit('32GPU cap')
        if C.already_submitted(body['name']):raise SystemExit('already submitted')
        for rel,p in files.items():
            if sha(p)!=hashes[rel]:raise ValueError('source changed during staging')
            dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and sha(dest)!=hashes[rel]:raise ValueError('immutable stage conflict')
            if not dest.exists():shutil.copyfile(p,dest)
        record['submission']=C.submit_one(body,dry_run=False)
    dest=ROOT/'results/train04dev'/f'plan_eval_{digest}_{tag}.json';dest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))
if __name__=='__main__':
    from submission_lock import campaign_submission_lock
    with campaign_submission_lock(ROOT):main()

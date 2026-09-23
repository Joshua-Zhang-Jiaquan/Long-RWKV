"""Submit only frozen terminal confirmation checkpoints under the shared cap."""
import argparse
import json
from pathlib import Path
import sys
import campaign as C
import emit_jobs as E
from confirm_capacity import live_usage, CAP, route_body
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.train04confirm import contract as K


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training-plan',type=Path,required=True);parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    plan=json.loads(args.training_plan.read_text());stage=Path(plan['stage']);train=Path(plan['out'])
    manifest,digest=K.validate(stage/'manifest.json',stage,stage/'model_source')
    done=json.loads((train/'completion.json').read_text());checkpoint=train/'resume.pt'
    if plan['qualification'] or done.get('qualification') or not done['execution_complete'] or done['step']!=2500 or plan['phase'] not in ('independent','complementary') or plan['seed'] not in (53,71,89) or plan['manifest_sha256']!=digest or K.sha(checkpoint)!=done['checkpoint_sha256']:
        raise ValueError('not a completed frozen terminal replicate')
    tag=f"{digest[:16]}_{plan['phase']}_seed{plan['seed']}"
    out=Path(E.OUTPUTS)/f'train04confirm_eval_{tag}'
    env=dict(CONFIRM_ROOT=str(stage),CONFIRM_EVAL_OUT=str(out),CONFIRM_CHECKPOINT=str(checkpoint),
             TRITON_F32_DEFAULT='ieee',TRITON_CACHE_DIR=str(out/'triton_ieee_cache'))
    body=E.make_body('lrwkv-confirm04-eval-'+tag,E.wrapped(env,str(stage/'qz/launch_train04confirm_eval.sh')),
                     'Frozen N8 confirmation;54conditions,4policies;IEEE;8H100 under48cap',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='10800000')
    record=dict(stage=str(stage),out=str(out),training_plan=str(args.training_plan.resolve()),manifest_sha256=digest,
                checkpoint_sha256=done['checkpoint_sha256'],seed=plan['seed'],phase=plan['phase'],body=body)
    dest=ROOT/'results/train04confirm'/f'plan_eval_{tag}.json'
    if dest.exists() and json.loads(dest.read_text()).get('submission'):raise ValueError('evaluation already submitted')
    if args.submit:
        record['capacity']=live_usage(C.list_all_jobs())
        record['budget_project_id']=route_body(body,record['capacity'])
        if C.already_submitted(body['name']):raise SystemExit('already submitted')
        record['submission']=C.submit_one(body,dry_run=False)
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='body'},indent=2))


if __name__=='__main__':
    from submission_lock import campaign_submission_lock
    with campaign_submission_lock(ROOT):main()

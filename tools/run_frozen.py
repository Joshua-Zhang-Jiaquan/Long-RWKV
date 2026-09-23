"""Launch unchanged evaluation code with caller-supplied local model paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
NAMES={'distance_intervention':('distance_eval_sources.json','distance_eval_manifest.json'),
       'history_response':('history_response_sources.json','history_response_manifest.json'),
       'header_distance':('header_distance_sources.json','header_distance_manifest.json')}
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',choices=NAMES,required=True);ap.add_argument('--role',required=True)
    for name in ('base','checkpoint','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--execute',action='store_true',help='Run on eight local GPUs; otherwise validate and print the command.')
    args=ap.parse_args();stage=ROOT/'reproduction/stages'/args.study
    sources,manifest=NAMES[args.study]
    for rel,h in json.loads((stage/sources).read_text()).items():
        if digest(stage/rel)!=h:raise ValueError('frozen stage changed: '+rel)
    design=json.loads((stage/manifest).read_text())
    # Original checkpoint is permitted only by the two preceding frozen panels.
    if args.role=='original' and args.study!='header_distance':expected=design['initial_checkpoint_sha256']
    elif 'checkpoints' in design:expected=design['checkpoints'][args.role]['sha256']
    else:
        plan=json.loads((ROOT/f'results/distance_intervention/plan_eval_{args.role}.json').read_text());expected=plan['checkpoint_sha256']
    if digest(args.checkpoint)!=expected:raise ValueError('checkpoint does not match frozen selector')
    if not args.base.is_dir():raise ValueError('native base model/tokenizer directory missing')
    if args.out.exists():raise ValueError('output must be a new directory')
    env=os.environ.copy();env.update(DISTANCE_ROLE=args.role,DISTANCE_CHECKPOINT_SHA256=expected,
      DISTANCE_EXECUTION=str(ROOT/'results/distance_intervention/FROZEN_EXECUTION.json'),
      TRITON_F32_DEFAULT='ieee',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    command=['torchrun','--standalone','--nproc_per_node=8','-m',f'lrwkv_evidence.{args.study}.evaluate',
       '--base',str(args.base.resolve()),'--model-root',str(stage/'model_source'),
       '--checkpoint',str(args.checkpoint.resolve()),'--out',str(args.out.resolve())]
    print(json.dumps(dict(cwd=str(stage),command=command,checkpoint_sha256=expected,execute=args.execute),indent=2),flush=True)
    if args.execute:subprocess.run(command,cwd=stage,env=env,check=True)
if __name__=='__main__':main()

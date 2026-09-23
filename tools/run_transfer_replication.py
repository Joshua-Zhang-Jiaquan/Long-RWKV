"""Portable full prospective workflow, using unchanged frozen scientific sources.

Dry-run by default. Uses eight local GPUs; submits no scheduler jobs. Analysis
path bindings are relocated into a separate workspace, preserving the original
protocol, functions, panels, choices and historical evidence.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def stage(kind):
    capsule=json.loads((ROOT/f'revision/transfer/{kind.upper()}_CAPSULE.json').read_text())
    path=ROOT/f'reproduction/stages/predictive_transfer_{kind}'
    for rel,digest in capsule['files'].items():
        if sha(path/rel)!=digest:raise ValueError('portable source mismatch '+rel)
    return path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('qualification','train','calibration','seal','heldout','report'),required=True)
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--base',type=Path)
    ap.add_argument('--seed',type=int);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    work=args.work.resolve();folder=work/'results/predictive_transfer';protocol_source=ROOT/'revision/transfer/FROZEN.json'
    protocol=json.loads(protocol_source.read_text())
    for field in ('scientific_sources','analysis_sources'):
        for rel,digest in protocol[field].items():
            if sha(ROOT/rel)!=digest:raise ValueError('frozen scientific source changed '+rel)
    local_protocol=work/'revision/transfer/FROZEN.json'
    if local_protocol.exists():
        if sha(local_protocol)!=sha(protocol_source):raise ValueError('replication protocol conflict')
    elif args.execute:
        local_protocol.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(protocol_source,local_protocol)
    if args.phase in ('seal','report'):
        print(json.dumps(dict(action=args.phase,work=str(work),execute=args.execute)),flush=True)
        if args.execute:
            from revision.transfer import analyze
            analyze.ROOT=work;analyze.FOLDER=folder
            if args.phase=='seal':analyze.seal()
            else:
                analyze.report()
                from tools import verify_transfer_evidence as audit
                audit.ROOT=work
                sys.argv=['verify_transfer_evidence','--folder',str(folder),'--out',str(folder/'independent_audit.json')]
                audit.main()
        return
    if args.base is None:raise ValueError('--base required for GPU phases')
    base=args.base.resolve();assets=json.loads((ROOT/'results/predictive_transfer/public_base_resolution.json').read_text())
    expected={r['file']:r['local_sha256'] for r in assets['files']};expected['model.safetensors']=assets['weight_sha256']
    for name,digest in expected.items():
        if sha(base/name)!=digest:raise ValueError('wrong public base asset '+name)
    qualification=args.phase=='qualification'
    seed=202709230 if qualification else args.seed
    if not qualification and seed not in protocol['training_seeds']:raise ValueError('registered training seed required')
    training=args.phase in ('qualification','train');runtime=stage('training' if training else 'evaluation')
    out=work/'runs'/('qualification' if qualification else f'{args.phase}_seed{seed}')
    if out.exists():raise ValueError('refuse existing output directory')
    env=dict(TRITON_F32_DEFAULT='ieee',TRITON_CACHE_DIR=str(work/'triton_ieee_cache'),CUBLAS_WORKSPACE_CONFIG=':4096:8',
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',TRANSFER_PROTOCOL=str(local_protocol))
    common=['--base',str(base),'--model-root',str(runtime/'model_source'),'--out',str(out)]
    if training:
        module='lrwkv_evidence.predictive_transfer.worker';common+=['--seed',str(seed)]
        if qualification:common+=['--qualification']
        else:common+=['--qualification-out',str(work/'runs/qualification')]
        record=dict(out=str(out),seed=seed,qualification=qualification,source_sha256=protocol['training_source_sha256'])
        plan_path=folder/('plan_qualification.json' if qualification else f'plan_seed{seed}.json')
    else:
        training_plan=json.loads((folder/f'plan_seed{seed}.json').read_text());train_out=Path(training_plan['out'])
        done=json.loads((train_out/'completion.json').read_text());checkpoint=train_out/'resume.pt'
        if not done.get('execution_complete') or done.get('qualification') or done['step']!=3100 or done['seed']!=seed:
            raise ValueError('fixed terminal checkpoint required')
        if sha(checkpoint)!=done['checkpoint_sha256']:raise ValueError('checkpoint changed')
        module='lrwkv_evidence.predictive_transfer.evaluate';common+=['--checkpoint',str(checkpoint),'--split',args.phase]
        env['TRANSFER_CHECKPOINT_SHA256']=done['checkpoint_sha256']
        if args.phase=='heldout':
            seal=json.loads((folder/'PREDICTION_SEAL.json').read_text())
            if set(seal['predictions'])!=set(map(str,protocol['training_seeds'])) or seal['protocol_sha256']!=sha(protocol_source):raise ValueError('complete prediction seal required')
            for pred in seal['predictions'].values():
                if sha(work/pred['path'])!=pred['sha256']:raise ValueError('prediction seal changed')
            pred=seal['predictions'][str(seed)]
            env.update(TRANSFER_PREDICTIONS=str(work/pred['path']),TRANSFER_PREDICTIONS_SHA256=pred['sha256'])
        record=dict(out=str(out),seed=seed,split=args.phase,checkpoint_sha256=done['checkpoint_sha256'])
        plan_path=folder/f'plan_{args.phase}_seed{seed}.json'
    command=['torchrun','--standalone','--nproc_per_node=8','-m',module,*common]
    print(json.dumps(dict(cwd=str(runtime),command=command,environment_overrides=env,execute=args.execute),indent=2),flush=True)
    if args.execute:
        if plan_path.exists():raise ValueError('refuse existing replication plan')
        write(plan_path,dict(**record,command=command,environment=env,status='execution_requested'))
        subprocess.run(command,cwd=runtime,env={**os.environ,**env},check=True)
        write(plan_path,dict(**record,command=command,environment=env,status='process_succeeded'))
if __name__=='__main__':main()

"""Reconstruct the original selected lineage from public weights on eight local GPUs.

Dry-run by default. Reconstructed checkpoints receive new identities; historical
hashes/results are never relabeled as newly reproduced artifacts.
"""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE_SHA='e162387e439dfa3387a0ca7da61638749d00c9862b8cc0192ae5d366c8c1a524'
ROLES=[f'{arm}_seed{seed}' for seed in (20271011,20271012,20271013) for arm in ('near','balanced')]
NAMES={'distance_intervention':('distance_eval_sources.json','distance_eval_manifest.json'),
       'history_response':('history_response_sources.json','history_response_manifest.json'),
       'header_distance':('header_distance_sources.json','header_distance_manifest.json')}

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def verify(stage,manifest):
    for rel,h in json.loads((stage/manifest).read_text()).items():
        if sha(stage/rel)!=h:raise ValueError('source mismatch '+rel)

def copy_stage(work,name,source,manifest):
    verify(source,manifest);target=work/name
    if target.exists():raise ValueError('refuse reused stage '+str(target))
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    return target

def launch(stage,module,args,env,execute):
    command=['torchrun','--standalone','--nproc_per_node=8','-m',module,*map(str,args)]
    print(json.dumps(dict(cwd=str(stage),command=command,environment_overrides=env,execute=execute),indent=2),flush=True)
    if execute:
        full=os.environ.copy();full.pop('TRITON_F32_DEFAULT',None);full.update(env)
        subprocess.run(command,cwd=stage,env=full,check=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('initial','adaptations','evaluation'),required=True)
    ap.add_argument('--base',type=Path,required=True);ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--initial-checkpoint',type=Path);ap.add_argument('--study',choices=NAMES,default='header_distance')
    ap.add_argument('--role',choices=['original',*ROLES]);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    base=args.base.resolve();work=args.work.resolve();work.mkdir(parents=True,exist_ok=True)
    if sha(base/'model.safetensors')!=BASE_SHA:raise ValueError('wrong public base weights')
    env=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    historical=ROOT/'reproduction/stages/training'
    if args.phase=='initial':
        stage=copy_stage(work,'initial_stage',historical,'distance_sources.json')
        common=['--manifest',stage/'manifest.json','--base',base,'--model-root',stage/'model_source','--seed',71]
        for phase in ('parent','independent'):
            output=work/phase
            if output.exists():raise ValueError('output exists')
            extra=[] if phase=='parent' else ['--parent',work/'parent/resume.pt']
            launch(stage,'lrwkv_evidence.train04confirm.worker',common+['--phase',phase,'--out',output]+extra,env,args.execute)
        write(work/'initial_workflow.json',dict(status='executed' if args.execute else 'dry_run',base_sha256=BASE_SHA,
              stages=['parent1500','independent2500'],manifest_sha256=sha(stage/'manifest.json'),
              historical_initial_sha256='52dbc7f289513e475d10422dbccffa52f74ec55fc887253ea13ae0476dc3e2c9',
              reconstructed_sha256=sha(work/'independent/resume.pt') if args.execute else None))
        return
    initial=(args.initial_checkpoint or work/'independent/resume.pt').resolve()
    initial_sha=sha(initial)
    if args.phase=='adaptations':
        import torch
        payload=torch.load(initial,map_location='cpu',weights_only=False,mmap=True)
        if payload['step']!=2500 or payload['contract'].get('seed')!=71 or payload['contract'].get('phase')!='independent':raise ValueError('wrong initial lineage')
        if canonical(payload['contract'])!=payload['contract_sha256']:raise ValueError('corrupt initial contract')
        del payload
        stage=copy_stage(work,'adaptation_stage',historical,'distance_sources.json')
        design=json.loads((stage/'distance_design.json').read_text());design['initial_checkpoint_sha256']=initial_sha
        write(stage/'distance_design.json',design)
        sources=json.loads((stage/'distance_sources.json').read_text());sources['distance_design.json']=sha(stage/'distance_design.json')
        write(stage/'distance_sources.json',sources);source_sha=canonical(sources)
        execution=json.loads((ROOT/'results/distance_intervention/FROZEN_EXECUTION.json').read_text())
        execution.update(source_sha256=source_sha,replication='new checkpoint identity; same600update science; historical metrics not reused')
        write(work/'replication_execution.json',execution)
        env.update(TRITON_F32_DEFAULT='ieee',DISTANCE_EXECUTION=str(work/'replication_execution.json'))
        common=['--base',base,'--model-root',stage/'model_source','--initial-checkpoint',initial,'--initial-sha256',initial_sha]
        qualification=work/'adaptation_qualification'
        launch(stage,'lrwkv_evidence.distance_intervention.worker',common+['--out',qualification,'--arm','balanced','--seed',20271011,'--steps',6,'--qualification'],env,args.execute)
        checkpoints={'original':dict(path=str(initial),sha256=initial_sha)}
        for role in ROLES:
            arm,seed=role.split('_seed');output=work/role
            launch(stage,'lrwkv_evidence.distance_intervention.worker',common+['--out',output,'--arm',arm,'--seed',seed,'--steps',600,'--qualification-out',qualification],env,args.execute)
            checkpoints[role]=dict(path=str(output/'resume.pt'),sha256=sha(output/'resume.pt') if args.execute else None)
        write(work/'replication_checkpoints.json',dict(status='executed' if args.execute else 'dry_run',checkpoints=checkpoints,
              training_source_sha256=source_sha,initial_sha256=initial_sha))
        return
    if not args.role or (args.study=='header_distance' and args.role=='original'):raise ValueError('valid study role required')
    index=json.loads((work/'replication_checkpoints.json').read_text())
    if index['status']!='executed':raise ValueError('adaptations have not run')
    for rec in index['checkpoints'].values():
        if sha(rec['path'])!=rec['sha256']:raise ValueError('reconstructed checkpoint changed')
    source_name,manifest_name=NAMES[args.study]
    stage=copy_stage(work,f'eval_{args.study}_{args.role}',ROOT/'reproduction/stages'/args.study,source_name)
    manifest=json.loads((stage/manifest_name).read_text());manifest['initial_checkpoint_sha256']=index['initial_sha256']
    if 'checkpoints' in manifest:
        for role in manifest['checkpoints']:
            manifest['checkpoints'][role].update(index['checkpoints'][role])
    manifest['replication_notice']='New reconstructed checkpoint identities; historical prediction artifacts are not used.'
    write(stage/manifest_name,manifest)
    sources=json.loads((stage/source_name).read_text());sources[manifest_name]=sha(stage/manifest_name);write(stage/source_name,sources)
    verify(stage,source_name)
    rec=index['checkpoints'][args.role]
    env.update(TRITON_F32_DEFAULT='ieee',DISTANCE_ROLE=args.role,DISTANCE_CHECKPOINT_SHA256=rec['sha256'],DISTANCE_EXECUTION=str(work/'replication_execution.json'))
    launch(stage,'lrwkv_evidence.'+args.study+'.evaluate',['--base',base,'--model-root',stage/'model_source','--checkpoint',rec['path'],
            '--out',work/f'results_{args.study}_{args.role}'],env,args.execute)

if __name__=='__main__':main()

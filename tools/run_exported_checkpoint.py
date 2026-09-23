"""Run an archived evaluation using a tensor-identical inference export.

Builds a separate selector manifest for the export's container hash. Frozen
scientific code and task panels remain byte-identical; original artifacts are
never overwritten. The export is for inference, not optimizer-state resumption.
"""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES={'distance_intervention':('distance_eval_sources.json','distance_eval_manifest.json'),
       'history_response':('history_response_sources.json','history_response_manifest.json'),
       'header_distance':('header_distance_sources.json','header_distance_manifest.json')}

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',choices=NAMES,required=True);ap.add_argument('--role',required=True)
    for name in ('base','checkpoint','work'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    registry=json.loads((ROOT/'models_release/EXPORTS.json').read_text());record=registry['exports'][args.role]
    if args.study=='header_distance' and args.role=='original':raise ValueError('header panel has no original selector')
    if sha(args.checkpoint)!=record['export_sha256']:raise ValueError('inference export checksum mismatch')
    import torch
    from export_inference_checkpoints import state_sha
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if payload['inference_export']['original_checkpoint_sha256']!=record['original_checkpoint_sha256']:raise ValueError('wrong original mapping')
    if payload['contract_sha256']!=record['contract_sha256'] or payload['step']!=record['step']:raise ValueError('wrong export selector')
    if state_sha(payload['model'])!=record['model_tensor_sha256']:raise ValueError('export tensor digest mismatch')
    del payload
    assets=json.loads((ROOT/'results/predictive_transfer/public_base_resolution.json').read_text())
    for item in assets['files']:
        if sha(args.base/item['file'])!=item['local_sha256']:raise ValueError('base tokenizer/config mismatch')
    if sha(args.base/'model.safetensors')!=assets['weight_sha256']:raise ValueError('base weights mismatch')
    work=args.work.resolve()
    if work.exists():raise ValueError('new work directory required')
    source=ROOT/'reproduction/stages'/args.study;index_name,manifest_name=NAMES[args.study]
    sources=json.loads((source/index_name).read_text())
    for rel,digest in sources.items():
        if sha(source/rel)!=digest:raise ValueError('frozen evaluation source mismatch '+rel)
    stage=work/'stage';shutil.copytree(source,stage,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    manifest=json.loads((stage/manifest_name).read_text())
    # Adaptation contracts retain the ORIGINAL full initial checkpoint hash.
    # Only original-role evaluation compares its own container hash to this field.
    if args.role=='original':manifest['initial_checkpoint_sha256']=record['export_sha256']
    if 'checkpoints' in manifest:
        manifest['checkpoints'][args.role]['sha256']=record['export_sha256']
        manifest['checkpoints'][args.role]['path']=str(args.checkpoint.resolve())
    manifest['inference_export_binding']=dict(role=args.role,**record)
    (stage/manifest_name).write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    sources[manifest_name]=sha(stage/manifest_name);(stage/index_name).write_text(json.dumps(sources,indent=2,sort_keys=True)+'\n')
    env=dict(DISTANCE_ROLE=args.role,DISTANCE_CHECKPOINT_SHA256=record['export_sha256'],
             DISTANCE_EXECUTION=str(ROOT/'results/distance_intervention/FROZEN_EXECUTION.json'),TRITON_F32_DEFAULT='ieee',
             TRITON_CACHE_DIR=str(work/'triton_ieee_cache'),CUBLAS_WORKSPACE_CONFIG=':4096:8',
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    command=['torchrun','--standalone','--nproc_per_node=8','-m',f'lrwkv_evidence.{args.study}.evaluate',
             '--base',str(args.base.resolve()),'--model-root',str(stage/'model_source'),
             '--checkpoint',str(args.checkpoint.resolve()),'--out',str(work/'evaluation')]
    receipt=dict(execute=args.execute,command=command,environment=env,source_manifest_sha256=sha(source/manifest_name),
                 rebound_manifest_sha256=sha(stage/manifest_name),tensor_identity=record['model_tensor_sha256'],
                 interpretation='Same model tensors and frozen numerical evaluation; export container hash rebound in separate manifest.')
    (work/'binding.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)
    if args.execute:subprocess.run(command,cwd=stage,env={**os.environ,**env},check=True)
if __name__=='__main__':main()

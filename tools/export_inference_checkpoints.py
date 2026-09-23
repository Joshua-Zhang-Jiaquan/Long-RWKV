"""Prepare lossless inference exports; never modify original checkpoint files."""
import argparse,hashlib,json,os
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def state_sha(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        value=tensor.detach().cpu().contiguous()
        h.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        data=memoryview(value.reshape(-1).view(torch.uint8).numpy())
        for start in range(0,len(data),8*1024*1024):h.update(data[start:start+8*1024*1024])
    return h.hexdigest()

def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(temp,path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    source_manifest=json.loads((ROOT/'reproduction/stages/history_response/history_response_manifest.json').read_text())
    originals=source_manifest['checkpoints']
    records={}
    for role,record in originals.items():
        source=Path(record['path']);require_digest=record['sha256']
        if sha(source)!=require_digest:raise ValueError('original checkpoint mismatch '+role)
        payload=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
        before=state_sha(payload['model']);target=args.out/f'{role}.pt'
        if target.exists():raise ValueError('refuse existing export '+str(target))
        exported={k:v for k,v in payload.items() if k!='optimizer'}
        exported['inference_export']=dict(original_checkpoint_sha256=require_digest,model_tensor_sha256=before,
                                           removed_keys=['optimizer'],format_version=1)
        temp=target.with_suffix('.pt.tmp');torch.save(exported,temp);os.replace(temp,target)
        check=torch.load(target,map_location='cpu',weights_only=False,mmap=True)
        after=state_sha(check['model'])
        if before!=after or check['contract']!=payload['contract']:raise ValueError('non-lossless export')
        if set(check['model'])!=set(payload['model']):raise ValueError('model keys changed')
        records[role]=dict(original_path=str(source),original_sha256=require_digest,export_path=str(target),export_sha256=sha(target),
                           model_tensor_sha256=before,bytes=target.stat().st_size,model_tensors=len(check['model']),step=check['step'],
                           contract_sha256=check.get('contract_sha256'),tensor_identity_verified=True)
        atomic(args.out/'EXPORTS.json',dict(status='local inference exports; no public download claimed',exports=records,
                tensor_digest='sorted names; JSON(name,dtype,shape) then contiguous raw tensor bytes; excludes optimizer state'))
        print(json.dumps(dict(role=role,bytes=records[role]['bytes'],tensor_identity_verified=True)),flush=True)
        del check,exported,payload
    print('all seven inference exports verified',flush=True)
if __name__=='__main__':main()

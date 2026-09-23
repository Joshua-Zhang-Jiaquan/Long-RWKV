"""Fetch the four exact public assets used by the native reconstruction worker."""
import argparse,hashlib,json,os,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
    manifest=json.loads((ROOT/'results/predictive_transfer/public_base_resolution.json').read_text())
    files={r['file']:r['local_sha256'] for r in manifest['files']}
    files['model.safetensors']=manifest['weight_sha256']
    args.out.mkdir(parents=True,exist_ok=True)
    for name,digest in files.items():
        target=args.out/name
        if target.exists():
            if sha(target)!=digest:raise ValueError('existing asset mismatch: '+str(target))
        elif args.verify_only:raise FileNotFoundError(target)
        else:
            url=f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{name}"
            temporary=target.with_suffix(target.suffix+'.download')
            with urllib.request.urlopen(url,timeout=60) as source,temporary.open('xb') as dest:
                while block:=source.read(8*1024*1024):dest.write(block)
            if sha(temporary)!=digest:raise ValueError('download checksum mismatch: '+name)
            os.replace(temporary,target)
        print(json.dumps(dict(file=str(target),sha256=digest,status='verified')),flush=True)
if __name__=='__main__':main()

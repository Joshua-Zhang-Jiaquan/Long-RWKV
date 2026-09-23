"""Complete checkpoint/log/source validation for bounded context adaptation."""
import json,math
from pathlib import Path
import torch
from lrwkv_evidence.train04.worker import file_sha,canonical_sha

def validate(path, terminal=200):
    plan=json.loads(Path(path).read_text());out=Path(plan['out']);stage=Path(plan['stage'])
    qualification=plan['phase']=='qualification';stop=20 if qualification else terminal
    for rel,digest in plan['sources'].items():
        if file_sha(stage/rel)!=digest:raise ValueError('changed source')
    done=json.loads((out/'completion.json').read_text());provenance=json.loads((out/'provenance.json').read_text());contract=provenance['contract']
    if done['qualification']!=qualification or done['step']!=stop or not done['execution_complete']:raise ValueError('wrong completion')
    if canonical_sha(contract)!=done['contract_sha256'] or provenance['contract_sha256']!=done['contract_sha256']:raise ValueError('bad contract')
    if contract['recipe']['terminal_step']!=terminal or contract['initial_checkpoint_sha256']!=plan['initialization']['checkpoint_sha256']:raise ValueError('wrong recipe or initialization')
    checkpoint=out/'resume.pt'
    if file_sha(checkpoint)!=done['checkpoint_sha256']:raise ValueError('checkpoint checksum')
    payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if payload['step']!=stop or payload['contract']!=contract or payload['contract_sha256']!=done['contract_sha256']:raise ValueError('payload mismatch')
    del payload
    for rank in range(8):
        evidence=json.loads((out/f'qualification_rank{rank}.json').read_text())
        if evidence['contract_sha256']!=done['contract_sha256'] or len(evidence['checks'])!=2:raise ValueError('incomplete rank qualification')
        for r in evidence['checks']:
            values=[r[k] for k in ('serial_difference','other_document_difference','own_document_difference')]
            if not all(math.isfinite(v) for v in values) or max(values[:2])>1e-5 or values[2]<1e-6:raise ValueError('isolation failed')
    rows=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows]!=list(range(1,stop+1)) or [r['data_step'] for r in rows]!=list(range(1,stop+1)):raise ValueError('missing updates')
    for r in rows:
        if not all(math.isfinite(r[k]) and r[k]>=0 for k in ('loss','grad_norm_rank0','lr','elapsed_seconds')) or r['global_input_tokens']<=0:raise ValueError('invalid training row')
    return dict(validated=True,qualification=qualification,step=stop,checkpoint_sha256=done['checkpoint_sha256'],contract_sha256=done['contract_sha256'],total_input_tokens=sum(r['global_input_tokens'] for r in rows),training_gpu_hours=done['elapsed_seconds']*8/3600,scope='Elapsed training time excludes queue/initialization/teardown',plan=str(Path(path).resolve()))

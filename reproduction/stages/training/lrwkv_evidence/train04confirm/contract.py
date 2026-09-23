"""Validate a frozen replication recipe before training, without test access."""
import hashlib
import json
from pathlib import Path

RECIPE='paired_history_full_lineage_v1'
TRAINING=dict(initial_seed=0,data_seeds=[53,71,89],global_batch=32,world_size=8,
              learning_rate=3e-5,warmup_updates=50,weight_decay=.01,betas=[.9,.95],eps=1e-8,clip_norm=1.,
              dtype='float32',triton_precision='default',torch_tf32=False,
              parent_terminal_step=1500,branch_terminal_step=2500,
              transitions={'1-300':'ordinary_n2','301-700':'label_mixture_n4','701-1500':'label_mixture_n8','1501-2500':'paired_n8'},
              objective='rao_blackwell',arms=['independent','complementary'])
REQUIRED=(
    'lrwkv_evidence/train04confirm/contract.py','lrwkv_evidence/train04confirm/recipe.py','lrwkv_evidence/train04confirm/worker.py',
    'lrwkv_evidence/train04/tasks.py','lrwkv_evidence/train04/worker.py','lrwkv_evidence/train04/evaluate.py',
    'lrwkv_evidence/train04dev/core.py','lrwkv_evidence/train04dev/worker.py',
    'lrwkv_evidence/train04mix/core.py','lrwkv_evidence/train04binding/core.py','lrwkv_evidence/train04binding/worker.py',
    'lrwkv_evidence/train04paired/core.py')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def validate(path,root,model_root,qualification=False):
    manifest=json.loads(path.read_text())
    status='qualification_only' if qualification else 'frozen'
    if manifest.get('status')!=status or manifest.get('recipe')!=RECIPE or manifest.get('training')!=TRAINING:
        raise ValueError('recipe is not the required frozen/qualification contract')
    if not qualification and (manifest.get('inference_triton_precision')!='ieee' or not manifest.get('selection_evidence')):
        raise ValueError('missing pre-confirmation selection or inference settings')
    required=set(REQUIRED)|{str(p.relative_to(root)) for p in (model_root/'longrwkv').glob('*.py')}
    sources=manifest.get('sources',{})
    if not required.issubset(sources):raise ValueError('missing required source hashes')
    for relative,digest in sources.items():
        source=(root/relative).resolve()
        if not source.is_relative_to(root.resolve()) or sha(source)!=digest:raise ValueError('source hash mismatch: '+relative)
    return manifest,sha(path)

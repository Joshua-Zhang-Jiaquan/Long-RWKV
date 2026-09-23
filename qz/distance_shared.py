"""Control artifacts must be visible from both authorized GPU projects."""
import hashlib
from pathlib import Path
import shutil


def stage_execution(local, global_root):
    local=Path(local);digest=hashlib.sha256(local.read_bytes()).hexdigest()
    target=Path(global_root)/'long_rwkv_distance_control'/f'execution_{digest}.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
        raise ValueError('shared execution conflict')
    if not target.exists():shutil.copyfile(local,target)
    return target

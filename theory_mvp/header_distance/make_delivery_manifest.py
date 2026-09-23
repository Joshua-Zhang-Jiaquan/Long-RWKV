"""Pin the completed, reviewed delivery; raw external predictions are pinned by collectors."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
audit=json.loads((ROOT/'results/header_distance/delivery_audit.json').read_text())
assert audit['complete']
previous=json.loads((ROOT/'results/paper_audit/history_response_delivery_manifest.json').read_text())
paths={ROOT/p for p in previous['files']}
for folder in ('lrwkv_evidence/header_distance','theory_mvp/header_distance','results/header_distance'):
    paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
paths.update((ROOT/'qz').glob('*header*.py'));paths.add(ROOT/'qz/launch_header_distance.sh')
paths.update(ROOT/p for p in ('paper/header_distance_supplement.tex','results/paper_audit/header_distance_visual_review.json','results/paper_audit/completed_history_response_archive.json'))
paths.update((ROOT/'results/paper_audit/header_distance_pages').glob('page-*.png'))
files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
result=dict(at_utc=datetime.now(timezone.utc).isoformat(),status='complete',scope='Updated paper plus frozen fresh-prompt independent header/task-layout evaluation, all controls and stopped startup-attempt receipts. Earlier completed delivery is preserved in its verified archive.',files=files)
dest=ROOT/'results/paper_audit/header_distance_delivery_manifest.json';dest.write_text(json.dumps(result,indent=2)+'\n')
for rel,digest in files.items():assert sha(ROOT/rel)==digest,rel
print(f'Pinned and verified {len(files)} delivery files.')

"""Verify the standalone release seal and preserved experiment source closures."""
import gzip
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def local(rel):
    p=ROOT/rel;require(p.resolve().is_relative_to(ROOT),'path escapes repository');return p
def main():
    seal=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    for rel,digest in seal['files'].items():require(sha(local(rel))==digest,'release file missing or changed: '+rel)
    historical=json.loads((ROOT/'results/paper_audit/header_distance_delivery_manifest.json').read_text())
    for rel,digest in historical['files'].items():require(sha(local(rel))==digest,'original delivery changed: '+rel)
    stages={
      'training':'distance_sources.json','distance_intervention':'distance_eval_sources.json',
      'history_response':'history_response_sources.json','header_distance':'header_distance_sources.json'}
    checked=0
    for stage,index in stages.items():
        root=ROOT/'reproduction/stages'/stage
        for rel,digest in json.loads((root/index).read_text()).items():
            require(sha(root/rel)==digest,'frozen stage changed: '+stage+'/'+rel);checked+=1
    for name in ('raw','run_records'):
        index=json.loads((ROOT/'evidence'/name/'INDEX.json').read_text())
        for item in index['files']:
            p=local(item['path']);require(sha(p)==item['gzip_sha256'],'compressed artifact changed')
            require(hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==item['sha256'],'decompressed artifact changed')
    # LaTeX resolves these files from the repository root through TEXINPUTS.
    visited=set()
    def tex(path):
        if path in visited:return
        visited.add(path);body=path.read_text()
        for kind,value in re.findall(r'\\(input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}',body):
            target=ROOT/value
            if not target.suffix:target=target.with_suffix('.tex' if kind=='input' else '.pdf')
            require(target.is_file(),'paper dependency missing: '+value)
            if kind=='input':tex(target)
    tex(ROOT/'Long_RWKV_ICLR2027.tex')
    audit=json.loads((ROOT/'release_checks/offline_evidence.json').read_text());require(audit['complete'],'offline evidence audit missing')
    rebuild=json.loads((ROOT/'release_checks/paper_rebuild.json').read_text());require(rebuild['text_identical_to_reviewed_pdf'],'paper rebuild mismatch')
    require(sha(ROOT/'build/Long_RWKV_ICLR2027.pdf')==rebuild['reviewed_pdf_sha256'],'reviewed PDF changed')
    print(json.dumps(dict(complete=True,sealed_files=len(seal['files']),historical_delivery_files=len(historical['files']),frozen_stage_source_checks=checked,latex_sources=len(visited),scope='Portable artifact integrity, complete frozen local source stages, raw relocation hashes, paper dependencies and recorded CPU/build checks. External model weights and GPU execution are outside this seal.'),indent=2))
if __name__=='__main__':main()

"""Full synthetic software check. Never generates trained-model evidence."""
import contextlib,hashlib,io,json,math,shutil,sys,tempfile,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from revision.transfer import analyze as A
from lrwkv_evidence.predictive_transfer import tasks as T,prediction as P,training as R
from tools import verify_transfer_evidence as V,package_transfer_evidence as Z
began=time.monotonic();work=Path(tempfile.mkdtemp(prefix='lrwkv_synthetic_pipeline_'));folder=work/'results/predictive_transfer';folder.mkdir(parents=True)
(work/'revision/transfer').mkdir(parents=True)
shutil.copyfile(root/'revision/transfer/FROZEN.json',work/'revision/transfer/FROZEN.json')
protocol_sha=hashlib.sha256((work/'revision/transfer/FROZEN.json').read_bytes()).hexdigest()
A.ROOT=V.ROOT=Z.ROOT=work;A.FOLDER=folder

def write_panel(seed,split):
 out=work/'raw'/f'{split}_{seed}';out.mkdir(parents=True)
 identity=hashlib.sha256(f'SYNTHETIC CHECKPOINT {seed}'.encode()).hexdigest()
 (folder/f'plan_{split}_seed{seed}.json').write_text(json.dumps(dict(out=str(out),checkpoint_sha256=identity)))
 for rank in range(8):
  with (out/f'rank{rank}.jsonl').open('w') as f:
   def emit(x):f.write(json.dumps(x)+'\n')
   emit(dict(kind='provenance',rank=rank,seed=seed,split=split,checkpoint_sha256=identity,protocol_sha256=protocol_sha,scope='SYNTHETIC SOFTWARE FIXTURE'))
   for i in range(6):emit(dict(kind='numerical_screen',within=0.,cross=0.,scope='FIXTURE ONLY'))
   cells=P.panel(split)[rank::8]
   for cell in cells:
    ex=P.example(cell);ys=T.support(ex);initial=[[-math.log(2)]*2 for _ in range(8)];conditional=[]
    for p in (0,1):
     features=dict(T.features(ex,p,cell['position']));hist=[]
     for y in ys:
      values=[]
      for i in range(8):
       z=features[i];error=.025*z[1]**2+.007*z[2]+.013*P.POSITIONS.index(cell['position'])+.002*(seed-R.SEEDS[0])+.001*i+.02
       correct=.5 if seed in R.SEEDS[-2:] else math.exp(-error)
       values.append([math.log(correct if b==y[i] else 1-correct) for b in (0,1)])
      hist.append(values)
     conditional.append(hist)
    emit(dict(kind='condition',cell=cell,instance_id=ex['instance_id'],initial=initial,conditional=conditional,metrics=P.metrics(ex,initial,conditional)))
   emit(dict(kind='complete',conditions=len(cells)))

for seed in R.SEEDS:write_panel(seed,'calibration')
with contextlib.redirect_stdout(io.StringIO()):A.seal()
print('sealed all source-only synthetic predictions',flush=True)
for seed in R.SEEDS:write_panel(seed,'heldout')
with contextlib.redirect_stdout(io.StringIO()):A.report()
report=json.loads((folder/'report.json').read_text())
assert len(report['cells'])==1440 and not report['gates']['competent_lineages'] and not report['all_primary_gates_pass']
print('full synthetic report retains two chance-level lineages and fails competence gate',flush=True)
sys.argv=['audit','--folder',str(folder),'--out',str(work/'original_audit.json')]
with contextlib.redirect_stdout(io.StringIO()):V.main()
print('independent audit passed on all uncompressed fixture ranks',flush=True)
sys.argv=['package','--folder',str(folder),'--out',str(folder/'raw')]
with contextlib.redirect_stdout(io.StringIO()):Z.main()
shutil.rmtree(work/'raw')
sys.argv=['audit','--folder',str(folder),'--raw-index',str(folder/'raw/INDEX.json'),'--out',str(work/'portable_audit.json')]
with contextlib.redirect_stdout(io.StringIO()):V.main()
a=json.loads((work/'original_audit.json').read_text());b=json.loads((work/'portable_audit.json').read_text());assert a==b
receipt=dict(status='passed',checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Synthetic software integration check only; no trained-model results or numerical GPU qualification.',temporary_workspace=str(work),elapsed_seconds=time.monotonic()-began,raw_rank_files=a['raw_rank_files'],full_endpoint_laws_per_audit=a['full256endpoint_laws'],heldout_cells=len(report['cells']),sealed_before_target_creation=True,two_chance_level_lineages_retained=True,competence_failure_propagated=True,independent_uncompressed_audit=True,independent_portable_audit_after_deleting_originals=True,checksums_and_audits_identical=True)
(root/'results/predictive_transfer/synthetic_pipeline_check.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)

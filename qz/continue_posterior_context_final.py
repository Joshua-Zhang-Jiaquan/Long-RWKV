"""Admit the eight fixed final shards, collect complete evidence, never retry silently."""
import json,subprocess,sys,time
from pathlib import Path
import campaign as C
from confirm_capacity import live_usage,CAP
ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'results/posterior_context_final'

def run(args):
 p=subprocess.run([sys.executable,*args],cwd=ROOT,text=True,capture_output=True);print(p.stdout,flush=True)
 if p.returncode:
  if f'{CAP} GPU cap' in p.stdout+p.stderr:return False
  raise RuntimeError((p.stdout+p.stderr)[-2000:])
 return True

def main():
 state=dict(status='running',started_unix=time.time());deadline=time.monotonic()+6*3600
 def save():
  state['updated_unix']=time.time();p=FOLDER/'watcher.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p)
 try:
  while time.monotonic()<deadline:
   count=0
   for role in ('unadapted','adapted200'):
    trainpath=ROOT/'results/long_context_mvp/plan_dacbbe1355018e7f_development200.json'
    if role=='adapted200' and (not trainpath.exists() or not (Path(json.loads(trainpath.read_text())['out'])/'completion.json').exists()):continue
    for shard in range(4):
     key=f'{role}_shard{shard}';planpath=FOLDER/f'plan_{key}.json';dest=FOLDER/f'exact_{key}.json'
     if dest.exists():count+=1;continue
     if not planpath.exists() or not json.loads(planpath.read_text()).get('submission'):
      capacity=live_usage(C.list_all_jobs());state['capacity']=capacity
      if capacity['live_reserved_gpus']+8>CAP:continue
      if not run(['qz/submit_posterior_context_final.py','--role',role,'--shard',str(shard),'--submit']):continue
     plan=json.loads(planpath.read_text());out=Path(plan['out']);complete=True
     for rank in range(8):
      f=out/f'rank{rank}.jsonl'
      try:complete=complete and f.exists() and json.loads(f.read_text().splitlines()[-1])['kind']=='complete'
      except (IndexError,json.JSONDecodeError):complete=False
     if complete:
      run(['-m','theory_mvp.posterior_context_final.collect','--plan',str(planpath),'--out',str(dest)]);count+=1;state[key]=dict(status='validated',report=str(dest));continue
     status,_=C.job_status(plan['submission']['job_id']);state[key]=dict(status=status,job_id=plan['submission']['job_id'])
     if status in ('job_failed','job_stopped','job_succeeded','job_canceled','job_cancelled'):raise RuntimeError(f'{key}: terminal scheduler status without complete raw evidence')
   state['completed_shards']=count
   if count==8:
    run(['-m','theory_mvp.posterior_context_final.report']);state['status']='all_eight_shards_validated';save();return
   save();time.sleep(30)
  state['status']='observation_timeout_no_resubmission';save()
 except Exception as e:state.update(status='needs_attention',error=str(e));save();raise
if __name__=='__main__':main()

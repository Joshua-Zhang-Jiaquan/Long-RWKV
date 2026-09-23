import argparse,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from lrwkv_evidence.train04 import worker as W
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04binding import core as B
from lrwkv_evidence.posterior_context_final import tasks as T
from lrwkv_evidence.long_context_baselines import models as M
from lrwkv_evidence.long_context_eval.numerical import run as numerical_screen
from lrwkv_evidence.long_context_eval.runtime import metadata

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser()
 for name in ('checkpoint','base','model-root','out'):ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--kind',choices=('rwkv','attention'),required=True);args=ap.parse_args()
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);root=Path(__file__).resolve().parents[2]
 if int(os.environ['WORLD_SIZE'])!=8 or os.environ.get('TRITON_F32_DEFAULT')!='ieee':raise ValueError('eight IEEE ranks required')
 for rel,digest in json.loads((root/'cost_sources.json').read_text()).items():
  if W.file_sha(root/rel)!=digest:raise ValueError('source mismatch')
 torch.set_num_threads(4);torch.cuda.set_device(local);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);dist.init_process_group('nccl')
 sha=W.file_sha(args.checkpoint)
 if sha!=os.environ['COST_SHA256']:raise ValueError('wrong checkpoint')
 payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False);contract=payload['contract']
 if W.canonical_sha(contract)!=payload['contract_sha256'] or W.file_sha(args.base/'model.safetensors')!=contract['base_sha256']:raise ValueError('contract/base mismatch')
 for rel,digest in contract['source_hashes'].items():
  if W.file_sha(root/rel)!=digest:raise ValueError('changed checkpoint source')
 if args.kind=='rwkv':
  if payload['step']!=200 or contract['recipe']['terminal_step']!=200:raise ValueError('wrong recurrent checkpoint')
  tok,bits,mask=W.load_tokenizer(args.base);model,_,_=W.build_model(args.model_root,args.base,bits,mask,0);model.backbone.lm_head=C.StableBinaryHead(model.backbone.lm_head)
 else:
  if payload['step']!=20 or contract['recipe']['comparator']!='attention':raise ValueError('wrong attention qualification checkpoint')
  tok,model=M.build('attention',args.base);bits=tok.binary_ids;mask=tok.mask_id
 model.load_state_dict(payload['model'],strict=True);del payload;model=model.to('cuda').eval();packed=C.SerialDenoiser(model) if args.kind=='rwkv' else model
 def predict(ex,encoded,visible):
  prefix=encoded['ids'];canvas=dict(ids=prefix+[bits[visible[i]] if i in visible else mask for i in range(8)],prefix=len(prefix),stage=8-len(visible),masked=[i not in visible for i in range(8)],gold=[0]*8,oracle=[.5]*8)
  return packed(B.pack([B.decorate(ex,canvas,tok)],'cuda'))[0].double().log_softmax(-1)
 args.out.mkdir(parents=True,exist_ok=True)
 with (args.out/f'rank{rank}.jsonl').open('x') as stream:
  def emit(row):stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
  emit(dict(kind='provenance',comparator=args.kind,rank=rank,world=8,checkpoint_sha256=sha,checkpoint_step=200 if args.kind=='rwkv' else 20,design_sha256=W.file_sha(root/'cost_design.json'),final_panel_sha256=W.file_sha(root/'final_design.json'),runtime=metadata(torch,torch.device('cuda',local)),precision='FP32/IEEE;TF32 disabled',timing_repeats=10,budget_seconds=1.5,scope='Mean complete-request latency at batch1; no per-request deadline guarantee; attention quality is not measured'))
  check=T.example(1)
  def screen(spec):
   visible={} if spec['history']=='all_masked' else {i:0 for i in check['information_set']}
   return predict(check,T.serialize(check,tok,spec['context_tokens'],spec['position']),visible)
  emit(dict(kind='numerical_screen',rows=numerical_screen(screen)))
  for index in (2*rank+1,2*rank+17):
   ex=T.example(index)
   for position in ('far','middle','near'):
    encoded=T.serialize(ex,tok,16384,position);costs=[]
    def request(groups,seed):
     g=torch.Generator(device='cuda').manual_seed(seed);visible={}
     for group in groups:
      lp=predict(ex,encoded,visible);draws=torch.multinomial(lp.exp(),1,generator=g).squeeze(-1).tolist();visible.update({i:draws[i] for i in group})
     return sum(visible[i]<<i for i in range(8))
    for method,groups in T.partitions(ex).items():
     request(groups,7100);torch.cuda.synchronize();times=[];peak=0;outputs=[]
     for rep in range(10):
      torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();y=request(groups,7101+rep);torch.cuda.synchronize();times.append(time.perf_counter()-start);outputs.append(y);peak=max(peak,torch.cuda.max_memory_allocated())
     costs.append(dict(method=method,calls=len(groups),seconds=times,peak_allocated_bytes=peak,sample_outputs=outputs))
    emit(dict(kind='condition',index=index,family=ex['family'],instance_id=ex['instance_id'],serial={k:v for k,v in encoded.items() if k!='ids'},costs=costs,scope='Timing outputs are not accuracy estimates'))
  emit(dict(kind='complete',cost_only=True))
 dist.destroy_process_group()
if __name__=='__main__':main()

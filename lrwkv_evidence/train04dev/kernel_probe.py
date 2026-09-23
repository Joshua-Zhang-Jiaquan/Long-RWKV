"""Compare FLA FP32 RWKV7 recurrence and gradients to a double-precision scan.

This probes the recurrence kernel, not parity of the full official RWKV model.
"""
import argparse
import json
from pathlib import Path
import torch


def reference(r,w,k,v,a,b):
    state=torch.zeros((*r.shape[:1],r.shape[2],r.shape[3],v.shape[3]),device=r.device,dtype=r.dtype)
    outputs=[]
    for t in range(r.shape[1]):
        read=(state*a[:,t].unsqueeze(-1)).sum(-2)
        state=state*w[:,t].exp().unsqueeze(-1)+b[:,t].unsqueeze(-1)*read.unsqueeze(-2)+k[:,t].unsqueeze(-1)*v[:,t].unsqueeze(-2)
        outputs.append((state*r[:,t].unsqueeze(-1)).sum(-2))
    return torch.stack(outputs,dim=1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    from fla.ops.rwkv7 import chunk_rwkv7
    import fla
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    rows=[]
    for seed,length in ((17,65),(29,117),(43,128)):
        torch.manual_seed(seed);shape=(1,length,2,64)
        r,k,v=[torch.randn(shape,device='cuda')*.2 for _ in range(3)]
        kk=torch.nn.functional.normalize(torch.randn(shape,device='cuda'),dim=-1)
        w=-.6065306597126334*torch.randn(shape,device='cuda').sigmoid();a=-kk;b=kk*torch.randn(shape,device='cuda').sigmoid()
        values=[x.detach().requires_grad_() for x in (r,w,k,v,a,b)]
        refs=[x.detach().double().requires_grad_() for x in values]
        cu=torch.tensor([0,length],device='cuda',dtype=torch.int32)
        out,_=chunk_rwkv7(*values,scale=1.,cu_seqlens=cu,safe_gate=True,chunk_size=64)
        target=reference(*refs);probe=torch.randn_like(out)
        grads=torch.autograd.grad((out*probe).sum(),values)
        expected=torch.autograd.grad((target*probe.double()).sum(),refs)
        def error(x,y):
            delta=x.double()-y
            return dict(max_absolute=float(delta.abs().max()),relative_l2=float(delta.norm()/y.norm().clamp_min(1e-30)),finite=bool(torch.isfinite(x).all()))
        rows.append(dict(seed=seed,length=length,output=error(out,target),gradients={name:error(x,y) for name,x,y in zip(('r','w','k','v','a','b'),grads,expected)}))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(torch=torch.__version__,fla=fla.__version__,device=torch.cuda.get_device_name(),rows=rows,
        scope='Synthetic bounded kernel inputs; full model and official kernel equivalence not established'),indent=2)+'\n')
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()

"""Summarize replicated precision diagnostics without treating them as a proof."""
import argparse
import hashlib
import json
from pathlib import Path


def collect(root):
    rows=[json.loads((root/f'rank{i}.json').read_text()) for i in range(8)]
    assert all(r['supported'] and r['parameters_unchanged'] for r in rows)
    assert len({r['checkpoint_sha256'] for r in rows})==1
    pairs=[]
    for i in range(4):
        a,b=rows[i],rows[i+4]
        assert a['variant']==b['variant'] and a['state_before']==b['state_before']
        aa={v['context']:v for v in a['outputs'] if v['repeat']==0}
        bb={v['context']:v for v in b['outputs'] if v['repeat']==0}
        assert aa.keys()==bb.keys()
        differences=[]
        for key in aa:
            assert aa[key]['input_sha256']==bb[key]['input_sha256']
            differences.extend(abs(x-y) for u,v in zip(aa[key]['logits'],bb[key]['logits']) for x,y in zip(u,v))
        pairs.append(dict(variant=a['variant'],ranks=[i,i+4],maximum_cross_process_logit_difference=max(differences),
                          maximum_within_process_logit_difference=max(a['maximum_repeated_logit_difference'],b['maximum_repeated_logit_difference'])))
    return dict(checkpoint_sha256=rows[0]['checkpoint_sha256'],contexts_per_rank=8,repetitions=6,pairs=pairs,
                parameters_unchanged=True,scope='Four test conditions; diagnostic only, no checkpoint selection',
                qualification='Repeated canvases stable within these processes. Cross-process BF16 differences observed. FP32 reduces observed differences; neither bitwise reproducibility in general nor a kernel root cause is established.',
                sources={str(root/f'rank{i}.json'):hashlib.sha256((root/f'rank{i}.json').read_bytes()).hexdigest() for i in range(8)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=collect(a.root);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['pairs'],indent=2))

"""Cross-check training targets against brute-force solutions of public text.

This does not use the training GF(2) solver to establish expected marginals.
It audits deterministic samples from actual curriculum index ranges, not model
competence or tokenizer/kernel correctness.
"""
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04mix import core as M
from lrwkv_evidence.train04binding import core as B


def public_solutions(prompt):
    names=prompt.split('\nOutput order: ')[1].split('\n')[0].split(', ')
    equations=[]
    for line in prompt.split('Constraints:\n')[1].split('\nOutput order:')[0].splitlines():
        lhs,rhs=line.split(' = ')
        equations.append(([names.index(name) for name in lhs.split(' XOR ')],int(rhs)))
    return [[(v>>i)&1 for i in range(len(names))] for v in range(1<<len(names))
            if all(sum((v>>i)&1 for i in indices)%2==rhs for indices,rhs in equations)]


def main():
    tok=SimpleNamespace(encode=lambda text:list(text.encode()),binary_ids=(256,257),mask_id=258)
    counts=Counter();cases=0;minimum_support=256
    for n,start in ((2,0),(4,300*32),(8,700*32)):
        for offset in range(256):
            ex=C.example('train',17,start+offset,n)
            support=public_solutions(ex['prompt'])
            assert len(support)==2**(n//2) and ex['bits'] in support
            for law in ('ordinary','mixture'):
                canvas=C.make_canvas(ex,tok,17) if law=='ordinary' else M.canvas(ex,tok,17)
                decorated=B.decorate(ex,canvas,tok)
                visible={i:decorated['ids'][pos]-256 for i,pos in enumerate(decorated['target_positions']) if not canvas['masked'][i]}
                conditioned=[bits for bits in support if all(bits[i]==v for i,v in visible.items())]
                assert conditioned
                minimum_support=min(minimum_support,len(conditioned))
                expected=[sum(bits[i] for bits in conditioned)/len(conditioned) for i in range(n)]
                assert expected==canvas['oracle']
                mask=canvas.get('loss_mask',canvas['masked'])
                assert all(not selected or canvas['masked'][i] for i,selected in enumerate(mask))
                for i,selected in enumerate(mask):
                    if not selected:continue
                    category='fair' if expected[i]==.5 else ('deterministic_one' if expected[i]==1 else 'deterministic_zero')
                    key=f'N{n}/{law}/{ex["family"]}/{canvas.get("branch","corruption")}/{category}'
                    counts[key]+=1
                cases+=1
    out=Path(__file__).resolve().parents[2]/'results/train04binding/public_supervision_audit.json'
    result=dict(scope='Deterministic sample of original training streams; independent public-text brute-force posterior check; no model claim',examples=768,canvases=cases,minimum_conditioned_support=minimum_support,all_public_posteriors_match=True,scored_target_counts=dict(sorted(counts.items())))
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='scored_target_counts'},indent=2))


if __name__=='__main__':main()

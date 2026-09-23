"""Matched soft-target development comparison; no confirmatory inference."""
import json
from pathlib import Path
from lrwkv_evidence.train04.worker import load_tokenizer, file_sha

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B')

def read(path):return json.loads(path.read_text())

def main():
    tok,_,_=load_tokenizer(BASE)
    overhead={i:len(tok.encode(f' bit{i}='))+len(tok.encode(';')) for i in range(1,9)}
    assert set(overhead.values())=={4}
    rows=[];cost=[]
    for law,folder in (('ordinary','train04dev'),('mixture','train04mix')):
        original=read(ROOT/f'results/{folder}/phase700/exact_rao_blackwell.json')
        revised=read(ROOT/f'results/train04binding/phase700/exact_{law}.json')
        for key in ('step','objective','split','seed','conditions_per_dimension','dimensions'):
            assert original['provenance'][key]==revised['provenance'][key]
        old={(r['n'],r['family'],r['method']):r for r in original['summary']}
        for r in revised['summary']:
            a=old[r['n'],r['family'],r['method']]
            rows.append(dict(mask_law=law,n=r['n'],family=r['family'],method=r['method'],original_kl=a['joint_kl_nats'],label_kl=r['joint_kl_nats'],label_minus_original_kl=r['joint_kl_nats']-a['joint_kl_nats'],original_valid_mass=a['exact_valid_mass'],label_valid_mass=r['exact_valid_mass']))
        plan=read(ROOT/f'results/train04binding/plan_86b2795e56d63506_{law}_rao_blackwell_700.json')
        out=Path(plan['out']);logs=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()]
        assert [r['step'] for r in logs]==list(range(301,701))
        actual=sum(r['global_input_tokens'] for r in logs)
        extra=sum(32*r['n']*4 for r in logs)
        cost.append(dict(mask_law=law,steps=[301,700],actual_label_input_tokens=actual,derived_original_input_tokens=actual-extra,extra_label_tokens=extra,token_ratio=actual/(actual-extra),scope='Training forwards only; excludes development evaluation and common steps1-300. Original token count reconstructed from identical prompts and masks by removing four marker/separator tokens per target.'))
    result=dict(scope='Single-seed matched-parent development comparison; N8 is pre-training transfer at step700',tokenizer_vocab_sha256=file_sha(tok.vocab_path),marker_tokens_per_target=4,cost=cost,comparisons=rows)
    dest=ROOT/'results/train04binding/phase700/format_comparison.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Matched format comparison at step 700','','N4 is trained; N8 is transfer before its training phase. Negative KL differences favor public labels. This is development evidence from one data seed.','','| Mask law | Bits | Family | Original KL | Label KL | Difference |','|---|---:|---|---:|---:|---:|']
    for r in rows:
        if r['n'] in (4,8) and r['method']=='information_set':lines.append(f"| {r['mask_law']} | {r['n']} | {r['family']} | {r['original_kl']:.6f} | {r['label_kl']:.6f} | {r['label_minus_original_kl']:+.6f} |")
    lines.extend(['','## Input cost',''])
    for r in cost:lines.append(f"- {r['mask_law']}: {r['actual_label_input_tokens']:,} label-format training tokens versus {r['derived_original_input_tokens']:,} reconstructed original-format tokens, ratio {r['token_ratio']:.4f} over steps 301–700.")
    lines.extend(['','These runs match optimizer updates and target examples, but not input-token compute. The explicit labels cost four additional tokens per target with the released native tokenizer. Both label-format arms remain near the N4 fair-predictor parity baseline. Improvements in some copying-task rows do not demonstrate useful parity learning.'])
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()

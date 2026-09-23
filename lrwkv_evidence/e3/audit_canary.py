"""Audit sixteen exploratory generation canaries without defining an accuracy score."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from . import positive_canary as P, grid324 as G, gpu_runner as U, causal_runner as A


def audit(plan_path):
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text());out=Path(plan['out']);stage=Path(plan['stage'])
    verified={};runtime=[]
    def check(path,expected,*,runtime_local=False):
        path=Path(path);key=str(path)
        if key in verified:
            if verified[key]['claimed_sha256']!=expected: raise ValueError('inconsistent claimed source hash')
            return
        if not path.is_file():
            if not runtime_local: raise ValueError('missing source: '+key)
            result=dict(path=key,status='runtime_local_unavailable',claimed_sha256=expected)
        else:
            actual=U.file_sha(path)
            if actual!=expected and not runtime_local: raise ValueError('source hash mismatch: '+key)
            result=dict(path=key,status='matched' if actual==expected else 'runtime_local_differs_from_job',
                        claimed_sha256=expected,observed_sha256=actual)
        verified[key]=result
        if runtime_local: runtime.append(result)
    for rel,expected in plan['files'].items():check(stage/rel,expected)
    if U.file_sha(Path(P.__file__))!=plan['files']['lrwkv_evidence/e3/positive_canary.py']:
        raise ValueError('local prompt contract differs from staged canary')
    _,_,registry=G._import_generator();encoder=registry.TrieEncoder.from_vocab(str(G.VOCAB))
    provenance={};receipts=[]
    provenance_paths=sorted(out.glob('provenance_rank*.json'))
    if len(provenance_paths)!=8:raise ValueError('expected eight rank provenance records')
    for path in provenance_paths:
        data=json.loads(path.read_text());rank=data['rank'];model='f2' if rank<4 else 'r0'
        if rank not in range(8) or rank in provenance or data['model']!=model or path.name!=f'provenance_rank{rank}.json':
            raise ValueError('rank provenance identity mismatch')
        if (data['scope']!='exploratory_positive_canary_not_primary' or data['budget_tokens']!=32
            or data['prompt_set_sha256']!=U.digest(P.PROMPTS) or data['seed']!=17
            or data['policy']!=(P.CONFIG if model=='f2' else A.POLICY)):
            raise ValueError('canary policy or prompt contract mismatch')
        for source,claimed in data['source_sha256'].items():check(source,claimed,runtime_local=source.startswith('/usr/'))
        for name,claimed in data['weight_sha256'].items():check(Path(data['checkpoint'])/name,claimed)
        model_dir=Path(data['checkpoint']) if model=='r0' else G.G/'models/RWKV7-Goose-World3-2.9B-HF'
        check(model_dir/'config.json',data['model_config_sha256'])
        check(G.VOCAB,data['tokenizer']['vocab_sha256'])
        provenance[rank]=dict(path=str(path),sha256=U.file_sha(path),record=data)
    for model in ('f2','r0'):
        paths=sorted((out/model).glob('*.json'))
        if {p.stem for p in paths}!={p[0] for p in P.PROMPTS}:raise ValueError('missing or unexpected canary prompts')
        for index,(name,prompt,expected) in enumerate(P.PROMPTS):
            path=out/model/f'{name}.json';payload=path.read_bytes();row=json.loads(payload)
            rank=index%4+(0 if model=='f2' else 4);prov=provenance[rank]['record']
            canvas,span=P.make_canvas(encoder,prompt,65535)
            if (row['status']!='ok' or row['scope']!='exploratory_positive_canary_not_primary'
                or row['model']!=model or row['rank']!=rank or row['prompt_name']!=name or row['prompt']!=prompt
                or row['expected_text_reference_only']!=expected or row['expected_ids']!=list(encoder.encode_text(expected))
                or row['input_ids_sha256']!=G.ids_sha256(canvas) or row['target_span']!=list(span)
                or row['provenance_sha256']!=U.digest(prov) or len(row['output_ids'])!=32
                or any(type(t) is not int or not 0<=t<65536 for t in row['output_ids'])
                or row['text']!=encoder.decode_sample_ids(row['output_ids']) or row['diagnostic_nfe']!=1
                or row['tokenizer_check']['native_decode_equal'] is not True
                or row['tokenizer_check']['prefix_ids_sha256']!=G.ids_sha256(canvas[:span[0]])):
                raise ValueError('canary prompt, budget, tokenizer, output or provenance mismatch: '+str(path))
            if model=='f2':
                trace=row['commits_per_step']
                if len(trace)!=row['actual_nfe_generation'] or not 1<=len(trace)<=8 or sum(trace)!=32 or any(type(v)is not int or v<0 for v in trace):
                    raise ValueError('invalid F2 reveal trace')
            elif row['actual_nfe_generation']!=32 or row['commits_per_step'] is not None:
                raise ValueError('invalid R0 fixed-budget generation')
            for metric in ('wall_seconds_generation','peak_allocated_bytes','peak_reserved_bytes'):
                if not math.isfinite(row[metric]) or row[metric]<0:raise ValueError('invalid resource metric')
            first=row['first_step']
            if first['position']!=('first_mask' if model=='f2' else 'last_visible_prefix') or len(first['top10'])!=10:
                raise ValueError('unexpected first-position diagnostic')
            if first['expected_first_token']['id']!=row['expected_ids'][0]:raise ValueError('expected token diagnostic mismatch')
            receipts.append(dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest(),record=row))
    return dict(schema='lrwkv_exploratory_canary_audit_v1',complete=True,n_records=16,n_prompts_per_model=8,
        plan_path=str(plan_path),plan_sha256=U.file_sha(plan_path),prompt_set_sha256=U.digest(P.PROMPTS),
        budget_tokens=32,accuracy_computed=False,checks=dict(exact_prompt_coverage=True,paired_identical_prompts=True,
            source_canvas_hashes=True,masked_suffix_budget=True,output_decode_equality=True,rank_provenance_binding=True),
        source_checks=list(verified.values()),runtime_local_source_checks=runtime,
        provenance=list(provenance.values()),outputs=receipts,
        limitations=['Exploratory prompts chosen after the development competence floor; no post-hoc accuracy threshold or aggregate success score.',
            'Compared with symbolic confirmation, both prompt serialization and output budget change; this does not identify a causal prompt-only or budget-only effect.',
            'Expected-text references and first-token ranks are diagnostics, not a scoring rule; whitespace/tokenization changes expected first-token ranks.',
            'Runtime /usr source hashes are job-reported. Matching current-host bytes corroborates file content only, not the full historical runtime environment; inaccessible or differing host files do not invalidate shared-source checks.',
            'Generation includes 32 output IDs even where special IDs decode to empty text. One additional diagnostic forward is separate from generation NFE.'])


def _escape(text):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','\n':r'\textbackslash{}n'}
    return ''.join(replacements.get(c,c) for c in text)


def latex(report):
    outputs={(v['record']['model'],v['record']['prompt_name']):v['record']['text'] for v in report['outputs']}
    selected=[('chat_capital','Capital, chat', ' It is Paris. Paris city name',' Paris.'),
              ('chat_lookup','Lookup, chat',' the, the.  ----> select the3 to the3.',' The value stored for key093 is v1911.'),
              ('symbolic_raw','Symbolic raw','a:\nc:\n2\nd:\n1','dvk093\ndvk093\ndvk093')]
    lines=[r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{@{}p{0.18\linewidth}p{0.38\linewidth}p{0.38\linewidth}@{}}',r'\toprule',r'Prompt & F2 leading excerpt & R0 leading excerpt \\',r'\midrule']
    for key,label,f2,r0 in selected:
        if not outputs['f2',key].startswith(f2) or not outputs['r0',key].startswith(r0):raise ValueError('quoted excerpt differs from source')
        lines.append(f'{label} & \\texttt{{{_escape(f2.strip())}}} & \\texttt{{{_escape(r0.strip())}}}'+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',
        r'\caption{Illustrative leading excerpts from exploratory 32-token canaries; leading whitespace is omitted and literal \texttt{\textbackslash{}n} denotes a newline. Excerpts are not a scoring rule. The paired models receive identical prompts and budgets; all eight prompt pairs, full output IDs/text and source hashes are retained in \texttt{results/e3\_canary\_audit.json}.}',
        r'\label{tab:e3-canary}',r'\end{table}','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=Path('results/e3_canary_plan.json'))
    parser.add_argument('--out',type=Path,default=Path('results/e3_canary_audit.json'))
    args=parser.parse_args();report=audit(args.plan)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    args.out.with_suffix('.tex').write_text(latex(report))
    print(json.dumps(dict(n_records=report['n_records'],source_files=len(report['source_checks']),
                         runtime_local=report['runtime_local_source_checks']),indent=2))


if __name__=='__main__':main()

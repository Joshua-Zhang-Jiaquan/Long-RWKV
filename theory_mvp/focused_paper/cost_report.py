"""Plot measured costs without attributing answer quality to cost-only runs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from lrwkv_evidence.long_context_eval.cost_collect import collect


def summarize(reports):
    by={r['comparator']:r for r in reports}
    if len(by)!=len(reports) or 'rwkv' not in by:raise ValueError('requires unique models and main RWKV')
    if any(not r['cost_only'] or not r['execution_complete'] for r in reports):raise ValueError('requires complete cost-only reports')
    summaries=[]
    for kind,report in by.items():
        for length in (1024,4096,16384):
            rows=[r for r in report['conditions'] if r['requested_context_tokens']==length]
            if len(rows)!=24:raise ValueError('missing cost-panel cell')
            for method in [c['method'] for c in rows[0]['costs']]:
                costs=[next(c for c in r['costs'] if c['method']==method) for r in rows]
                means=[statistics.mean(c['seconds']) for c in costs]
                summaries.append(dict(model=kind,context_tokens=length,method=method,
                                      mean_seconds=statistics.mean(means),condition_mean_range=[min(means),max(means)],
                                      peak_allocated_gib=max(c['peak_allocated_bytes'] for c in costs)/2**30,
                                      records_range=[min(r['records'] for r in rows),max(r['records'] for r in rows)]))
    lookup={(r['model'],r['context_tokens'],r['method']):r for r in summaries}
    bands=[]
    if 'attention' in by:
        for length in (1024,4096,16384):
            rwkv=lookup['rwkv',length,'information_set']['mean_seconds']
            one=lookup['attention',length,'one']['mean_seconds']
            others=[r['mean_seconds'] for r in summaries if r['model']=='attention' and r['context_tokens']==length and r['method']!='one']
            lower=max(rwkv,one);upper=min(others)
            bands.append(dict(context_tokens=length,lower_inclusive_mean_seconds=lower,upper_exclusive_mean_seconds=upper,
                              nonempty_mean_cost_band=lower<upper,quality_certificate='not established by cost-only measurements'))
    return dict(cost_only=True,measurements=summaries,provisional_mean_cost_bands=bands,
                checkpoints={k:dict(sha256=r['checkpoint_sha256'],step=r['checkpoint_step']) for k,r in by.items()},
                limitations=['Batch1 pretokenized full requests, FP32/IEEE on H100; these are implementation-specific measured costs.',
                             'Twenty-four condition means per long length: four paired base problems, two task prompts, three positions. Ten timing draws per condition are not independent quality samples.',
                             'Whiskers show the range of condition means, not confidence intervals or per-request deadline guarantees.',
                             'Native tokenization/distractor counts differ. Baseline qualification weights and main terminal weights have different training histories.',
                             'A nonempty mean-cost band alone does not establish the joint error/budget certificate. No answer-quality endpoint is measured here.'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,action='append',required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();reports=[]
    for path in args.report:
        old=json.loads(path.read_text());fresh=collect(json.loads(Path(old['evaluation_plan']).read_text()))
        if any(old[k]!=fresh[k] for k in ('checkpoint_sha256','conditions','shard_sha256','comparator','aggregates')):raise ValueError('changed raw cost evidence')
        reports.append(old)
    result=summarize(reports)
    result['input_sha256']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.report}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Development request-cost measurements','',*result['limitations'],'',
           '| Model | Native context | Policy | Mean seconds | Peak allocated GiB |',
           '|---|---:|---|---:|---:|']
    for r in result['measurements']:
        lines.append(f'| {r["model"]} | {r["context_tokens"]} | {r["method"]} | {r["mean_seconds"]:.6f} | {r["peak_allocated_gib"]:.3f} |')
    args.out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(7.0,3.3),constrained_layout=True)
    styles=[('rwkv','one','RWKV, one call','#2962a3','--'),('rwkv','information_set','RWKV, two calls','#2962a3','-'),
            ('attention','one','Attention, one call','#d97b22','--'),('attention','information_set','Attention, two calls','#d97b22','-'),
            ('causal_rwkv','cached_causal','Causal RWKV, cached','#4e8b52','-')]
    for model,method,label,color,style in styles:
        rows=[r for r in result['measurements'] if r['model']==model and r['method']==method]
        if not rows:continue
        x=[r['context_tokens'] for r in rows];y=[r['mean_seconds'] for r in rows]
        errors=[[r['mean_seconds']-r['condition_mean_range'][0] for r in rows],
                [r['condition_mean_range'][1]-r['mean_seconds'] for r in rows]]
        axes[0].errorbar(x,y,yerr=errors,label=label,color=color,linestyle=style,marker='o',capsize=3)
        if method!='one':axes[1].plot(x,[r['peak_allocated_gib'] for r in rows],label=label,color=color,linestyle=style,marker='o')
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks([1024,4096,16384],['1K','4K','16K']);ax.set_xlabel('Native context tokens');ax.grid(alpha=.2)
    axes[0].set_yscale('log');axes[0].set_ylabel('Full-request latency (seconds)');axes[0].legend(fontsize=7)
    axes[1].set_ylabel('Peak allocated GPU memory (GiB)');axes[1].legend(fontsize=7)
    fig.suptitle('Development cost screen — answer quality not measured',fontsize=9.5)
    fig.savefig(args.out.with_suffix('.pdf'));fig.savefig(args.out.with_suffix('.png'),dpi=180)
    plt.close(fig);print(json.dumps(dict(report=str(args.out),models=list(result['checkpoints']),bands=result['provisional_mean_cost_bands'])))


if __name__=='__main__':main()

import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 data=json.loads(args.report.read_text())
 if not data['execution_complete'] or not data['long_sweep_complete']:raise ValueError('complete probe required')
 rows=[r for r in data['summary'] if r['family']=='paired_parity' and r['method']=='information_set']
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axes=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
 short=next(r for r in rows if r['context_tokens'] is None)
 for position,color in zip(('far','middle','near'),('#c43c39','#bc8600','#2375af')):
  cells=[short]+[next(r for r in rows if r['context_tokens']==n and r['position']==position) for n in (1024,4096,16384)]
  axes[0].plot(range(4),[r['mean_kl'] for r in cells],'o-',color=color,label=position.capitalize(),markersize=4)
  axes[1].plot(range(4),[r['mean_valid'] for r in cells],'o-',color=color,label=position.capitalize(),markersize=4)
 axes[0].axhline(4*math.log(2),color='.35',ls='--',lw=1,label='One-call KL lower bound')
 axes[0].set_yscale('log');axes[0].set_ylabel('Mean two-call KL (nats)');axes[0].legend(fontsize=7,loc='lower right')
 axes[1].axhline(1/16,color='.35',ls='--',lw=1);axes[1].set_ylabel('Mean exact valid mass');axes[1].set_ylim(-.03,1.03)
 for ax in axes:ax.set_xticks(range(4),['Short','1K','4K','16K']);ax.set_xlabel('Requested native context length');ax.grid(axis='y',alpha=.18)
 fig.suptitle('Selected checkpoint: dependent posterior, four development conditions',fontsize=10)
 for suffix in ('.pdf','.png'):fig.savefig(args.out.with_suffix(suffix),dpi=170)
if __name__=='__main__':main()

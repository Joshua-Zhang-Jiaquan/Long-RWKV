"""Standalone publication figures/tables from the complete frozen report."""
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def write(report,folder):
 folder=Path(folder);rows=report['summary'];plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axes=plt.subplots(2,3,figsize=(7,4.5),layout='constrained',sharex=True,sharey='row')
 for column,position in enumerate(('far','middle','near')):
  for role,color,label in (('unadapted','#777777','Before context adaptation'),('adapted200','#1672af','After 200 updates')):
   selected=[]
   for length in (None,1024,4096,16384):
    pos='evidence_only' if length is None else position
    selected.append(next(r for r in rows if r['role']==role and r['family']=='paired_parity' and r['context_tokens']==length and r['position']==pos and r['method']=='information_set'))
   for row,(metric,interval) in enumerate((('mean_kl','kl_interval'),('mean_valid','valid_interval'))):
    ax=axes[row,column];ys=[r[metric] for r in selected];lower=[r[interval][0] for r in selected];upper=[r[interval][1] for r in selected]
    if row==0:ys=[max(1e-8,x) for x in ys];lower=[max(1e-8,x) for x in lower];upper=[max(1e-8,x) for x in upper]
    ax.plot(range(4),ys,'o-',color=color,label=label,markersize=3);ax.fill_between(range(4),lower,upper,color=color,alpha=.15,lw=0)
  axes[0,column].axhline(4*math.log(2),color='.35',ls='--',lw=.8);axes[0,column].set_yscale('log');axes[0,column].set_title(position.capitalize()+' evidence')
  axes[1,column].axhline(1/16,color='.35',ls='--',lw=.8);axes[1,column].set_ylim(-.025,1.025)
  axes[1,column].set_xticks(range(4),['Short','1K','4K','16K'])
  for row in range(2):axes[row,column].grid(axis='y',alpha=.15)
 axes[0,0].set_ylabel('Information-set KL (nats)');axes[1,0].set_ylabel('Exact valid mass')
 axes[0,0].legend(fontsize=6.6,loc='best');fig.supxlabel('Requested native context length',fontsize=9)
 for suffix in ('.pdf','.png'):fig.savefig(folder/('mechanism_final'+suffix),dpi=180)
 plt.close(fig)
 lines=[r'\begin{tabular}{@{}lrrrrl@{}}',r'\toprule',r'Checkpoint & One-call KL & Two-call KL & Two-call validity & KL gain & 95\% interval\\',r'\midrule']
 for role,label in (('unadapted','Before adaptation'),('adapted200','After adaptation')):
  def mean(method,key):return sum(r[key] for r in rows if r['role']==role and r['family']=='paired_parity' and r['context_tokens']==16384 and r['method']==method)/3
  primary=next(r for r in report['checkpoint_analyses'][role]['primary_estimands'] if r['comparator']=='one' and r['metric']=='joint_kl_nats');lo,hi=primary['pointwise_95_percentile_interval']
  lines.append(f"{label} & {mean('one','mean_kl'):.3f} & {mean('information_set','mean_kl'):.3f} & {mean('information_set','mean_valid'):.3f} & {primary['mean']:.3f} & $[{lo:.3f},{hi:.3f}]$"+r'\\')
 lines += [r'\bottomrule',r'\end{tabular}'];(folder/'paper_table.tex').write_text('\n'.join(lines)+'\n')

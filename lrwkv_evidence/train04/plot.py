"""Export scientific figures from qualified training and evaluation receipts."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluate import METHODS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--summary',type=Path);ap.add_argument('--training',type=Path,nargs='+',required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.2,3.3),layout='constrained')
    largest_loss=1.5
    for d in args.training:
        rows=[json.loads(s) for s in (d/'train.jsonl').read_text().splitlines()];prov=json.loads((d/'provenance.json').read_text())
        x=np.array([r['step'] for r in rows]);y=np.array([r['mean_loss'] for r in rows]);largest_loss=max(largest_loss,float(y.max())*1.05);color=f'C{(17,29,43).index(prov["seed"])}'
        ax.plot(x,y,color=color,alpha=.13,lw=.5)
        if len(y)>=20:ax.plot(x[19:],np.convolve(y,np.ones(20)/20,mode='valid'),color=color,label=f'Seed {prov["seed"]}: trailing20 updates')
    ax.axhline(math.log(2),c='.5',ls=':',lw=1,label='Always-fair predictor (expected)')
    ax.axhline(17*math.log(2)/32,c='black',ls='--',lw=1,label='Exact-posterior optimum (expected)')
    ax.set(xlabel='Optimizer update',ylabel='Absorbing loss per target token (nats)',ylim=(0,largest_loss))
    ax.legend(fontsize=7,loc='upper right');ax.spines[['top','right']].set_visible(False)
    for ext in ('pdf','png'):fig.savefig(args.out/f'training_curves.{ext}',dpi=180)
    plt.close(fig)
    if not args.summary:return
    data=json.loads(args.summary.read_text());reference=json.loads((Path(__file__).resolve().parents[2]/'results/train04/oracle_reference.json').read_text())
    labels=['One round','Information set (2)','Random halves (2)','Bernoulli (2)','Bernoulli (4)','Bernoulli (8)','Sequential (8)']
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained',sharey=True)
    for ax,family,title in zip(axes,('systematic','paired_parity'),('Systematic constraints','Pair-parity constraints')):
        values=np.array([[next(c['valid_fraction'] for c in r['cells'] if c['family']==family and c['method']==m) for m in METHODS] for r in data['runs']])
        x=np.arange(len(METHODS));ax.bar(x,100*values.mean(0),color=['.7','#377eb8','#e41a1c','.7','.7','.7','.7'],width=.6,alpha=.65)
        for j,v in enumerate(values):ax.scatter(x+(j-(len(values)-1)/2)*.11,100*v,s=17,color='black',marker=('o','s','^')[j],label=f'Seed{data["runs"][j]["seed"]}')
        oracle=[next(c['oracle_valid_fraction'] for c in reference['cells'] if c['family']==family and c['method']==m) for m in METHODS]
        ax.scatter(x,100*np.array(oracle),marker='_',s=180,c='#4daf4a',linewidths=2,label='Exact-posterior oracle')
        ax.set(title=title,xticks=x,xticklabels=labels,ylim=(0,104));ax.tick_params(axis='x',rotation=50);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Constraint-valid samples (%)');axes[1].legend(fontsize=7,loc='best')
    for ext in ('pdf','png'):fig.savefig(args.out/f'sampling_validity.{ext}',dpi=180)
    plt.close(fig)
    fixed=('one','information_set','random_halves','sequential')
    fig,axes=plt.subplots(1,2,figsize=(7.5,3.3),layout='constrained',sharey=True)
    for ax,family in zip(axes,('systematic','paired_parity')):
        costs=[];errors=[]
        for method in fixed:
            cells=[c for r in data['runs'] for c in r['cells'] if c['family']==family and c['method']==method]
            costs.append(np.mean([c['small_panel_dependence_cost_nats'] for c in cells]));errors.append(np.mean([c['small_panel_estimation_error_nats'] for c in cells]))
        x=np.arange(4);ax.bar(x,costs,label='Discarded dependence',color='#e69f00');ax.bar(x,errors,bottom=costs,label='Denoiser estimation error',color='#56b4e9')
        ax.set(title=family.replace('_',' ').title(),xticks=x,xticklabels=['One','Info set','Random halves','Sequential']);ax.tick_params(axis='x',rotation=25);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Exact endpoint KL (nats)')
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=7)
    fig.suptitle('Small audit: two conditions per family, averaged across three seeds',fontsize=9)
    for ext in ('pdf','png'):fig.savefig(args.out/f'kl_decomposition.{ext}',dpi=180)
    plt.close(fig)
if __name__=='__main__':main()

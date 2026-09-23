"""Mechanism figure with paired-problem uncertainty, conditional on three seeds."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parent))
from report import FOLDER,SEEDS,POSITIONS,primary,metric,estimate


def main():
    roles=['original']+[f'{a}_seed{s}' for s in SEEDS for a in ('near','balanced')]
    data={r:primary(json.loads((FOLDER/f'exact_{r}.json').read_text())) for r in roles}
    panels=[('joint_kl_nats','Joint KL (nats)'),
            ('stages.second_round_estimation_nats','Second-pass conditional error'),
            ('response.mean_correct_conditional_ce','Counterfactual target CE'),
            ('response.signed_logit_contrast','Signed evidence logit contrast'),
            ('response.centering_penalty','Counterfactual centering penalty'),
            ('response.unaffected_mean_abs_probability_change','Unaffected-target probability change')]
    fig,axes=plt.subplots(2,3,figsize=(12,6.4),constrained_layout=True)
    for ax,(key,title) in zip(axes.flat,panels):
        for arm,label,color in [('original','Starting checkpoint','#888888'),('near','Near-only adaptation','#cb6b32'),('balanced','Distance-balanced adaptation','#2477b4')]:
            vals=[];lo=[];hi=[]
            for position in POSITIONS:
                values=metric(data['original'],key,position) if arm=='original' else np.mean([metric(data[f'{arm}_seed{s}'],key,position) for s in SEEDS],axis=0)
                stat=estimate(values);vals.append(stat['mean']);lo.append(stat['ci95'][0]);hi.append(stat['ci95'][1])
            ax.plot(range(3),vals,'o-',label=label,color=color,linewidth=1.7,markersize=4)
            ax.fill_between(range(3),lo,hi,color=color,alpha=.12)
        if key=='joint_kl_nats':ax.axhline(4*np.log(2),color='black',linestyle=':',linewidth=1,label='One-call product lower bound')
        ax.set_xticks(range(3),['Far','Middle','Near']);ax.set_title(title,fontsize=10)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=8)
    fig.suptitle('16K distance intervention: three adaptation seeds, one selected starting lineage',fontsize=12)
    fig.savefig(FOLDER/'mechanism_distance.pdf',bbox_inches='tight')
    fig.savefig(FOLDER/'mechanism_distance.png',dpi=180,bbox_inches='tight')


if __name__=='__main__':main()

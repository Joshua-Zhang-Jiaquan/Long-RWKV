"""Publication diagram of the implemented recurrent sampler and equal-cost test."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(7.2,3.15));ax.set(xlim=(0,10),ylim=(0,4.6));ax.axis('off')
blue='#36688b';orange='#b76935';grey='#e8ecef'
def box(x,y,w,h,label,color=grey):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',facecolor=color,edgecolor='#607080',lw=.8))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center')
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#394956',lw=1.2))
box(.1,3.42,2.65,.87,'Full retained context\n+ labeled answer canvas')
box(3.18,3.42,3.5,.87,'Tied RWKV forward / reverse scans\nlearned directional fusion + FFN')
box(7.13,3.42,2.65,.87,'Binary marginals\nat answer positions')
arrow((2.8,3.85),(3.12,3.85));arrow((6.73,3.85),(7.07,3.85))
ax.text(5,3.03,'Each call rebuilds recurrent states and reads the full context.',ha='center',fontsize=8.5)

def canvas(x,y,filled,label):
    for i in range(8):
        color=(blue if i<4 else orange) if i in filled else 'white'
        ax.add_patch(Rectangle((x+i*.24,y),.205,.34,facecolor=color,edgecolor='#66717a',lw=.7))
        if i not in filled:ax.text(x+i*.24+.103,y+.17,'?',ha='center',va='center',fontsize=8)
    ax.text(x+.95,y-.22,label,ha='center',va='top',fontsize=8)
for y,first,name in ((2.1,list(range(4)),'A first'),(.95,list(range(4,8)),'B first')):
    ax.text(.1,y+.17,name,va='center',weight='bold',color=blue if name=='A first' else orange)
    canvas(1.4,y,[],'all masked');canvas(4.0,y,first,'sample first basis')
    canvas(6.6,y,list(range(8)),'sample remaining basis')
    arrow((3.37,y+.17),(3.92,y+.17));arrow((5.97,y+.17),(6.52,y+.17))
    ax.text(3.64,y+.48,'call 1',ha='center',fontsize=8);ax.text(6.24,y+.48,'call 2',ha='center',fontsize=8)
    ax.text(8.85,y+.16,'$D=0$\n2 calls',ha='center',va='center')
ax.text(5,.15,'Public invertible bases: equal discarded dependence and scan count; learned errors may differ.',ha='center',fontsize=8.5)
fig.tight_layout(pad=.2)
folder=Path(__file__).resolve().parent
for ext in ('pdf','png','svg'):fig.savefig(folder/f'sampler_schematic.{ext}',dpi=180,bbox_inches='tight')

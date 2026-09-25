"""Redraw only existing table values; no new model evaluation or re-ranking."""
from pathlib import Path
import csv,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
def read(name):
    rows={}
    for line in (ROOT/'generated'/name).read_text().splitlines():
        cells=[s.strip().replace('$','') for s in line.split('&')]
        if len(cells)==5 and cells[1] in ('yes','no'):
            rows[cells[0]]=(cells[1],float(cells[2]),int(cells[3]),float(cells[4].replace('\\','').strip()))
    return rows
contact,interface=read('contact_task.tex'),read('masif_task.tex')
names=sorted(contact.keys() & interface.keys(),key=lambda n:contact[n][3])
assert len(names)==12
for name in names: assert contact[name][3]==interface[name][3]
with (ROOT/'generated/residue_rankings.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['encoder','reads_coordinates','pooled_mean_rank','contact_rank','interface_rank','contact_auprc','interface_auprc'])
    for n in names:w.writerow([n,contact[n][0],contact[n][3],contact[n][2],interface[n][2],contact[n][1],interface[n][1]])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.labelcolor':'#333333','text.color':'#333333'})
fig,axs=plt.subplots(1,2,figsize=(7.15,3.2),sharey=True)
for ax,data,title in zip(axs,[contact,interface],['A  Contact prediction','B  Interface prediction']):
    for i,n in enumerate(names):
        coord,_,rank,pooled=data[n];color='#B9786F' if coord=='yes' else '#4C65A4'
        if i%2==0:ax.axhspan(i-.5,i+.5,color='#F6F5F9',zorder=0)
        ax.plot([pooled,rank],[i,i],color=color,lw=1.5,alpha=.85,zorder=2)
        ax.scatter(pooled,i,s=27,marker='o',edgecolor=color,facecolor='white',lw=1.1,zorder=3)
        ax.scatter(rank,i,s=23,marker='D',color=color,zorder=4)
    ax.set(title=title,xlim=(.2,33.8),xlabel='Rank (lower is better)',xticks=[1,5,10,15,20,25,30,33])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.grid(axis='x',color='#e5e5eb',lw=.55,zorder=1);ax.set_axisbelow(True)
    ax.tick_params(length=0,labelsize=9,pad=4)
    for spine in ax.spines.values():spine.set_visible(False)
axs[0].set_ylim(len(names)-.4,-.7)
handles=[Line2D([],[],marker='o',color='#555555',markerfacecolor='white',linestyle='',label='Pooled mean rank'),Line2D([],[],marker='D',color='#555555',linestyle='',label='Residue-level rank'),Line2D([],[],color='#4C65A4',label='Sequence-only'),Line2D([],[],color='#B9786F',label='Coordinate-reading')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.006),ncol=2,frameon=False,fontsize=9)
fig.subplots_adjust(left=.145,right=.985,top=.91,bottom=.29,wspace=.13)
fig.savefig(ROOT/'figures/residue_rankings.pdf')
fig.savefig(ROOT/'figures/residue_rankings.png',dpi=200)

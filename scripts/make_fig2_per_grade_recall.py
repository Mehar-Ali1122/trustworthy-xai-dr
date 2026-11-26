import os; os.makedirs('figures', exist_ok=True)
# Re-renders the per-grade recall figure at 600 dpi from per_class_test.csv (notebook cell 21 outputs).
import csv, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':0.8})
R={}
for r in csv.DictReader(open('results/per_class_test.csv')):
    R.setdefault(r['run'],{})[int(r['grade'])]=float(r['recall'])
runs=[('Swin-T_IDRiD','Swin-T · IDRiD','#7fb3d8'),('ViT-B16_IDRiD','ViT-B/16 · IDRiD','#1f5f8b'),('Swin-T_APTOS','Swin-T · APTOS','#f2b27c'),('ViT-B16_APTOS','ViT-B/16 · APTOS','#b8730f')]
fig,ax=plt.subplots(figsize=(3.5,2.2)); w=0.2; x=np.arange(5)
for k,(run,lab,c) in enumerate(runs):
    ax.bar(x+(k-1.5)*w,[R[run][g] for g in range(5)],w,color=c,edgecolor='black',linewidth=0.5,label=lab)
ax.text(1-1.0*w,0.02,'0',color='#b03a2e',fontweight='bold',ha='center',va='bottom',fontsize=8)
ax.text(4.6,0.97,'IDRiD Grade 1: recall = 0 (n = 5)',color='#b03a2e',ha='right',va='top',fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels([f'G{g}' for g in range(5)]); ax.set_ylim(0,1.02)
ax.set_ylabel('Recall'); ax.set_xlabel('DR severity grade')
ax.yaxis.grid(True,color='#dddddd',linewidth=0.5); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.legend(ncol=2,fontsize=6.5,frameon=False,loc='lower center',bbox_to_anchor=(0.5,1.0))
fig.tight_layout(); fig.savefig('figures/Fig2_PerGrade_Recall.png',dpi=600,facecolor='white')

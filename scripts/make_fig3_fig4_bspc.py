import os; os.makedirs('figures', exist_ok=True)
# Builds the two new BSPC figures from values printed/saved by the executed notebooks.
import csv, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':0.6})
OUT='figures/'
# --- Sanity: per-fraction mean |Spearman| (notebook cells 37/37/38/37 text output) ---
S={
 'Swin-Tiny · IDRiD':{'Grad-CAM':[0.202,0.148,0.234],'Transf. Attr.':[0.183,0.155,0.152],'TokenNorm‡':[0.164,0.169,0.164],'Input-Grad.':[0.436,0.361,0.096],'Integrated Grad.':[0.710,0.691,0.539]},
 'ViT-B/16 · IDRiD':{'Grad-CAM':[0.092,0.145,0.138],'Transf. Attr.':[0.128,0.102,0.146],'Attn. Rollout†':[0.997,0.908,0.390],'Input-Grad.':[0.604,0.483,0.174],'Integrated Grad.':[0.740,0.694,0.585]},
 'Swin-Tiny · APTOS':{'Grad-CAM':[0.179,0.172,0.201],'Transf. Attr.':[0.161,0.157,0.187],'TokenNorm‡':[0.204,0.190,0.228],'Input-Grad.':[0.463,0.375,0.117],'Integrated Grad.':[0.752,0.731,0.575]},
 'ViT-B/16 · APTOS':{'Grad-CAM':[0.102,0.192,0.243],'Transf. Attr.':[0.136,0.096,0.134],'Attn. Rollout†':[0.989,0.933,0.463],'Input-Grad.':[0.592,0.466,0.129],'Integrated Grad.':[0.761,0.722,0.648]},
}
# cross-check means against scorecards.csv
sc={(r['run'],r['explainer']):float(r['mean_abs_rho']) for r in csv.DictReader(open('results/scorecards.csv'))}
runmap={'Swin-Tiny · IDRiD':'Swin-T_IDRiD','ViT-B/16 · IDRiD':'ViT-B16_IDRiD','Swin-Tiny · APTOS':'Swin-T_APTOS','ViT-B/16 · APTOS':'ViT-B16_APTOS'}
exmap={'Grad-CAM':'GradCAM','Transf. Attr.':'TransformerAttribution','TokenNorm‡':'TokenNorm (Swin surrogate)','Attn. Rollout†':'AttentionRollout','Input-Grad.':'InputGrad (baseline)','Integrated Grad.':'IntegratedGradients'}
for cfg,d in S.items():
    for ex,v in d.items():
        m=round(sum(v)/3+1e-9,3); ref=sc[(runmap[cfg],exmap[ex])]
        assert abs(m-ref)<=0.0011,(cfg,ex,m,ref)
fig,axes=plt.subplots(2,2,figsize=(7.0,4.4))
cmap=plt.get_cmap('Blues'); norm=Normalize(0,1)
for ax,(cfg,d) in zip(axes.flat,S.items()):
    rows=list(d); M=np.array([d[r]+[sum(d[r])/3] for r in rows])
    disp=np.full((len(rows),5),np.nan); disp[:,:3]=M[:,:3]; disp[:,4]=M[:,3]
    ax.imshow(np.ma.masked_invalid(disp),cmap=cmap,norm=norm,aspect='auto')
    for i in range(len(rows)):
        for j,val in [(0,M[i,0]),(1,M[i,1]),(2,M[i,2]),(4,M[i,3])]:
            ax.text(j,i,(f'{val:.3f}' if j==4 else f'{val:.2f}'),ha='center',va='center',fontsize=(6.6 if j==4 else 7.5),color='white' if val>0.55 else '#1a1a1a')
        fail=M[i,3]>=0.30
        ax.text(5.0,i,'FAIL' if fail else 'pass',ha='left',va='center',fontsize=7,color='#b2182b' if fail else '#1a1a1a',fontweight='bold' if fail else 'normal')
    ax.set_xticks([0,1,2,4]); ax.set_xticklabels(['0.25','0.50','1.00','mean'])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xlim(-0.5,5.9); ax.set_title(cfg,fontsize=8.5,fontweight='bold')
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
for ax in axes[1]: ax.set_xlabel('Fraction of layers re-initialized (top-down)')
fig.subplots_adjust(left=0.15,right=0.86,top=0.93,bottom=0.11,wspace=0.62,hspace=0.42)
cax=fig.add_axes([0.89,0.25,0.015,0.55])
cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax); cb.set_label('|Spearman ρ|, trained vs. randomized (lower = better)',fontsize=7)
cb.ax.axhline(0.30,color='#b2182b',lw=1.2); cb.ax.text(1.6,0.30,'τ',color='#b2182b',va='center',fontsize=8,transform=cb.ax.get_yaxis_transform())
fig.savefig(OUT+'Fig3_Sanity_Randomization.png',dpi=600,facecolor='white'); plt.close(fig)
# --- Plausibility per lesion type (plausibility_per_lesion.csv, notebook cell 42) ---
P={}
for r in csv.DictReader(open('results/plausibility_per_lesion.csv')):
    P.setdefault(r['run'],{})[r['explainer']]=[float(r[k]) for k in ('MA','HE','EX','SE')]
# ViT Attention Rollout EX prints 1.965; the per-image results round to 1.96 (docs/RESULTS_MAP.md, Table 6 rounding)
P['ViT-B16_IDRiD']['AttentionRollout'][2]=1.96
lab={'GradCAM':'Grad-CAM','TransformerAttribution':'Transf. Attr.','TokenNorm (Swin surrogate)':'TokenNorm‡','AttentionRollout':'Attn. Rollout†','InputGrad (baseline)':'Input-Grad.','IntegratedGradients':'Integrated Grad.'}
order={'Swin-T_IDRiD':['GradCAM','TransformerAttribution','TokenNorm (Swin surrogate)','InputGrad (baseline)','IntegratedGradients'],
       'ViT-B16_IDRiD':['GradCAM','TransformerAttribution','AttentionRollout','InputGrad (baseline)','IntegratedGradients']}
fig,axes=plt.subplots(1,2,figsize=(7.0,2.5))
cmap=plt.get_cmap('Greens'); norm=LogNorm(1,25)
for ax,(run,title) in zip(axes,[('Swin-T_IDRiD','Swin-Tiny · IDRiD'),('ViT-B16_IDRiD','ViT-B/16 · IDRiD')]):
    rows=order[run]; M=np.array([P[run][e] for e in rows])
    ax.imshow(M,cmap=cmap,norm=norm,aspect='auto')
    for i in range(M.shape[0]):
        for j in range(4):
            v=M[i,j]; ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7.5,color='white' if v>6 else '#1a1a1a')
    ax.set_xticks(range(4)); ax.set_xticklabels(['MA','HE','EX','SE'])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab[e] for e in rows])
    ax.set_title(title,fontsize=8.5,fontweight='bold'); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlabel('Lesion type')
fig.subplots_adjust(left=0.165,right=0.86,top=0.88,bottom=0.2,wspace=0.62)
cax=fig.add_axes([0.885,0.2,0.015,0.68])
cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=[1,2,5,10,20]); cb.ax.set_yticklabels(['1','2','5','10','20'])
cb.set_label('AUPRC ratio (1 = chance; log scale)',fontsize=7)
fig.savefig(OUT+'Fig4_Plausibility_Lesion.png',dpi=600,facecolor='white'); plt.close(fig)
print('ok')

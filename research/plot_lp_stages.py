"""Publication-exportable LP comparison figure; descriptive single-seed results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/analysis_lp_20261117'
r=json.loads((OUT/'summary.json').read_text())
lookup={(x['setting'],x['group']):x for x in r['metrics']}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(11.4,4.7),gridspec_kw={'width_ratios':[1.25,1]})
names=['full','center_only','center_repeat','hop_mismatch']
labels=['Full HO','Center\nonly','Center\nrepeat','Hop\nmismatch']
values=[lookup['simteg_original',n]['official_accuracy_pct'] for n in names]
bars=axes[0].bar(labels,values,color=['#277da8','#a5becb','#a5becb','#d38c63'],width=.66)
axes[0].bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=4)
axes[0].set_ylim(0,100)
axes[0].set_ylabel('Official accuracy (%)')
axes[0].set_title('(a) Existing SimTeG HO model\nOriginal protocol, 680 records')
axes[0].axhline(379/680*100,color='#777',ls='--',lw=1,label='Always-no reference')
axes[0].legend(frameon=False,loc='upper right',fontsize=8)
axes[0].set_axisbelow(True);axes[0].yaxis.grid(alpha=.18)
matrix=np.array([[lookup['bert_full','full']['official_accuracy_pct'],lookup['bert_full','center_repeat']['official_accuracy_pct']],
                 [lookup['bert_center_train','full']['official_accuracy_pct'],lookup['bert_center_train','center_repeat']['official_accuracy_pct']]])
im=axes[1].imshow(matrix,vmin=50,vmax=90,cmap='Blues',aspect='auto')
axes[1].set_xticks([0,1],['Full HO','Center-repeat'])
axes[1].set_yticks([0,1],['Full HO','Center-repeat'])
axes[1].set_xlabel('Test input');axes[1].set_ylabel('Training input')
axes[1].set_title('(b) Frozen BERT, disjoint LP context\n651 unique pairs; all four cells 100% valid')
for i in range(2):
    for j in range(2):
        axes[1].text(j,i,f'{matrix[i,j]:.2f}%',ha='center',va='center',fontsize=16,color='white' if matrix[i,j]>76 else '#20344b')
fig.colorbar(im,ax=axes[1],label='Official accuracy (%)',fraction=.045,pad=.03)
fig.suptitle('LP input dependence and matched training — training seed 42',fontsize=14,y=.98)
fig.text(.5,.025,'Separate protocols: cross-panel differences are not encoder effects. Point estimates; no training-seed SD.',ha='center',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.07,1,.9],w_pad=3)
for suffix in ('png','pdf','svg'):
    fig.savefig(OUT/f'lp_stages_comparison.{suffix}',dpi=220,bbox_inches='tight')
print(OUT/'lp_stages_comparison.png')

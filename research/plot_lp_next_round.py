"""Plot completed cross-graph LP evidence; error bars are sample SD, not CI."""
from pathlib import Path
import csv
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('results/lp_next_round');out=root/'analysis_20261117'
rows=list(csv.DictReader((out/'llm_confusion.csv').open()))
baseline=list(csv.DictReader((root/'simple_baselines/metrics.csv').open()))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(16,4.9))
colors=['#277da8','#de9054'];modes=['full','center_repeat']
for mode,color,label in zip(modes,colors,['Full HO','Center-repeat']):
    selected=[r for r in rows if r['dataset']=='cora' and r['mode']==mode]
    axes[0].plot([int(r['seed']) for r in selected],[float(r['accuracy_pct']) for r in selected],
                 marker='o',color=color,lw=2,label=label)
    for r in selected:axes[0].annotate(f"{float(r['accuracy_pct']):.2f}",(int(r['seed']),float(r['accuracy_pct'])),
                                     xytext=(0,8 if mode=='full' else -15),textcoords='offset points',ha='center',fontsize=9)
axes[0].set(xticks=[42,43,44],ylim=(65,85),xlabel='Training seed',ylabel='Accuracy (%)',title='(a) Cora: matched training and input')
axes[0].legend(frameon=False,loc='upper right')
x=np.arange(3)
for j,(mode,color,label) in enumerate(zip(modes,colors,['Full HO','Center-repeat'])):
    values=[[float(r['accuracy_pct']) for r in rows if r['mode']==mode and r['dataset']==d] for d in ['arxiv','pubmed','cora']]
    axes[1].bar(x+(j-.5)*.34,[statistics.mean(v) for v in values],width=.34,
                yerr=[statistics.stdev(v) for v in values],capsize=3,color=color,label=label)
axes[1].set(xticks=x,xticklabels=['Arxiv\nsource val','Pubmed\nsource val','Cora\ntarget test'],ylim=(0,105),ylabel='Accuracy (%)',title='(b) Source and target performance')
axes[1].legend(frameon=False,loc='lower left')
names=['Full HO','Center','Endpoint MLP','CN','Cosine'];x=np.arange(len(names))
for j,(key,color,label) in enumerate([('accuracy_pct','#277da8','ACC'),('balanced_accuracy_pct','#70ad92','Balanced ACC')]):
    vals=[];errors=[]
    for i in range(len(names)):
        selected=([r for r in rows if r['dataset']=='cora' and r['mode']==modes[i]] if i<2 else
                  [r for r in baseline if r['dataset']=='cora' and r['method']==['endpoint_MLP','CN','cosine'][i-2]])
        v=[float(r[key]) for r in selected];vals.append(statistics.mean(v));errors.append(statistics.stdev(v) if len(v)>1 else 0)
    axes[2].bar(x+(j-.5)*.34,vals,width=.34,yerr=errors,capsize=3,color=color,label=label)
axes[2].set(xticks=x,xticklabels=names,ylim=(0,105),ylabel='Score (%)',title='(c) Cora: semantic and structural baselines')
axes[2].tick_params(axis='x',rotation=20);axes[2].legend(frameon=False,loc='lower left')
for ax in axes:ax.set_axisbelow(True);ax.yaxis.grid(alpha=.18)
fig.suptitle('Frozen BERT / clean LP protocol: Arxiv + Pubmed → Cora',fontsize=15)
fig.text(.5,.01,'Three training seeds; error bars = sample SD (ddof=1). Deterministic baselines have no seed SD. Source/target candidate difficulty differs.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.055,1,.93])
for suffix in ['png','pdf']:fig.savefig(out/f'lp_round_comparison.{suffix}',dpi=200,bbox_inches='tight')
print(out/'lp_round_comparison.png')

"""Render static figures from the audited results; requires numpy/matplotlib."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/analysis_20261116'
report=json.loads((OUT/'summary.json').read_text())
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,
                     'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
BLUE='#2878B5';ORANGE='#E68A2E';GRAY='#59636E'

def save(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=240,facecolor='white',bbox_inches='tight')
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11.8,6.5),sharey=True)
for ax,template in zip(axes,['HO','ND']):
    x=np.arange(2);width=.31
    for offset,lr,color in [(-width/2,'2e-3',BLUE),(width/2,'2e-5',ORANGE)]:
        records=[next(m for m in report['table5'] if (m['template'],m['projector'],m['learning_rate'])==(template,arch,lr))
                 for arch in ['linear','mlp2']]
        bars=ax.bar(x+offset,[m['accuracy_pct'] for m in records],width=width,color=color,zorder=3)
        for bar,m in zip(bars,records):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.4,
                    f"{m['accuracy_pct']:.2f}%\n{m['correct']}/680",ha='center',va='bottom',fontsize=10,
                    zorder=5,bbox=dict(facecolor='white',edgecolor='none',alpha=.92,pad=.5))
    ref=87.35 if template=='HO' else 86.47
    ax.axhline(ref,ls='--',lw=1.4,color=GRAY,zorder=4,label=f'论文 {template} 参考：{ref:.2f}%')
    ax.set_title(f'{template} 模板',fontsize=15,pad=16,fontweight='bold')
    ax.set_xticks(x,['Linear','两层 MLP'])
    ax.set_ylim(0,105);ax.set_yticks(np.arange(0,101,20));ax.set_xlim(-.6,1.6)
    ax.grid(axis='y',alpha=.18,zorder=0)
    ax.legend(loc='lower left',framealpha=.95,fontsize=10)
axes[0].set_ylabel('Cora 链接预测准确率（%）')
fig.suptitle('Table 5：模板 × Projector × 学习率完整对照',fontsize=18,y=.99,fontweight='bold')
fig.text(.5,.925,'Arxiv + Pubmed LP 训练 → Cora 零样本 LP 测试',ha='center',fontsize=12,color=GRAY)
fig.legend(handles=[Patch(facecolor=BLUE,label='学习率 2e-3'),Patch(facecolor=ORANGE,label='学习率 2e-5')],
           loc='upper center',bbox_to_anchor=(.5,.89),ncol=2,frameon=False)
fig.subplots_adjust(top=.77,bottom=.20,wspace=.12)
fig.text(.07,.09,'统一：训练 seed=42、1 epoch、有效 batch=16、SimTeG、冻结 Vicuna-7B；每格使用一次原始推理。',fontsize=9,color=GRAY)
fig.text(.07,.055,'评分沿用官方 yes 匹配规则；柱上标注正确数/680。单次结果不画误差条；论文虚线仅供参考，配方未完全确认。',fontsize=9,color=GRAY)
save(fig,'table5_eight_configs')

fig,axes=plt.subplots(1,2,figsize=(13,6.4),gridspec_kw={'width_ratios':[1.08,1]})
groups=['E_official','E','ET','T']
generation={m['group']:m for m in report['generation']}
scoring={m['group']:m for m in report['candidate_scoring']}
x=np.arange(4);width=.35
for offset,values,errors,color,label in [(-width/2,[generation[g]['mean_pct'] for g in groups],
        [generation[g]['sample_std_pp'] for g in groups],BLUE,'自由生成：三次均值 ± SD'),
        (width/2,[scoring[g]['accuracy_pct'] for g in groups],None,ORANGE,'候选打分：完整类名平均对数概率')]:
    bars=axes[0].bar(x+offset,values,width,yerr=errors,capsize=3,color=color,label=label,zorder=3)
    for bar,v in zip(bars,values):
        axes[0].text(bar.get_x()+bar.get_width()/2,v+2.2,f'{v:.2f}',ha='center',fontsize=9)
axes[0].set_xticks(x,['原始 E','对齐 E','E+T','T'])
axes[0].set_ylim(0,70);axes[0].set_ylabel('准确率（%）');axes[0].grid(axis='y',alpha=.18)
axes[0].set_title('候选打分未恢复分类表现',fontsize=14,pad=14)
axes[0].legend(loc='upper left',fontsize=8,frameon=False)
order=['T','center_only','center_repeat','ET','neighbor_shuffle']
names=['仅中心文本 T','单个中心表示 + 文本','中心表示重复五次 + 文本','完整 HO + 文本','错配多跳表示 + 文本']
yy=np.arange(len(order))
for i,g in enumerate(order):
    m=generation[g]
    color=BLUE if g=='ET' else GRAY
    axes[1].errorbar(m['mean_pct'],i,xerr=m['sample_std_pp'],fmt='o',capsize=4,color=color,markersize=7)
    axes[1].text(61.1,i,f"{m['mean_pct']:.2f} ± {m['sample_std_pp']:.2f}",va='center',fontsize=10,color=color)
axes[1].set_yticks(yy,names);axes[1].invert_yaxis();axes[1].set_ylim(4.6,-.7)
axes[1].set_xlim(53.5,64.7);axes[1].set_xticks([54,56,58,60]);axes[1].set_xlabel('准确率（%）；误差线为三次推理 SD')
axes[1].axvline(generation['T']['mean_pct'],color=GRAY,ls=':',lw=1,alpha=.6)
axes[1].grid(axis='x',alpha=.18);axes[1].set_title('正确多跳配对的净收益较小',fontsize=14,pad=14)
fig.suptitle('Table 8 诊断：输出协议与多跳表示',fontsize=18,y=.99,fontweight='bold')
fig.subplots_adjust(left=.06,right=.98,top=.85,bottom=.20,wspace=.62)
fig.text(.06,.095,'所有条件使用同一 Cora 测试集（542 节点）；图条件使用同一个 HO/linear/lr2e-3、训练 seed42 的 nd Projector。',fontsize=9,color=GRAY)
fig.text(.06,.055,'候选打分是独立协议，不替换自由生成指标；SD 只反映推理波动。邻域扰动仅使用一个固定置换，尚非多训练种子结论。',fontsize=9,color=GRAY)
save(fig,'table8_diagnostics')
print('Saved PNG/PDF/SVG figures in',OUT)

"""Source-only stratified bootstrap; target labels never select thresholds."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,csv,json
import numpy as np
from research.run_lp_simple_baselines import select_threshold,metrics
from research.check_lp_prediction import sha


def choose(sets,names,rng=None):
 scores=[];labels=[];graphs=[]
 for i,name in enumerate(names):
  y,s=sets[name]
  idx=np.arange(len(y)) if rng is None else np.concatenate([rng.choice(np.flatnonzero(y==c),size=int((y==c).sum()),replace=True) for c in [0,1]])
  scores.append(s[idx]);labels.append(y[idx]);graphs.append(np.full(len(idx),i))
 return select_threshold(np.concatenate(scores),np.concatenate(labels),np.concatenate(graphs))[0]


def rates(y,s,t):
 pred=s>=t
 return dict(accuracy_pct=float((pred==y).mean()*100),balanced_accuracy_pct=50*(pred[y==1].mean()+(~pred[y==0]).mean()),
             FPR_pct=float(pred[y==0].mean()*100),TPR_pct=float(pred[y==1].mean()*100))


def write(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--direction',choices=['pubmed','cora'],required=True)
 p.add_argument('--repeats',type=int,default=500)
 p.add_argument('--bootstrap-seed',type=int,default=20261118)
 a=p.parse_args();assert a.repeats>1
 root=Path('results/lp_phase2' if a.direction=='pubmed' else 'results/lp_cora_scores')
 sources=['arxiv','cora'] if a.direction=='pubmed' else ['arxiv','pubmed'];target=a.direction
 out=root/'threshold_stability';out.mkdir(parents=True,exist_ok=True)
 all_metrics=[];draws=[];summary=[];hashes={}
 for seed in [42,43,44]:
  for mode in ['full','center_repeat']:
   sets={}
   for ds in sources+[target]:
    split='test' if ds==target else 'val';path=root/f'binary_scores/seed{seed}_{mode}/{ds}_{split}.jsonl'
    recipe=json.loads(path.with_suffix('.recipe.json').read_text());assert recipe['pilot'] is False
    assert recipe['seed']==seed and recipe['ablation']==mode and recipe['dataset']==ds
    if target=='pubmed':ref=Path(f'results/lp_phase2/arxiv-cora_to_pubmed/seed{seed}_{mode}/{ds}_{split}_{mode}.jsonl')
    elif ds=='cora':ref=Path(f'results/lp_clean/arxiv-pubmed_to_cora_trainseed{seed}_{mode}/screen/cora_{mode}.jsonl')
    else:ref=Path(f'results/lp_next_round/source_validation/seed{seed}_{mode}/{ds}_val_{mode}.jsonl')
    assert sha(ref)==recipe['input_sha256']
    z=[json.loads(l) for l in path.open()];r=[json.loads(l) for l in ref.open()]
    assert len(z)==len(r) and all(x['pair']==v['question_id'] and x['gt']==v['gt'] and x['original_text']==v['text'] for x,v in zip(z,r))
    y=np.array([int(x['gt']=='yes') for x in z]);s=np.array([x['score'] for x in z]);assert np.isfinite(s).all() and set(y)=={0,1}
    sets[ds]=(y,s);hashes[str(path)]=sha(path)
   rules={'equal_source':sources,**{g+'_only':[g] for g in sources}}
   thresholds={'candidate_zero':0.,**{k:choose(sets,v) for k,v in rules.items()}}
   for rule,t in thresholds.items():
    for ds,(y,s) in sets.items():
     all_metrics.append(dict(seed=seed,mode=mode,dataset=ds,rule=rule,**metrics(y,s,t)))
   for rule,names in rules.items():
    # Same seed across rules gives reproducible resampling; only source arrays accessed by choose.
    rng=np.random.default_rng(a.bootstrap_seed+seed)
    local=[]
    for rep in range(a.repeats):
     t=choose(sets,names,rng);v=rates(*sets[target],t)
     row=dict(seed=seed,mode=mode,rule=rule,replicate=rep,threshold=t,**v);draws.append(row);local.append(row)
    stats=dict(seed=seed,mode=mode,rule=rule,repeats=a.repeats,original_threshold=thresholds[rule])
    for k in ['threshold','accuracy_pct','balanced_accuracy_pct','FPR_pct','TPR_pct']:
     q=np.quantile([r[k] for r in local],[.025,.5,.975]);stats.update({k+'_p025':q[0],k+'_median':q[1],k+'_p975':q[2]})
    summary.append(stats)
   print('Completed',target,seed,mode,flush=True)
 write(out/'metrics.csv',all_metrics);write(out/'bootstrap_draws.csv',draws);write(out/'bootstrap_summary.csv',summary)
 (out/'protocol.json').write_text(json.dumps(dict(target=target,sources=sources,repeats=a.repeats,bootstrap_seed=a.bootstrap_seed,
  primary_rule='equal_source',resampling='within each source graph and label; same counts with replacement',
  caveat='Percentiles describe validation-sampling sensitivity at fixed weights and fixed target pairs; not training-seed CI or graph-dependent-sample CI. Source validation metrics are selection-biased. Never select rule by target scores.',
  code_sha256=sha(Path(__file__)),inputs=hashes),indent=2)+'\n')
 lines=['# 源验证阈值稳定性与固定协议评分','',
  'equal_source始终为主规则；单源阈值只作预先定义的诊断，不根据目标分数选赢家。',
  'bootstrap_summary.csv中的2.5%/97.5%分位数描述固定权重下源验证抽样敏感性；不是训练种子置信区间，也没有校正节点对之间的依赖。',
  '目标标签只计算指标；没有用于阈值选择。metrics.csv提供两个源验证与目标的ACC/BA/AUC/AP。','',
  '|模式|规则|目标ACC均值±训练种子SD|目标AUC均值±训练种子SD|','|---|---|---:|---:|']
 for mode in ['full','center_repeat']:
  for rule in thresholds:
   z=[r for r in all_metrics if r['mode']==mode and r['rule']==rule and r['dataset']==target]
   acc=np.array([r['accuracy_pct'] for r in z]);auc=np.array([r['roc_auc'] for r in z])
   lines.append(f'|{mode}|{rule}|{acc.mean():.2f} ± {acc.std(ddof=1):.2f}|{auc.mean():.4f} ± {auc.std(ddof=1):.4f}|')
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print('Saved',out)
if __name__=='__main__':main()

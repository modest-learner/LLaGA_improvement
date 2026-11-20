"""Three training seeds: source gains, neighborhood gains, difference of gains."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/lp_source_control'));p.add_argument('--output',type=Path);a=p.parse_args()
 output=a.output or a.root/'three_seed_summary';allrows=[];inputs={}
 for seed,sub in [(42,'summary'),(43,'seed43/summary'),(44,'seed44/summary')]:
  f=a.root/sub/'comparison.csv';rows=list(csv.DictReader(f.open()));assert len(rows)==24
  keys={(r['sources'],r['mode'],r['dataset'],r['protocol']) for r in rows};assert len(keys)==24
  assert all(int(r['seed'])==seed for r in rows)
  allrows.extend(rows);inputs[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 metrics=['original_accuracy_pct','accuracy_pct','balanced_accuracy_pct','roc_auc','average_precision']
 records=[];paired=[]
 for protocol in ['candidate_zero','arxiv_val_threshold']:
  target=[r for r in allrows if r['dataset']=='pubmed' and r['protocol']==protocol]
  lookup={(r['sources'],r['mode'],int(r['seed'])):r for r in target};assert len(lookup)==12
  for source in ['arxiv_only','arxiv_cora']:
   for mode in ['full','center_repeat']:
    r=dict(sources=source,mode=mode,protocol=protocol,n_training_seeds=3)
    for metric in metrics:
     values=np.array([float(lookup[source,mode,s][metric]) for s in [42,43,44]])
     r[metric+'_mean']=float(values.mean());r[metric+'_std']=float(values.std(ddof=1))
    records.append(r)
  for seed in [42,43,44]:
   for metric in metrics:
    sf=float(lookup['arxiv_only','full',seed][metric]);sc=float(lookup['arxiv_only','center_repeat',seed][metric])
    mf=float(lookup['arxiv_cora','full',seed][metric]);mc=float(lookup['arxiv_cora','center_repeat',seed][metric])
    paired.append(dict(seed=seed,protocol=protocol,metric=metric,source_gain_full=mf-sf,source_gain_center=mc-sc,neighbor_gain_single=sf-sc,neighbor_gain_multi=mf-mc,interaction=(mf-mc)-(sf-sc)))
 gains=[]
 for protocol in ['candidate_zero','arxiv_val_threshold']:
  for metric in metrics:
   rows=[r for r in paired if r['protocol']==protocol and r['metric']==metric];assert len(rows)==3
   for effect in ['source_gain_full','source_gain_center','neighbor_gain_single','neighbor_gain_multi','interaction']:
    v=np.array([r[effect] for r in rows]);gains.append(dict(protocol=protocol,metric=metric,effect=effect,mean=float(v.mean()),std=float(v.std(ddof=1)),positive_seeds=int((v>0).sum()),negative_seeds=int((v<0).sum())))
 output.mkdir(parents=True,exist_ok=True)
 for name,rows in [('all_metrics.csv',allrows),('target_mean_std.csv',records),('paired_by_seed.csv',paired),('paired_mean_std.csv',gains)]:
  with (output/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# 等预算单源/多源 × 邻域：三个训练种子','', '±为seed42/43/44的样本标准差（ddof=1）。源阈值对照均只用Arxiv验证集。原生成ACC独立报告，不按目标结果选择协议。','', '|source|mode|protocol|original ACC|candidate ACC|BA|AUC|AP|','|---|---|---|---:|---:|---:|---:|---:|']
 for r in records:lines.append('|'+ '|'.join([r['sources'],r['mode'],r['protocol']]+[f"{r[k+'_mean']:.4f} ± {r[k+'_std']:.4f}" for k in metrics])+'|')
 lines+=['','## Paired gains','', '|protocol|metric|effect|mean ± std|positive seeds|negative seeds|','|---|---|---|---:|---:|---:|']
 for r in gains:
  if r['metric'] in ['original_accuracy_pct','roc_auc']:
   lines.append(f"|{r['protocol']}|{r['metric']}|{r['effect']}|{r['mean']:.4f} ± {r['std']:.4f}|{r['positive_seeds']}|{r['negative_seeds']}|")
 lines+=['','交互效应=(多源Full−多源Center)−(单源Full−单源Center)。百分比指标差值单位为百分点；AUC/AP差值保持0–1量纲。三种子不足以仅凭均值或正号声称显著性。']
 (output/'REPORT.md').write_text('\n'.join(lines)+'\n');(output/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n');print('Saved',output/'REPORT.md')
if __name__=='__main__':main()

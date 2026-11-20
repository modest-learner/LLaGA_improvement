"""Aggregate paired seeds; never mix inference repeats with training seeds."""
from pathlib import Path
import csv,argparse,json
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed44-csv',type=Path,default=Path('results/lp_weight_control_seed44/summary/comparison.csv'));p.add_argument('--output',type=Path,default=Path('results/lp_weight_control_three_seeds'));a=p.parse_args()
 paths=[Path('results/lp_random_control/comparison.csv'),Path('results/lp_weight_control_seed43/summary/comparison.csv'),a.seed44_csv]
 table=[]
 for seed,path in zip([42,43,44],paths):
  rows=list(csv.DictReader(path.open()));assert len(rows)==18
  for r in rows:
   assert int(r['seed'])==seed
   if r['dataset']=='pubmed':table.append(r)
 assert len(table)==18
 values=['original_accuracy_pct','accuracy_pct','balanced_accuracy_pct','roc_auc','average_precision','FPR_pct','TPR_pct']
 summary=[];deltas=[]
 for protocol in ['candidate_zero','equal_source']:
  for model in ['baseline','semantic_weighted','random_weighted']:
   rows=[r for r in table if r['protocol']==protocol and r['model']==model];assert sorted(int(r['seed']) for r in rows)==[42,43,44]
   record=dict(model=model,protocol=protocol,n_training_seeds=3)
   for k in values:
    v=np.array([float(r[k]) for r in rows]);record[k+'_mean']=float(v.mean());record[k+'_std']=float(v.std(ddof=1))
   summary.append(record)
   if model!='baseline':
    base={int(r['seed']):r for r in table if r['protocol']==protocol and r['model']=='baseline'}
    record=dict(model=model,protocol=protocol,n_training_seeds=3)
    for k in values:
     v=np.array([float(r[k])-float(base[int(r['seed'])][k]) for r in rows]);record[k+'_mean']=float(v.mean());record[k+'_std']=float(v.std(ddof=1))
    deltas.append(record)
 a.output.mkdir(parents=True,exist_ok=True)
 for name,rows in [('all_seed_metrics.csv',table),('mean_std.csv',summary),('paired_deltas.csv',deltas)]:
  with (a.output/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# Arxiv+Cora → Pubmed 加权对照：三个训练种子','', '±为三个独立训练种子的样本标准差（ddof=1），不是推理重复或置信区间。原生成与两种评分协议分别报告。','', '|model|protocol|original ACC|candidate ACC|BA|AUC|AP|','|---|---|---:|---:|---:|---:|---:|']
 for r in summary:
  cells=[r['model'],r['protocol']]+[f"{r[k+'_mean']:.4f} ± {r[k+'_std']:.4f}" for k in values[:5]]
  lines.append('|'+ '|'.join(cells)+'|')
 lines+=['','配对增益（逐训练种子减去对应原基线，再统计）见 paired_deltas.csv；不能从均值大小直接推断统计显著性。']
 (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
 (a.output/'inputs.json').write_text(json.dumps([str(p) for p in paths],indent=2)+'\n')
 print('Saved',a.output/'REPORT.md')
if __name__=='__main__':main()

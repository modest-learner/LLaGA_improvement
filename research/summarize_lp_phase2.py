"""Check phase2 coverage and report direction-specific results without mixing protocols."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.audit_lp_outputs import audit
from research.check_lp_prediction import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/lp_phase2'))
    p.add_argument('--data-root',type=Path,default=Path('dataset_research/lp_bert_disjoint_v1'))
    p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();out=a.root/'summary';out.mkdir(parents=True,exist_ok=True)
    missing=[];llm=[];all_baselines=[]
    strata=a.root/'error_strata/status.json'
    if not strata.exists() or not json.loads(strata.read_text()).get('complete'):missing.append(str(strata))
    for direction,target,models in [('arxiv-pubmed_to_cora','cora',['structure','fusion']),
                                     ('arxiv-cora_to_pubmed','pubmed',['endpoint','structure','fusion'])]:
        d=a.root/direction/'pair_baselines';status=d/'status.json'
        if not status.exists() or not (d/'metrics.csv').exists():
            missing.append(str(status));continue
        state=json.loads(status.read_text());assert state['complete']
        config=json.loads((d/'run_config.json').read_text())
        assert state['recipe_sha256']==sha(d/'run_config.json')
        assert config['target']==target and config['models']==models and config['seeds']==[42,43,44]
        assert config['sources']==direction.split('_to_')[0].split('-')
        rows=list(csv.DictReader((d/'metrics.csv').open()))
        expected={(method,seed,n) for method,seed in
                  [(m,'') for m in ['cosine','CN','AA','RA','PA']]+[(m+'_MLP',str(s)) for m in models for s in [42,43,44]]
                  for n in config['sources']+[target]}
        assert len(rows)==len(expected)==state['rows']
        assert {(r['method'],r['seed'],r['dataset']) for r in rows}==expected
        all_baselines.extend(dict(direction=direction,**r) for r in rows)
    for mode in ['full','center_repeat']:
        cp=Path(f'checkpoints/lp_phase2/llaga-HO-arxiv-cora-lp-linear-bert-lr2e-3-seed42-bs16-{mode}')
        for dataset in ['arxiv','cora','pubmed']:
            split='test' if dataset=='pubmed' else 'val'
            path=a.root/f'arxiv-cora_to_pubmed/seed42_{mode}/{dataset}_{split}_{mode}.jsonl'
            if not path.exists():missing.append(str(path));continue
            command=[sys.executable,'research/check_lp_prediction.py','--checkpoint',str(cp),'--output',str(path),
                     '--dataset',dataset,'--split',split,'--ablation',mode,'--data-root',str(a.data_root)]
            recipe=a.root/'recipe.json'
            if recipe.exists():command+=['--model-base',json.loads(recipe.read_text())['model_base']]
            result=subprocess.run(command,capture_output=True,text=True)
            if result.returncode in (10,20):missing.append(str(path)+' (incomplete)');continue
            if result.returncode:raise RuntimeError(result.stdout+result.stderr)
            score,_=audit(path,a.data_root)
            pos=score['positive'];neg=score['n']-pos
            llm.append(dict(mode=mode,seed=42,dataset=dataset,split=split,**score,
                balanced_accuracy_pct=50*(score['TP']/pos+score['TN']/neg)))
    for filename,rows in [('llm.csv',llm),('baselines.csv',all_baselines)]:
        if rows:
            with (out/filename).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# LP第二轮诊断与新目标验证','',f'完整：{not missing}；新方向LLM文件：{len(llm)}/6。',
           '', 'LLM新方向仅seed42，不能给训练种子标准差；简单MLP为三个种子。源验证用于简单模型选择，有选择偏差。',
           '', '|目标|方法|ACC均值|训练种子样本SD|','|---|---|---:|---:|']
    for target,direction in [('cora','arxiv-pubmed_to_cora'),('pubmed','arxiv-cora_to_pubmed')]:
        selected=[r for r in all_baselines if r['direction']==direction and r['dataset']==target]
        for method in sorted({r['method'] for r in selected}):
            values=[float(r['accuracy_pct']) for r in selected if r['method']==method]
            sd=f'{statistics.stdev(values):.2f}' if len(values)>1 else '不适用'
            lines.append(f'|{target}|{method}|{statistics.mean(values):.2f}%|{sd}|')
    for r in llm:
        if r['dataset']=='pubmed':lines.append(f"|pubmed|LLaGA {r['mode']}|{r['overall_accuracy_pct']:.2f}%|单种子|")
    lines+=['','Cora原端点MLP和LLaGA三种子参照：results/lp_next_round/analysis_20261117/REPORT.md。',
            '不同目标分数不能直接相减解释图分布因果效应；分层是探索性分析，稀疏分组需谨慎。']
    if missing:lines+=['','待完成：']+['- '+s for s in missing]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'status.json').write_text(json.dumps(dict(complete=not missing,missing=missing,llm_files=len(llm)),indent=2)+'\n')
    print('Summary:',out/'REPORT.md','complete=',not missing)
    if a.require_complete and missing:raise SystemExit(1)


if __name__=='__main__':main()

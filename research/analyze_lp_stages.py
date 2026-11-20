"""Independently audit and compare completed LP stages; no model inference."""
import ast
from collections import Counter
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from eval.lp_inputs import parse_binary, donors_for_pair, endpoint_prompt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/analysis_lp_20261117'
GROUPS=('full','center_only','center_repeat','hop_mismatch','endpoint_text')


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def sha(path):
    d=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            d.update(b)
    return d.hexdigest()


def csv_write(path,rows):
    if not rows:
        return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    # Evaluate the original function body unchanged, independent of our auxiliary parser.
    tree=ast.parse((ROOT/'eval/eval_res.py').read_text())
    fn=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='eval_lp']
    env=dict(json=json,args=SimpleNamespace(sample=0))
    exec(compile(ast.Module(body=fn,type_ignores=[]),'official_lp_scorer','exec'),env)
    configurations={
        'simteg_original':(ROOT/'results/lp_input_screen/HO_linear_simteg_trainseed42_greedy',ROOT/'dataset'),
        'bert_full':(ROOT/'results/lp_clean/arxiv-pubmed_to_cora_trainseed42_full/screen',ROOT/'dataset_research/lp_bert_disjoint_v1'),
        'bert_center_train':(ROOT/'results/lp_clean/arxiv-pubmed_to_cora_trainseed42_center_repeat/screen',ROOT/'dataset_research/lp_bert_disjoint_v1'),
    }
    metrics=[];records={};predictions={};artifact_checks={};weights={};invalid=[];lexical=[]
    for setting,(directory,data_root) in configurations.items():
        refs=read(data_root/'cora/edge_sampled_2_10_only_test.jsonl')
        data=torch.load(data_root/'cora/processed_data.pt',map_location='cpu')
        stored={r['file'].split('/')[-1]:r for r in csv.DictReader((directory/'audit/metrics.csv').open())}
        for group in GROUPS:
            path=directory/f'cora_{group}.jsonl'
            rows=read(path)
            assert len(rows)==len(refs)
            cfg=rows[0]['lp_diagnostic_config']
            assert all(r['lp_diagnostic_config']==cfg for r in rows)
            assert cfg['greedy'] and cfg['seed']==0 and cfg['start']==cfg['end']==-1
            assert cfg['code_sha256']==sha(ROOT/'eval/eval_pretrain.py')
            assert cfg['input_code_sha256']==sha(ROOT/'eval/lp_inputs.py')
            assert cfg['prompt_sha256']==sha(data_root/'cora/edge_sampled_2_10_only_test.jsonl')
            assert cfg['data_root']==str(data_root.resolve())
            if setting!='simteg_original':
                assert cfg['manifest_sha256']==sha(data_root/'cora/clean_lp_manifest.json')
            if group!='endpoint_text':
                ckpt=Path(cfg['model_path'])
                assert cfg['projector_sha256']==sha(ckpt/'mm_projector.bin')
                if str(ckpt) not in weights:
                    tensors=torch.load(ckpt/'mm_projector.bin',map_location='cpu')
                    assert all(torch.isfinite(v).all() for v in tensors.values())
                    state=json.loads((ckpt/'trainer_state.json').read_text())
                    assert state['global_step']==state['max_steps']
                    weights[str(ckpt)]={'steps':state['global_step'],'finite':True,'sha256':cfg['projector_sha256']}
            counts=Counter();pred=[];pairs=set();examples=Counter()
            for i,(r,ref) in enumerate(zip(rows,refs)):
                assert r['question_id']==ref['id'] and r['gt']==ref['conversations'][1]['value']
                pair=tuple(sorted(r['question_id']));pairs.add(pair)
                if group=='endpoint_text':
                    assert r['graph']==[] and r['prompt']==endpoint_prompt(data.raw_texts,r['question_id'],2000)
                else:
                    length=1 if group=='center_only' else 5
                    assert r['graph']==[[v]*length for v in r['question_id']]
                    if group=='hop_mismatch':
                        assert r['donor_ids']==donors_for_pair(r['question_id'],data.num_nodes,42)
                    else:
                        assert r['donor_ids'] is None
                label=r['gt'].strip().lower()
                p=parse_binary(r['text']);pred.append(p)
                original='yes' if 'yes' in r['text'].strip() else 'no'
                counts['official_correct']+=original==label
                counts['correct']+=p==label
                counts['valid']+=p is not None
                counts['positive']+=label=='yes'
                counts['empty']+=not r['text'].strip()
                if p is not None:
                    counts[{('yes','yes'):'TP',('yes','no'):'FN',('no','yes'):'FP',('no','no'):'TN'}[label,p]]+=1
                else:
                    invalid.append(dict(setting=setting,group=group,index=i,pair=r['question_id'],gt=label,text=r['text']))
                if original=='yes' and 'yes' not in __import__('re').findall(r'\b(?:yes|no)\b',r['text'].lower()):
                    words=__import__('re').findall(r'\w*yes\w*',r['text'])
                    examples.update(words)
                    lexical.append(dict(setting=setting,group=group,index=i,pair=r['question_id'],gt=label,trigger_words=words,text=r['text']))
            n=len(rows);positive=counts['positive'];negative=n-positive
            m=dict(setting=setting,group=group,n=n,unique_pairs=len(pairs),positive=positive,negative=negative,
                   correct=counts['correct'],official_correct=counts['official_correct'],
                   official_accuracy_pct=100*counts['official_correct']/n,accuracy_pct=100*counts['correct']/n,
                   valid_rate_pct=100*counts['valid']/n,conditional_accuracy_pct=100*counts['correct']/counts['valid'],
                   invalid=n-counts['valid'],empty=counts['empty'],
                   **{k:counts[k] for k in ('TP','FP','FN','TN')},
                   positive_recall_pct=100*counts['TP']/positive,
                   negative_recall_pct=100*counts['TN']/negative,
                   balanced_accuracy_pct=50*(counts['TP']/positive+counts['TN']/negative),
                   lexical_substring_triggers=json.dumps(examples))
            # Balanced accuracy only measures a binary classifier if all responses valid.
            if m['invalid']:
                m['balanced_accuracy_pct']=None
            old=stored[path.name]
            for new_key,old_key in [('official_accuracy_pct','official_accuracy_pct'),('accuracy_pct','overall_accuracy_pct'),('valid_rate_pct','valid_rate_pct')]:
                assert abs(m[new_key]-float(old[old_key]))<1e-9
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):
                env['eval_lp'](str(path))
            score=float(stream.getvalue().split('acc: ')[-1])
            assert abs(score-m['official_accuracy_pct']/100)<0.000051
            metrics.append(m);records[setting,group]=rows;predictions[setting,group]=pred
            artifact_checks[f'{setting}/{group}']={'records':n,'sha256':sha(path),'official_score_checked':True}
    comparisons=[]
    for name,a,b in [
        ('simteg_full_vs_repeat',('simteg_original','full'),('simteg_original','center_repeat')),
        ('simteg_full_vs_mismatch',('simteg_original','full'),('simteg_original','hop_mismatch')),
        ('bert_full_vs_repeat_intervention',('bert_full','full'),('bert_full','center_repeat')),
        ('bert_full_vs_mismatch',('bert_full','full'),('bert_full','hop_mismatch')),
        ('bert_matched_full_vs_center',('bert_full','full'),('bert_center_train','center_repeat')),
    ]:
        x,y=records[a],records[b]
        assert [r['question_id'] for r in x]==[r['question_id'] for r in y]
        left=predictions[a];right=predictions[b]
        gain=sum(p==r['gt'] and q!=r['gt'] for p,q,r in zip(left,right,x))
        loss=sum(p!=r['gt'] and q==r['gt'] for p,q,r in zip(left,right,x))
        comparisons.append(dict(comparison=name,n=len(x),left_gain=gain,left_loss=loss,net_correct_gain=gain-loss,delta_accuracy_pp=100*(gain-loss)/len(x)))
    # T has no trained Projector; confirm the two clean conditions are the same baseline.
    assert [r['text'] for r in records['bert_full','endpoint_text']]==[r['text'] for r in records['bert_center_train','endpoint_text']]
    # Exploratory error strata only; no prompting/threshold/model changes are selected here.
    data=torch.load(ROOT/'dataset_research/lp_bert_disjoint_v1/cora/processed_data.pt',map_location='cpu')
    emb=torch.load(ROOT/'dataset_research/lp_bert_disjoint_v1/cora/bert_x.pt',map_location='cpu')
    emb=torch.nn.functional.normalize(emb,dim=1)
    neighbors=[set() for _ in range(data.num_nodes)]
    for u,v in data.edge_index.t().tolist():
        neighbors[u].add(v)
    errors=[]
    for i,(r,p) in enumerate(zip(records['bert_full','full'],predictions['bert_full','full'])):
        u,v=r['question_id'];gt=r['gt'];cos=float((emb[u]*emb[v]).sum())
        errors.append(dict(index=i,u=u,v=v,gt=gt,pred=p,outcome={('yes','yes'):'TP',('yes','no'):'FN',('no','yes'):'FP',('no','no'):'TN'}[gt,p],
                           degree_u=len(neighbors[u]),degree_v=len(neighbors[v]),common_neighbors=len(neighbors[u]&neighbors[v]),cosine=cos))
    error_summary={}
    for outcome in ('TP','FN','FP','TN'):
        selected=[r for r in errors if r['outcome']==outcome]
        error_summary[outcome]={'n':len(selected),**{k+'_median':statistics.median(r[k] for r in selected) for k in ('common_neighbors','cosine')}}
    csv_write(OUT/'all_metrics.csv',metrics)
    csv_write(OUT/'paired_comparisons.csv',comparisons)
    csv_write(OUT/'bert_full_pair_errors.csv',errors)
    for filename,rows in [('invalid_answers.jsonl',invalid),('incidental_yes_substrings.jsonl',lexical)]:
        (OUT/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report=dict(metrics=metrics,comparisons=comparisons,error_summary=error_summary,
                artifacts=artifact_checks,weights=weights,
                limits=['one training seed for new BERT pair','one Cora target','one donor seed',
                        'old and new change encoder, adjacency/split/normalization/order/training length jointly',
                        'whole-word lexical parser is not a semantic judge','no continuous scores: no AUC/AP'])
    supplemental=ROOT/'results/lp_protocol_audit/full_v1/ho_cache_author_normalization_20261117.json'
    if supplemental.exists():
        report['author_normalization_cache_audit']=json.loads(supplemental.read_text())
    (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Paired:',json.dumps(comparisons,ensure_ascii=False))
    print('Exploratory error strata:',error_summary)
    print('Complete audit:',len(artifact_checks),'files;',sum(r['records'] for r in artifact_checks.values()),'records')
    print('Output:',OUT)


if __name__=='__main__':
    main()

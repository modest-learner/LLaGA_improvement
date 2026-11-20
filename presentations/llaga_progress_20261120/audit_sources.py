from pathlib import Path
import csv,json,hashlib,datetime,statistics,shutil
root=Path(__file__).resolve().parents[2]; out=Path(__file__).parent
paths=[
'results/table1/official_20261114_192146/general_model_audit_20261116.csv',
'results/table1/official_20261114_192146/GENERAL_MODEL_RESULTS_20261116.md',
'results/table5/ho_linear_multiseed_summary.json',
'results/table8/response_validity_audit_20261116/REPORT.md',
'results/table8/HO_linear_simteg_nctrainseed42/audit_20261116/REPORT.md',
'results/lp_next_round/summary/REPORT.md',
'results/lp_phase2/pubmed_three_seed_summary/REPORT.md',
'results/lp_phase2/pubmed_three_seed_summary/metrics.csv',
'results/lp_phase2/pubmed_three_seed_summary/analysis/REPORT.md',
'results/lp_phase2/analysis_20261118/REPORT.md',
'results/lp_cross_direction_analysis/REPORT_20261119.md',
'results/lp_weight_control_seed43/summary/REPORT.md',
'results/lp_weight_control_seed43/summary/two_seed_comparison.csv',
'results/lp_weight_control_seed43/summary/seed_configuration_audit.json',
'results/lp_source_control/summary/ANALYSIS_20261120.md',
'results/lp_source_control/summary/comparison.csv',
'results/lp_source_control/summary/REPORT.md',
]
manifest=[]
for rel in paths:
 p=root/rel
 assert p.is_file(),rel
 b=p.read_bytes();dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
 manifest.append({'path':rel,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'mtime':datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat()})
(out/'source_manifest.json').write_text(json.dumps({'snapshot_time':datetime.datetime.now().isoformat(),'sources':manifest,'pending_reports':{q:(root/q).exists() for q in ['results/lp_weight_control_three_seeds/REPORT.md','results/lp_source_control/three_seed_summary/REPORT.md']}},ensure_ascii=False,indent=2))
def read(rel):return list(csv.DictReader(open(root/rel)))
a=read(paths[0]);assert len(a)==12
assert sum(abs(float(x['delta_pp']))<1 for x in a)==11
w=read('results/lp_weight_control_seed43/summary/two_seed_comparison.csv')
w=[x for x in w if x['protocol']=='candidate_zero' and x['dataset']=='pubmed']
for seed,expected in [(42,-10.624144),(43,3.424006)]:
 vals={x['model']:float(x['original_accuracy_pct']) for x in w if int(x['seed'])==seed}
 assert abs(vals['semantic_weighted']-vals['baseline']-expected)<.001
s=read('results/lp_source_control/summary/comparison.csv');s=[x for x in s if x['dataset']=='pubmed' and x['protocol']=='candidate_zero']
f={x['sources']:x for x in s if x['mode']=='full'}
assert abs(float(f['arxiv_cora']['original_accuracy_pct'])-float(f['arxiv_only']['original_accuracy_pct'])-6.0457836)<.001
assert int(f['arxiv_only']['FP'])-int(f['arxiv_cora']['FP'])==337
(out/'data_checks.json').write_text(json.dumps({'status':'passed','checks':['Table1共12项，11项偏差小于1pp','语义加权seed42/43原生成ACC收益反转','单源多源Full原生成ACC差与候选FP差'],'table1':a,'weighted_two_seeds':w,'source_control_seed42':s},ensure_ascii=False,indent=2))
print('Source snapshots and numeric assertions passed:',len(paths))

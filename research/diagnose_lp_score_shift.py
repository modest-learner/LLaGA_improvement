"""Read saved scores only; never fit a threshold using target labels."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.run_lp_simple_baselines import metrics, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/lp_phase2'))
    a = p.parse_args()
    out = a.root / 'score_diagnostics'; out.mkdir(parents=True, exist_ok=True)
    rows = []; inputs = {}
    for direction in ['arxiv-pubmed_to_cora', 'arxiv-cora_to_pubmed']:
        d = a.root / direction / 'pair_baselines'
        config = json.loads((d / 'run_config.json').read_text())
        inputs[str(d / 'run_config.json')] = sha(d / 'run_config.json')
        for r in csv.DictReader((d / 'metrics.csv').open()):
            suffix = r['method'] + ('_seed' + r['seed'] if r['seed'] else '')
            path = d / f"{r['dataset']}_{suffix}_scores.npz"
            inputs[str(path)] = sha(path)
            with np.load(path) as z:
                scores, labels, threshold = z['scores'], z['labels'], float(z['threshold'])
                assert np.isfinite(scores).all() and len(scores) == int(r['n'])
                calculated = metrics(labels, scores, threshold)
                for key, value in calculated.items():
                    assert abs(value - float(r[key])) < 1e-7, (path, key)
                for label in [0, 1]:
                    x = scores[labels == label]
                    q = np.quantile(x, [.05, .25, .5, .75, .95])
                    rows.append(dict(direction=direction, dataset=r['dataset'],
                        role='target_test' if r['dataset'] == config['target'] else 'source_validation',
                        method=r['method'], seed=r['seed'], label=label, n=len(x),
                        source_selected_threshold=threshold,
                        predicted_positive_pct=float((x >= threshold).mean() * 100),
                        q05=q[0], q25=q[1], median=q[2], q75=q[3], q95=q[4],
                        median_minus_threshold=q[2]-threshold,
                        **{k: calculated[k] for k in ['accuracy_pct','balanced_accuracy_pct','roc_auc','average_precision']}))
    with (out / 'score_distributions.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / 'protocol.json').write_text(json.dumps(dict(
        description='Read-only class-conditional score distributions. Original source-selected thresholds retained. No target fitting.',
        inputs=inputs, code_sha256=sha(Path(__file__))), indent=2)+'\n')
    (out / 'README.md').write_text('# LP分数分布诊断\n\n'
        'score_distributions.csv：每个模型/种子/图分别统计正负例分数分位数与原源验证阈值。\n'
        'label=0的predicted_positive_pct为误报率；label=1为正例召回率。\n'
        '同一模型跨图比较分数位置；不同模型的原始logit绝对值不可直接横比。\n'
        '目标标签只用于事后分组和指标计算，没有重新选择阈值。没有给LLM构造连续评分。\n')
    print('Saved', out, 'rows=', len(rows), flush=True)


if __name__ == '__main__':
    main()

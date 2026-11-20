"""Read-only audit of the downloaded LLaGA data (no model/GPU required).

Run from any directory with the llaga Python environment. The JSON report is
written only to --output. Source files and cached embeddings are never changed.
This audits stored ND trees; it does not certify HO preprocessing provenance.
"""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def audit_graph(root, name):
    directory = root / "dataset" / name
    report = {"files": {}, "jsonl": {}, "overlap": {}}
    for path in sorted(directory.glob("*")):
        if path.is_file():
            report["files"][path.name] = {"bytes": path.stat().st_size}
    required = ["processed_data.pt", "sbert_x.pt"]
    required += [f"sampled_2_10_{s}.jsonl" for s in ("train", "test")]
    required += [f"edge_sampled_2_10_only_{s}.jsonl" for s in ("train", "test")]
    report["missing_core_files"] = [f for f in required if not (directory / f).is_file()]
    report["missing_sbert_ho4_files"] = [
        f"sbert_{h}hop_x.pt" for h in range(1, 5)
        if not (directory / f"sbert_{h}hop_x.pt").is_file()
    ]
    if not (directory / "processed_data.pt").is_file():
        return report
    # These are the user's existing trusted PyG artifacts, not remote downloads.
    data = torch.load(directory / "processed_data.pt", map_location="cpu")
    n = int(data.num_nodes)
    edge = data.edge_index.cpu().numpy().astype(np.int64)
    canonical = np.unique(np.minimum(edge[0], edge[1]) * n + np.maximum(edge[0], edge[1]))

    def in_full_graph(u, v):
        key = min(u, v) * n + max(u, v)
        pos = np.searchsorted(canonical, key)
        return bool(pos < len(canonical) and canonical[pos] == key)

    report["graph"] = {"nodes": n, "stored_directed_edges": int(edge.shape[1]),
                       "unique_undirected_edges_including_loops": int(len(canonical)),
                       "self_loops": int((edge[0] == edge[1]).sum()), "keys": sorted(data.keys())}
    for key in ("train_mask", "val_mask", "test_mask"):
        value = getattr(data, key, None)
        if torch.is_tensor(value):
            report["graph"][key] = {"shape": list(value.shape), "sum": int(value.sum())}
    base = directory / "sbert_x.pt"
    if base.is_file():
        x = torch.load(base, map_location="cpu")
        report["sbert"] = {"shape": list(x.shape), "dtype": str(x.dtype),
                           "node_count_matches": int(x.shape[0]) == n,
                           "all_finite": bool(torch.isfinite(x).all()),
                           "provenance": "UNKNOWN: filename and shape do not certify encoder or training data"}
        del x
    ids_by_file = {}
    for path in sorted(directory.glob("*.jsonl")):
        counts = Counter()
        ids, labels_by_pair = set(), {}
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for raw in stream:
                digest.update(raw)
                if not raw.strip():
                    continue
                counts["records"] += 1
                try:
                    item = json.loads(raw)
                    idx = item["id"]
                    label = item["conversations"][1]["value"].strip().lower()
                    is_pair = isinstance(idx, list)
                    if is_pair:
                        if len(idx) != 2:
                            counts["invalid_pair_length"] += 1
                            continue
                        u, v = map(int, idx)
                        key = tuple(sorted((u, v)))
                        counts["label_" + label] += 1
                        counts["self_pairs"] += int(u == v)
                        valid = 0 <= u < n and 0 <= v < n
                        counts["out_of_range_ids"] += int(not valid)
                        if valid:
                            present = in_full_graph(u, v)
                            counts["positive_absent_from_full_graph"] += int(label == "yes" and not present)
                            counts["negative_present_in_full_graph"] += int(label == "no" and present)
                        if key in labels_by_pair and labels_by_pair[key] != label:
                            counts["conflicting_pair_labels"] += 1
                        labels_by_pair[key] = label
                        trees = item.get("graph", [])
                        if trees and not isinstance(trees[0], list):
                            trees = [trees]
                        visible = root_visible = False
                        for tree in trees:
                            if len(tree) != 111:
                                counts["tree_length_not_111"] += 1
                                continue
                            # ND breadth-first tree: fanout=10, two hops.
                            for child in range(1, len(tree)):
                                parent = (child - 1) // 10
                                a, b = int(tree[parent]), int(tree[child])
                                if a < 0 or b < 0:
                                    continue
                                if tuple(sorted((a, b))) == key:
                                    visible = True
                                    root_visible |= parent == 0
                        counts["records_with_query_edge_in_stored_tree"] += int(visible)
                        counts["records_with_query_edge_at_tree_root"] += int(root_visible)
                    else:
                        key = int(idx)
                        counts["out_of_range_ids"] += int(not 0 <= key < n)
                    counts["duplicate_ids_or_undirected_pairs"] += int(key in ids)
                    ids.add(key)
                except (KeyError, TypeError, ValueError, IndexError) as exc:
                    counts["malformed_records"] += 1
                    report.setdefault("parse_error_examples", []).append(
                        {"file": path.name, "record": counts["records"], "error": str(exc)}
                    )
        ids_by_file[path.name] = ids
        report["jsonl"][path.name] = {"sha256": digest.hexdigest(), "unique_ids_or_pairs": len(ids), **dict(counts)}
    for prefix, suffix in (("sampled_2_10_", ""), ("edge_sampled_2_10_", "only_")):
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            left, right = f"{prefix}{suffix}{a}.jsonl", f"{prefix}{suffix}{b}.jsonl"
            if left in ids_by_file and right in ids_by_file:
                report["overlap"][f"{left} vs {right}"] = len(ids_by_file[left] & ids_by_file[right])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed", "ogbn-arxiv"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.root, text=True).strip()
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
              "limitations": ["No training/inference or HO recomputation was performed.",
                              "Full-graph edge presence alone is not leakage: the model's actual input view must be checked.",
                              "A query edge in a stored ND tree is an input contamination risk requiring trace-through verification.",
                              "JSONL files are hashed; tensor files have size checks only.",
                              "Feature provenance, upstream supervision and cross-dataset text overlap remain unaudited."],
              "datasets": {}}
    for name in args.datasets:
        print("Auditing", name, flush=True)
        result["datasets"][name] = audit_graph(args.root, name)
        r = result["datasets"][name]
        print(json.dumps({"dataset": name, "graph": r.get("graph"),
                          "missing_core_files": r["missing_core_files"],
                          "missing_sbert_ho4_files": r["missing_sbert_ho4_files"],
                          "jsonl": r["jsonl"], "overlap": r["overlap"]}, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Report:", args.output)


if __name__ == "__main__":
    main()

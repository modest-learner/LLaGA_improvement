"""Download official weights, or evaluate them using unchanged upstream scripts.

Examples (run in the llaga environment):
  python research/reproduce_official.py download
  python research/reproduce_official.py download --only-projector
  python research/reproduce_official.py run --gpu 0 --task nc --limit 32
  python research/reproduce_official.py run --gpu 0 --task nc
  python research/reproduce_official.py run --gpu 0 --task lp
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE_ID = "lmsys/vicuna-7b-v1.5-16k"
BASE_REV = "c8df3ca4436a3bce5c4b5877e0117032081852b4"
PROJECTOR_ID = "Runjin/llaga-vicuna-7b-simteg-HO-general_model-2-layer-mlp-projector"
PROJECTOR_REV = "5fbc78bde96eadefeaaf4abd3bf8f070a194148c"


def local_base_complete(base):
    """Check the local index and files before considering a base download."""
    required = ["config.json", "generation_config.json", "tokenizer.model",
                "tokenizer_config.json", "pytorch_model.bin.index.json"]
    try:
        index = json.loads((base / "pytorch_model.bin.index.json").read_text())
        shards = set(index["weight_map"].values())
        if not shards:
            return False
        required += sorted(shards)
        return all((base / name).is_file() and (base / name).stat().st_size > 0
                   for name in required)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Reuse a complete local Vicuna; download missing base and official HO general Projector")
    download.add_argument("--only-projector", action="store_true",
                          help="Download only the official HO general Projector; never request Vicuna")
    run = sub.add_parser("run", help="Run original inference, check coverage, then original scoring")
    run.add_argument("--gpu", required=True, help="One available GPU index or UUID")
    run.add_argument("--task", choices=["nc", "lp"], required=True)
    run.add_argument("--dataset", choices=["cora", "pubmed"], default="cora")
    run.add_argument("--limit", type=int, default=-1, help="-1 for all examples; 32 for a smoke test")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output-dir", type=Path, default=ROOT / "results" / "official_ho_general")
    args = parser.parse_args()
    base = ROOT / "models" / BASE_ID.split("/")[-1]
    projector = ROOT / "models" / PROJECTOR_ID.split("/")[-1]
    if args.command == "download":
        from huggingface_hub import snapshot_download
        downloads = [(PROJECTOR_ID, PROJECTOR_REV, projector)]
        if args.only_projector:
            print("Projector-only download selected; Vicuna will not be downloaded.", flush=True)
        elif local_base_complete(base):
            print(f"Reusing complete local Vicuna at {base}; skipping its download.", flush=True)
        else:
            downloads.insert(0, (BASE_ID, BASE_REV, base))
        for repo, revision, target in downloads:
            print(f"Downloading {repo}@{revision} to {target}", flush=True)
            snapshot_download(repo_id=repo, revision=revision, local_dir=str(target),
                              allow_patterns=["*.json", "*.model", "*.bin"])
        print("Download finished. Next reserve a healthy GPU with sufficient free VRAM.")
        return
    if args.limit == 0 or args.limit < -1:
        parser.error("--limit must be -1 or a positive integer")
    args.output_dir = args.output_dir.resolve()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    required = [base / "config.json", base / "tokenizer.model", base / "tokenizer_config.json",
                base / "pytorch_model.bin.index.json", projector / "config.json", projector / "mm_projector.bin"]
    index = base / "pytorch_model.bin.index.json"
    if index.is_file():
        required += [base / name for name in set(json.loads(index.read_text())["weight_map"].values())]
    data_dir = ROOT / "dataset" / args.dataset
    source = data_dir / ("sampled_2_10_test.jsonl" if args.task == "nc" else "edge_sampled_2_10_only_test.jsonl")
    required += [source, data_dir / "processed_data.pt"]
    for encoder in ("sbert", "roberta", "e5"):
        required.append(data_dir / f"simteg_{encoder}_x.pt")
        if args.task == "nc":
            required += [data_dir / f"simteg_{encoder}_{i}hop_x.pt" for i in range(1, 5)]
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise SystemExit("Missing files (prepare/download before running):\n" + "\n".join(missing))
    suffix = "full" if args.limit == -1 else f"n{args.limit}"
    output = args.output_dir / f"{args.dataset}_{args.task}_seed{args.seed}_{suffix}.jsonl"
    if output.exists():
        raise SystemExit(f"Output already exists: {output}\nUse a different --output-dir for a new run. Automatic resume is disabled by this wrapper.")
    # Set device visibility before importing torch/transformers.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    from transformers import set_seed
    if not torch.cuda.is_available():
        raise SystemExit(f"CUDA unavailable on selected GPU {args.gpu}.")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU {args.gpu}: {torch.cuda.get_device_name(0)}, free {free / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_args = ["--model_path", str(projector), "--model_base", str(base),
                 "--conv_mode", "v1", "--dataset", args.dataset, "--task", args.task,
                 "--pretrained_embedding_type", "simteg", "--use_hop", "4",
                 "--sample_neighbor_size", "10", "--template", "HO", "--temperature", "0.2",
                 "--num_beams", "1", "--answers_file", str(output), "--cache_dir", str(ROOT / "models" / "cache")]
    if args.limit > 0:
        eval_args += ["--end", str(args.limit)]
    manifest = {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "base_id": BASE_ID, "base_revision": BASE_REV, "projector_id": PROJECTOR_ID,
                "projector_revision": PROJECTOR_REV, "seed": args.seed, "gpu": args.gpu,
                "torch": torch.__version__, "eval_args": eval_args,
                "note": "Original inference and scoring; seed fixed by wrapper. Model revisions describe the supplied download step; local weights must not be replaced."}
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Running original eval/eval_pretrain.py", flush=True)
    sys.argv = [str(ROOT / "eval" / "eval_pretrain.py")] + eval_args
    runpy.run_path(sys.argv[0], run_name="__main__")
    with source.open() as stream:
        expected = [json.loads(line)["id"] for line in stream if line.strip()]
    if args.limit > 0:
        expected = expected[:args.limit]
    with output.open() as stream:
        predictions = [json.loads(line) for line in stream if line.strip()]
    empty = sum(not row.get("text", "").strip() for row in predictions)
    ids_match = [row["question_id"] for row in predictions] == expected
    coverage = {"expected": len(expected), "actual": len(predictions), "empty_outputs": empty,
                "ids_match_in_original_order": ids_match}
    manifest["coverage"] = coverage
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("COVERAGE", json.dumps(coverage), flush=True)
    if empty or not ids_match:
        raise SystemExit("Coverage check failed. Inspect the log and predictions before interpreting metrics.")
    print("Running original eval/eval_res.py", flush=True)
    sys.argv = [str(ROOT / "eval" / "eval_res.py"), "--dataset", args.dataset,
                "--task", args.task, "--res_path", str(output)]
    runpy.run_path(sys.argv[0], run_name="__main__")
    print("Predictions:", output)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()

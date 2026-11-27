#!/usr/bin/env python3
"""Merge loop_eval.py logs (one per shard, or a single unsharded log) into one validated result JSON.

Log format and checks follow the original work by burtenshaw:
https://huggingface.co/buckets/burtenshaw/qwen-arc-loop/tree/eval
Shard logs are merged, then every summary is recomputed from the merged paired rows.
"""
import argparse, hashlib, json, socket, sys
from pathlib import Path

SOURCE = "https://huggingface.co/buckets/burtenshaw/qwen-arc-loop/tree/eval"
ARMS = ("baseline", "damped_euler", "heavy_ball")
METRICS = ["acc", "acc_norm"]
FULL_SIZES = {"arc_challenge": 1172, "arc_easy": 2376, "piqa": 1838, "hellaswag": 10042}
FRESH_N = 916
LOOP_OK = {"finite_changed_logits": True, "restored_logits_equal": True}


def parse(text):
    records, rows, seconds = {}, {}, []
    for line in text.splitlines():
        label, _, payload = line.partition(" ")
        if label == "PAIRED_ROWS":
            chunk = json.loads(payload)
            rows.setdefault(chunk["task"], []).extend(chunk["rows"])
        elif label == "ARM_RESULT":
            d = json.loads(payload)
            seconds.append((d["task"], d["arm"], d["seconds"]))
        elif label in {"RUN_CONFIG", "LOOP_CHECK", "RUN_RESULT"}:
            if label in records:
                raise ValueError(f"Duplicate {label} log record")
            records[label] = json.loads(payload)
    if "RUN_CONFIG" not in records or "LOOP_CHECK" not in records:
        raise ValueError("Log is incomplete: missing RUN_CONFIG or LOOP_CHECK")
    return records, rows, seconds


def collect(texts, preset, script_hash):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from loop_eval import summarize  # same summary code as the run itself

    configs, rows_by_task, arm_seconds, shards, run_results = [], {}, {}, [], []
    for text in texts:
        records, rows, seconds = parse(text)
        cfg = records["RUN_CONFIG"]
        if cfg["preset"] != preset or cfg["script_sha256"] != script_hash:
            raise ValueError("Run config differs from the submitted preset or script")
        if records["LOOP_CHECK"] != LOOP_OK:
            raise ValueError("The loop check did not pass")
        configs.append(cfg)
        shard = tuple(cfg.get("shard", [0, 1]))
        shards.append(shard)
        if shard == (0, 1) and "RUN_RESULT" in records:
            run_results = records["RUN_RESULT"]["tasks"]
        for task, task_rows in rows.items():
            rows_by_task.setdefault(task, []).extend(task_rows)
        for task, arm, sec in seconds:
            arm_seconds[(task, arm)] = arm_seconds.get((task, arm), 0.0) + sec

    base = configs[0]
    for cfg in configs[1:]:
        for key in ("task_sizes", "model_revision", "dataset_revisions", "layers", "arm_betas", "challenge_pilot_indices"):
            if cfg.get(key) != base.get(key):
                raise ValueError(f"Shard configs disagree on {key}")
    sizes = base["task_sizes"]
    if preset == "full":
        if sizes != FULL_SIZES:
            raise ValueError("Required task sizes are missing")
    elif set(sizes) != set(FULL_SIZES) or any(not 0 < n <= FULL_SIZES[t] for t, n in sizes.items()):
        raise ValueError("Smoke run must cover all four tasks")
    counts = {n for _, n in shards}
    if len(counts) != 1:
        raise ValueError("Shard logs use different shard counts")
    total = counts.pop()
    if sorted(k for k, _ in shards) != [k for k in range(total)] or any(n != total for _, n in shards):
        raise ValueError(f"Expected shards 0..{total - 1} exactly once")
    if set(rows_by_task) != set(sizes):
        raise ValueError("Tasks missing from the logs")

    summaries = {}
    for task, rows in rows_by_task.items():
        rows.sort(key=lambda r: r["index"])
        indices = [r["index"] for r in rows]
        if len(indices) != sizes[task] or len(set(indices)) != sizes[task]:
            raise ValueError(f"Incomplete or duplicate paired rows for {task}")
        if preset == "full" and set(indices) != set(range(sizes[task])):
            raise ValueError(f"Invalid test split indices for {task}")
        for arm in ARMS:
            for position in range(len(METRICS)):
                if any(row[arm][position] not in (0, 1) for row in rows):
                    raise ValueError(f"Non-binary scores for {task} {arm}")
        times = {arm: arm_seconds.get((task, arm), 0.0) for arm in ARMS}
        summaries[task] = summarize(rows, times)
        if total == 1 and task in run_results:
            for arm in ARMS:
                if run_results[task]["arms"][arm]["correct"] != summaries[task]["arms"][arm]["correct"]:
                    raise ValueError(f"Recomputed {task} {arm} differs from the run's own summary")

    if preset == "full":
        pilot = set(base["challenge_pilot_indices"])
        fresh = [row for row in rows_by_task["arc_challenge"] if row["index"] not in pilot]
        if len(fresh) != FRESH_N:
            raise ValueError(f"Expected {FRESH_N} fresh ARC-Challenge questions")
        summaries["arc_challenge_fresh"] = summarize(fresh)

    return {
        "source": SOURCE,
        "config": {**base, "shards": sorted(shards)},
        "loop_check": LOOP_OK,
        "metrics": METRICS,
        "tasks": summaries,
        "rows": rows_by_task,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--preset", required=True, choices=("smoke", "full"))
    parser.add_argument("--job", required=True, help="label for the output file, e.g. an OAR job id or 'full-merged'")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    script_hash = hashlib.sha256((root / "loop_eval.py").read_bytes()).hexdigest()
    texts = [Path(p).read_text(encoding="utf-8") for p in args.logs]
    result = collect(texts, args.preset, script_hash)
    out = Path(args.out_dir) / f"arc-{args.preset}-{args.job}.json"
    result = {"job": {"label": args.job, "host": socket.gethostname()}, **result}
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Turn one loop_eval.py log into a validated result JSON (same checks as the upstream run_arc.sh)."""
import argparse, hashlib, json, socket
from pathlib import Path

ARMS = ("baseline", "damped_euler", "heavy_ball")


def collect(logs, preset, script_hash):
    records = {}
    rows_by_task = {}
    for line in logs.splitlines():
        label, _, payload = line.partition(" ")
        if label == "PAIRED_ROWS":
            chunk = json.loads(payload)
            rows_by_task.setdefault(chunk["task"], []).extend(chunk["rows"])
        elif label in {"RUN_CONFIG", "LOOP_CHECK", "RUN_RESULT"}:
            if label in records:
                raise ValueError(f"Duplicate {label} log record")
            records[label] = json.loads(payload)

    config = records["RUN_CONFIG"]
    loop_check = records["LOOP_CHECK"]
    result = records["RUN_RESULT"]
    if config["preset"] != preset or config["script_sha256"] != script_hash:
        raise ValueError("Run config differs from the submitted preset or script")
    if loop_check != {"finite_changed_logits": True, "restored_logits_equal": True}:
        raise ValueError("The loop check did not pass")

    sizes = config["task_sizes"]
    full_sizes = {"arc_challenge": 1172, "arc_easy": 2376, "piqa": 1838, "hellaswag": 10042}
    if preset == "full":
        if sizes != full_sizes:
            raise ValueError("Required task sizes are missing")
        expected_summaries = {*full_sizes, "arc_challenge_fresh"}
    else:
        if set(sizes) != set(full_sizes) or any(not 0 < n <= full_sizes[t] for t, n in sizes.items()):
            raise ValueError("Smoke run must cover all four tasks")
        expected_summaries = set(full_sizes)
    if rows_by_task.keys() != sizes.keys() or result["tasks"].keys() != expected_summaries:
        raise ValueError("Tasks missing from the log")
    if preset == "full" and result["tasks"]["arc_challenge_fresh"]["n"] != 916:
        raise ValueError("Expected 916 fresh ARC-Challenge questions")

    metrics = result["metrics"]
    if metrics != ["acc", "acc_norm"]:
        raise ValueError("Expected raw and length-normalized accuracy")
    for task, rows in rows_by_task.items():
        summary = result["tasks"][task]
        count = sizes[task]
        indices = {row["index"] for row in rows}
        if len(rows) != count or len(indices) != count or summary["n"] != count:
            raise ValueError(f"Incomplete or duplicate paired rows for {task}")
        if preset == "full" and indices != set(range(count)):
            raise ValueError(f"Invalid test split indices for {task}")
        for arm in ARMS:
            for position, metric in enumerate(metrics):
                scores = [row[arm][position] for row in rows]
                if any(score not in (0, 1) for score in scores) or sum(scores) != summary["arms"][arm]["correct"][metric]:
                    raise ValueError(f"Paired rows disagree with {task} {arm} {metric}")
    return {"config": config, "loop_check": loop_check, **result, "rows": rows_by_task}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--preset", required=True, choices=("smoke", "full"))
    parser.add_argument("--job", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    script_hash = hashlib.sha256((root / "loop_eval.py").read_bytes()).hexdigest()
    result = collect(Path(args.log).read_text(encoding="utf-8"), args.preset, script_hash)
    out = Path(args.out_dir) / f"arc-{args.preset}-{args.job}.json"
    out.write_text(json.dumps({"job": {"oar_job": args.job, "host": socket.gethostname()}, **result}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

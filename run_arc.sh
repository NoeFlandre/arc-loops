#!/bin/sh
exec python3 - "$(dirname "$0")" "$@" <<'PY'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit


IMAGE = "pytorch/pytorch@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee"
TIMEOUTS = {"l4x1": "15m", "a10g-small": "12m"}
ARMS = ("baseline", "damped_euler", "heavy_ball")


def hf(*args):
    return subprocess.run(["hf", "jobs", *args], check=True, capture_output=True, text=True).stdout


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
    expected_sizes = {"arc_challenge": 256}
    expected_summaries = set(expected_sizes)
    if preset == "full":
        expected_sizes = {"arc_challenge": 1172, "arc_easy": 2376}
        expected_summaries = {*expected_sizes, "arc_challenge_fresh"}
    if (sizes != expected_sizes or rows_by_task.keys() != sizes.keys()
            or result["tasks"].keys() != expected_summaries):
        raise ValueError("Required ARC tasks or sample sizes are missing")
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
    root = Path(sys.argv[1]).resolve()
    parser = argparse.ArgumentParser(prog="run_arc.sh", description="Run the ARC loop evaluation on Hugging Face Jobs.")
    parser.add_argument("preset", choices=("full", "smoke"), nargs="?", default="full")
    parser.add_argument("--flavor", choices=TIMEOUTS, default="l4x1")
    args = parser.parse_args(sys.argv[2:])
    timeout = TIMEOUTS[args.flavor]

    source = (root / "loop_eval.py").read_bytes()
    script_hash = hashlib.sha256(source).hexdigest()
    command = [
        "/bin/sh", "-eu", "-c",
        '''printf '%s' "$1" > /tmp/loop_eval.py
shift
python -m pip install --quiet --disable-pip-version-check 'transformers==4.57.6' 'datasets==4.4.1' 'accelerate==1.12.0' 'lm-eval==0.4.11'
exec python -u /tmp/loop_eval.py "$@"''',
        "qwen-loop-eval", source.decode("utf-8"), "--preset", args.preset,
    ]
    launch = json.loads(hf(
        "run", "--detach", "--json", "--flavor", args.flavor, "--timeout", timeout, "--attempts", "1",
        "--name", f"qwen3-0p6b-base-loop-arc-{args.preset}",
        "--env", "HF_HUB_DISABLE_TELEMETRY=1", "--env", "HF_HUB_DISABLE_PROGRESS_BARS=1",
        "--env", "TOKENIZERS_PARALLELISM=false", "--env", "OMP_NUM_THREADS=4",
        "--", IMAGE, *command,
    ))
    job_id, url = launch["id"], launch["url"]
    job_ref = urlsplit(url).path.split("/jobs/", 1)[1]
    print(f"Job {job_id}\n{url}", flush=True)

    waited = subprocess.run(["hf", "jobs", "wait", job_ref], check=False)
    (job,) = json.loads(hf("inspect", "--json", job_ref))
    if waited.returncode or job["status"]["stage"] != "COMPLETED":
        subprocess.run(["hf", "jobs", "logs", "--format", "human", "--tail", "40", job_ref], check=False)
        raise ValueError(f"Job {job_id} finished with status {job['status']['stage']}")
    if (job["id"] != job_id or job["docker_image"] != IMAGE or job["flavor"] != args.flavor
            or job["command"] != command):
        raise ValueError("Job metadata differs from the requested launch")

    result = collect(hf("logs", "--format", "human", job_ref), args.preset, script_hash)
    metadata = {key: job.get(key) for key in (
        "id", "url", "status", "flavor", "docker_image", "created_at", "started_at", "finished_at", "durations",
    )}
    metadata.update(timeout=timeout, inline_source_matches=True)
    destination = root / f"arc-{args.preset}-{job_id}.json"
    destination.write_text(json.dumps({"job": metadata, **result}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {destination}")


try:
    main()
except subprocess.CalledProcessError as error:
    if error.stderr:
        print(error.stderr.strip(), file=sys.stderr)
    sys.exit(f"hf jobs {error.cmd[2]} exited with code {error.returncode}")
except (OSError, ValueError, KeyError) as error:
    sys.exit(f"ARC evaluation: {error}")
PY

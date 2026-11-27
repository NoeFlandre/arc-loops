# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.6.0",
#     "transformers==4.57.6",
#     "datasets==4.4.1",
#     "accelerate==1.12.0",
#     "lm-eval==0.4.11",
# ]
# ///
"""Evaluate cache-free layer looping on pinned ARC test sets."""

import argparse
import hashlib
import json
import random
import time
from contextlib import contextmanager, nullcontext
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
# ARC: allenai/ai2_arc. PIQA: baber/piqa (parquet copy of ybisk/piqa). HellaSwag: Rowan/hellaswag.
# PIQA and HellaSwag hide test labels, so their validation split is the evaluated test set.
DATASET_REVISIONS = {
    "arc_challenge": "210d026faf9955653af8916fad021475a3f00453",
    "arc_easy": "210d026faf9955653af8916fad021475a3f00453",
    "piqa": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
    "hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
}
CHALLENGE = "arc_challenge"
TEST_SIZES = {CHALLENGE: 1172, "arc_easy": 2376, "piqa": 1838, "hellaswag": 10042}
PILOT_SEED = 42
PILOT_SIZE = 256
LAYERS = (12, 13, 14, 15)
ARMS = {"baseline": None, "damped_euler": 0.0, "heavy_ball": 0.5}
METRICS = ("acc", "acc_norm")


def emit(label, data):
    print(f"{label} {json.dumps(data, separators=(',', ':'), allow_nan=False)}", flush=True)


def doc_key(doc):
    return doc.get("id", doc.get("ind"))


def two_steps(forward, beta):
    def looped(hidden_states, **kwargs):
        if kwargs.get("past_key_values") is not None:
            raise ValueError("Layer looping requires a cache-free forward.")
        midpoint = hidden_states + 0.5 * (forward(hidden_states, **kwargs) - hidden_states)
        refined = midpoint + 0.5 * (forward(midpoint, **kwargs) - midpoint)
        return refined + beta * (midpoint - hidden_states)

    return looped


@contextmanager
def loop_middle(model, beta):
    layers = [model.model.layers[index] for index in LAYERS]
    original_forwards = [layer.forward for layer in layers]
    try:
        for layer, forward in zip(layers, original_forwards, strict=True):
            layer.forward = two_steps(forward, beta)
        yield
    finally:
        for layer, forward in zip(layers, original_forwards, strict=True):
            layer.forward = forward


@torch.inference_mode()
def check_loop(model, tokenizer):
    prompt = tokenizer("The capital of France is", return_tensors="pt").to(model.device)

    def logits():
        return model(**prompt, use_cache=False, logits_to_keep=1).logits

    baseline = logits()
    with loop_middle(model, beta=0.5):
        looped = logits()
    if not torch.isfinite(looped).all() or torch.equal(looped, baseline):
        raise RuntimeError("Loop check expected finite and changed logits.")
    if not torch.equal(logits(), baseline):
        raise RuntimeError("Restoring the layer forwards changed baseline logits.")


def evaluate_arm(model, tokenizer, task, manager, indices, beta, batch_size):
    task_name = task.config.task
    evaluator_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    start = time.perf_counter()
    with nullcontext() if beta is None else loop_middle(model, beta):
        result = simple_evaluate(
            model=evaluator_model,
            tasks=[task],
            task_manager=manager,
            num_fewshot=0,
            samples={task_name: indices},
            log_samples=True,
            bootstrap_iters=0,
        )
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
    seconds = time.perf_counter() - start
    samples = sorted(result["samples"][task_name], key=lambda sample: sample["doc_id"])
    observed = [(sample["doc_id"], doc_key(sample["doc"])) for sample in samples]
    expected = [(index, doc_key(task.eval_docs[index])) for index in indices]
    if observed != expected:
        raise RuntimeError(f"The evaluated rows differed from the {task_name} test sample.")
    for metric in METRICS:
        values = [sample[metric] for sample in samples]
        if not all(value in (0, 1) for value in values):
            raise RuntimeError(f"Expected binary {metric} scores for {task_name}.")
        if not np.isclose(np.mean(values), result["results"][task_name][f"{metric},none"]):
            raise RuntimeError(f"The per-row and aggregate {task_name} {metric} scores differ.")
    return samples, seconds


def paired_stats(before, after):
    delta = np.asarray(after, dtype=int) - np.asarray(before, dtype=int)
    losses = int((delta == -1).sum())
    gains = int((delta == 1).sum())
    n = len(delta)
    # Resample the three paired outcomes through their counts.
    draws = np.random.default_rng(0).multinomial(n, np.array([losses, n - losses - gains, gains]) / n, size=20_000)
    draw_deltas = (draws[:, 2] - draws[:, 0]) * (100 / n)
    return {
        "wrong_to_right": gains,
        "right_to_wrong": losses,
        "delta_pp": float(100 * delta.mean()),
        "ci95_pp": np.quantile(draw_deltas, [0.025, 0.975]).tolist(),
    }


def score_task(model, tokenizer, task, manager, indices, batch_size):
    task_name = task.config.task
    rows_by_arm = {}
    times = {}
    for arm, beta in ARMS.items():
        emit("START_ARM", {"task": task_name, "arm": arm})
        rows_by_arm[arm], times[arm] = evaluate_arm(
            model, tokenizer, task, manager, indices, beta, batch_size
        )
        emit("ARM_RESULT", {"task": task_name, "arm": arm, "seconds": times[arm]})

    rows = []
    for matched in zip(*rows_by_arm.values(), strict=True):
        first = matched[0]
        for field in ("doc_hash", "prompt_hash", "target_hash"):
            if any(row[field] != first[field] for row in matched):
                raise RuntimeError(f"Arms used different {field} values for {task_name} row {first['doc_id']}.")
        rows.append({
            "index": first["doc_id"],
            "id": doc_key(first["doc"]),
            **{arm: [int(row[metric]) for metric in METRICS] for arm, row in zip(ARMS, matched, strict=True)},
        })
    for start in range(0, len(rows), 16):
        emit("PAIRED_ROWS", {"task": task_name, "rows": rows[start : start + 16]})
    return rows, times


def summarize(rows, times=None):
    arms = {}
    for arm in ARMS:
        counts = {metric: sum(row[arm][i] for row in rows) for i, metric in enumerate(METRICS)}
        arms[arm] = {"correct": counts}
        if times is not None:
            arms[arm]["seconds"] = times[arm]

    paired = {}
    for before, after in combinations(ARMS, 2):
        paired[f"{after}_vs_{before}"] = {
            metric: paired_stats([row[before][i] for row in rows], [row[after][i] for row in rows])
            for i, metric in enumerate(METRICS)
        }
    return {"n": len(rows), "arms": arms, "paired": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--samples", type=int, default=PILOT_SIZE, help="Questions in the smoke preset.")
    parser.add_argument("--seed", type=int, default=PILOT_SEED, help="Sampling seed for the smoke preset.")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1:
        parser.error("Sample count and batch size must be positive.")
    if args.samples > min(TEST_SIZES.values()):
        parser.error(f"Smoke samples must not exceed the smallest test set ({min(TEST_SIZES.values())}).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Select a GPU with native BF16 support.")
    manager = TaskManager()
    full = args.preset == "full"
    tasks = {}
    indices = {}
    for name in TEST_SIZES:
        task = manager.load_config({"task": name, "dataset_kwargs": {"revision": DATASET_REVISIONS[name]}})[name]
        if len(task.eval_docs) != TEST_SIZES[name]:
            raise RuntimeError(f"Unexpected test split size for {name}.")
        tasks[name] = task
        docs = range(len(task.eval_docs))
        indices[name] = list(docs) if full else sorted(random.Random(args.seed).sample(docs, args.samples))
    pilot_indices = sorted(random.Random(PILOT_SEED).sample(range(TEST_SIZES[CHALLENGE]), PILOT_SIZE))
    emit("RUN_CONFIG", {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dataset_revisions": DATASET_REVISIONS,
        "preset": args.preset,
        "task_sizes": {name: len(items) for name, items in indices.items()},
        "split": "test",
        "fewshot": 0,
        "sample_seed": None if full else args.seed,
        "challenge_pilot_indices": pilot_indices if full else None,
        "batch_size": args.batch_size,
        "layers": LAYERS,
        "alpha": 0.5,
        "arm_betas": ARMS,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": str(dtype),
        "packages": {name: version(name) for name in ("torch", "transformers", "datasets", "lm-eval")},
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION, dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval().requires_grad_(False)
    model.config.use_cache = False
    check_loop(model, tokenizer)
    emit("LOOP_CHECK", {"finite_changed_logits": True, "restored_logits_equal": True})

    summaries = {}
    pilot = set(pilot_indices)
    for name, task in tasks.items():
        rows, times = score_task(model, tokenizer, task, manager, indices[name], args.batch_size)
        summaries[name] = summarize(rows, times)
        emit("TASK_RESULT", {"task": name, **summaries[name]})
        if full and name == CHALLENGE:
            fresh = [row for row in rows if row["index"] not in pilot]
            summaries["arc_challenge_fresh"] = summarize(fresh)
    emit("RUN_RESULT", {"metrics": METRICS, "tasks": summaries})


if __name__ == "__main__":
    main()

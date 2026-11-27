# arc-loops

Reproduction and extension of the ARC layer-looping evaluation on Qwen3-0.6B-Base, run on Grid'5000 (abaca).

## Source and credit

The original experiment is by **burtenshaw**. The evaluation scripts (`loop_eval.py`, `run_arc.sh`), this README, and the original results (`results.json`) come from:

https://huggingface.co/buckets/burtenshaw/qwen-arc-loop/tree/eval

The original README is kept unchanged below. The method and original results are burtenshaw's work.

## What this repo adds

- `oar/`: scripts to run the eval on Grid'5000 abaca through OAR (environment build, GPU/CPU run).
- PIQA (`baber/piqa`, revision `142f6d7`, validation split) and HellaSwag (`Rowan/hellaswag`, revision `218ec52`, validation split). These were **not** in the original run.
- `collect.py`: checks each run log and writes the result JSON.
- Results and data are stored in the HF bucket [NoeFlandre/arc-loops](https://huggingface.co/buckets/NoeFlandre/arc-loops), with the original upstream files under `upstream-eval/`.

## Results

See [RESULTS.md](RESULTS.md): ARC-Easy, ARC-Challenge, PIQA and HellaSwag on one H100 (bf16), compared with the original run. Short version: the ARC-Easy gain from damped Euler reproduces; the fresh ARC-Challenge gain does not; PIQA and HellaSwag show no clear change.

## How it was run

- Environment and dataset cache per site: `oar/env_build.sh` (CPU job).
- GPU or CPU run: `oar/run_eval.sh <smoke|full> [k/N]` through OAR `-q abaca`.
- Result JSON from logs: `collect.py` (merges shard logs, recomputes summaries).

---

# Original README (from https://huggingface.co/buckets/burtenshaw/qwen-arc-loop/tree/eval)

# Four extra layer calls on Qwen3-0.6B-Base

We tried a small inference-time change on a frozen language model. On ARC-Easy, running four middle layers twice with damped updates lifted length-normalized accuracy from **58.00% to 61.15%**. The two ARC test sets contain 3,548 questions in total, and the bundle includes each question's outcome under all three settings.

## How the loop works

A transformer layer takes a hidden state `x` and returns a new one, `L(x)`. Think of the difference `L(x) - x` as the step the layer wants to take. Damped Euler uses half of that step, applies the same layer to the resulting state, and uses half of the newly computed step.

```python
midpoint = x + 0.5 * (layer(x) - x)
out = midpoint + 0.5 * (layer(midpoint) - midpoint)
```

We do this independently at zero-based layers 12, 13, 14, and 15. The model runs both calls to layer 12 before it reaches layer 13. The four extra calls bring a forward pass from 28 decoder-layer calls to 32. Heavy-ball adds momentum from the first half-step to the second, with the same call count as damped Euler.

This is an ARC-only test of the layer-looping idea from *Training-Free Looped Transformers* by Chen et al. The paper reports a heavy-ball result on Qwen3-0.6B-Base across a 16-task suite. Figure 1 illustrates the loop modes, and Figure 2 compares naive looping and an anchored Runge-Kutta variant on a toy residual MLP. [Paper](https://arxiv.org/abs/2605.23872) · [Figure 1](https://arxiv.org/html/2605.23872v1#S1.F1) · [Figure 2](https://arxiv.org/html/2605.23872v1#S2.F2) · [0.6B results](https://arxiv.org/html/2605.23872v1#A12.T18)

## What we measured

Baseline, damped Euler, and heavy-ball scored identical zero-shot prompts on the official ARC-Easy and ARC-Challenge test splits. We used one pinned Qwen3-0.6B-Base checkpoint in BF16, cache-free likelihood scoring, and the ARC tasks in `lm-eval` 0.4.11. Length-normalized scoring ranks choices by their log-likelihood divided by answer text length. The 916-question row excludes the questions in the initial pilot and uses predictions from the full ARC-Challenge run. [ARC dataset revision](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453) · [Model revision](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd)

Each cell shows raw accuracy / length-normalized accuracy in percent.

| Test set | Baseline | Damped Euler | Heavy-ball |
|---|---:|---:|---:|
| ARC-Easy, 2,376 | 65.49 / 58.00 | 67.26 / 61.15 | 65.57 / 58.12 |
| ARC-Challenge, 1,172 | 33.28 / 38.40 | 34.90 / 39.16 | 34.30 / 37.97 |
| ARC-Challenge outside the pilot, 916 | 34.06 / 38.65 | 35.70 / 39.85 | 35.15 / 38.43 |

On ARC-Easy, damped Euler gained **3.16 percentage points** in length-normalized accuracy over baseline, with a **paired 95% bootstrap interval of +2.23 to +4.08**. It changed 102 incorrect answers to correct and 27 correct answers to incorrect under that scoring rule. On the 916 Challenge questions outside the pilot, the corresponding delta was **+1.20 points** with an interval of **−0.11 to +2.51**. Heavy-ball's length-normalized changes were close to zero. The paired results make it possible to audit these differences question by question in `results.json`.

## Run it on Hugging Face Jobs

The public bucket contains the two scripts and the results from the original run. The launcher uses Python 3 and the `hf` CLI (tested with 2.1.1). Your Hugging Face account needs Jobs compute credit to launch a new run. [Browse the files](https://huggingface.co/buckets/burtenshaw/qwen-arc-loop/tree/eval) · [Hugging Face Jobs documentation](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)

```sh
hf auth login
hf sync hf://buckets/burtenshaw/qwen-arc-loop/eval ./arc
bash ./arc/run_arc.sh full --flavor a10g-small
```

This uses the same pinned container as the evaluated run. The launcher submits one BF16 GPU job with a 12-minute timeout, prints the URL, then saves `arc-full-<job-id>.json` beside the scripts. For a smaller check, use `bash ./arc/run_arc.sh smoke --flavor a10g-small`. To inspect the original run, open `results.json`. Each question has a pair of 0/1 correctness flags per method, ordered as raw and length-normalized scoring.

# arc-loops

Looping a few middle layers of a small language model (Qwen3-0.6B-Base), tested on four multiple-choice benchmarks.

**Original work:** [burtenshaw](https://huggingface.co/buckets/burtenshaw/qwen-arc-loop/tree/eval). This repo reruns it on Grid'5000 and adds PIQA and HellaSwag.

## Idea

Four middle layers are run twice with a damped update. No training. This adds 4 layer calls per forward pass (28 → 32).

Two modes:
- **Layer:** each layer is looped on its own.
- **Block:** layers 12–15 are looped together as one unit.

## Results

Accuracy in %, length-normalized (acc_norm). Last column: damped Euler minus baseline, with 95% interval. One H100, bf16.

### Layer mode

| Benchmark (questions) | Baseline | Damped Euler | Heavy-ball | Damped − baseline |
|---|---|---|---|---|
| ARC-Easy (2,376) | 57.9 | 61.2 | 58.3 | +3.3 [+2.4, +4.3] |
| ARC-Challenge (1,172) | 38.6 | 38.8 | 37.9 | +0.3 [−1.0, +1.6] |
| PIQA, validation (1,838) | 69.8 | 69.9 | 69.9 | +0.1 [−0.5, +0.7] |
| HellaSwag, validation (10,042) | 53.8 | 53.6 | 54.3 | −0.2 [−0.5, +0.1] |

![Layer mode: change vs. baseline](figures/layer-delta.svg)

![Layer mode: accuracy](figures/layer-accuracy.svg)

### Block mode

| Benchmark (questions) | Baseline | Damped Euler | Heavy-ball | Damped − baseline |
|---|---|---|---|---|
| ARC-Easy (2,376) | 57.9 | 60.9 | 59.7 | +3.1 [+2.2, +4.0] |
| ARC-Challenge (1,172) | 38.6 | 37.9 | 38.0 | −0.7 [−1.9, +0.5] |
| PIQA, validation (1,838) | 69.8 | 69.6 | 69.6 | −0.2 [−0.9, +0.4] |
| HellaSwag, validation (10,042) | 53.8 | 53.1 | 53.7 | −0.7 [−1.0, −0.4] |

![Block mode: change vs. baseline](figures/block-delta.svg)

![Block mode: accuracy](figures/block-accuracy.svg)

### Takeaways

- Layer mode: the ARC-Easy gain reproduces (+3.3 points).
- Block mode: the ARC-Easy gain is smaller (+3.1), and HellaSwag drops slightly (−0.7).
- No other clear gains in either mode.

Details: [RESULTS.md](RESULTS.md). Raw data: [HF bucket](https://huggingface.co/buckets/NoeFlandre/arc-loops).

## Run it

On Grid'5000 (OAR, queue `abaca`):

```bash
# once per site, CPU
oarsub -q abaca -p gpu_count=0 "bash oar/env_build.sh"

# layer mode, one GPU
oarsub -q abaca -l gpu=1,walltime=2:00 "bash oar/run_eval.sh full"

# block mode, one GPU
oarsub -q abaca -l gpu=1,walltime=2:00 "LOOP_MODE=block bash oar/run_eval.sh full"
```

## Files

- `loop_eval.py`: evaluation script, adapted from the original work. Adds `--loop-mode {layer,block}`.
- `collect.py`: turns logs into a result JSON.
- `figures.py`: makes the SVG figures in `figures/`.
- `oar/`: Grid'5000 scripts.
- `RESULTS.md`: full numbers and notes.
- `env/requirements.txt`: pinned packages.

## Credit

Method, original code and original results: **burtenshaw**. Source: https://huggingface.co/buckets/burtenshaw/qwen-arc-loop/tree/eval

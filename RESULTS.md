# Results

Layer and block runs of the ARC, PIQA and HellaSwag looping eval on Qwen3-0.6B-Base.

Method and original results: **burtenshaw**, https://huggingface.co/buckets/burtenshaw/qwen-arc-loop/tree/eval

Setup: one NVIDIA H100 NVL, bf16, batch 8. Result JSON and logs are in the HF bucket `NoeFlandre/arc-loops` under `runs/`. Jobs: layer 3427685, block 3427806.

Accuracy in %, as `raw acc / length-normalized acc` (acc_norm). The last column is damped Euler minus baseline on acc_norm, with a paired 95% bootstrap interval.

## Layer mode

| Test set (n) | Baseline | Damped Euler | Heavy-ball | Damped − baseline (acc_norm) | Original run (damped − baseline) |
|---|---|---|---|---|---|
| ARC-Easy (2376) | 65.57 / 57.87 | 67.09 / 61.20 | 65.61 / 58.29 | +3.32 [+2.40, +4.25] | +3.16 [+2.23, +4.08] |
| ARC-Challenge (1172) | 33.70 / 38.57 | 34.98 / 38.82 | 34.47 / 37.88 | +0.26 [−1.02, +1.62] | +0.76 (38.40 → 39.16) |
| PIQA, validation (1838) | 69.80 / 69.80 | 69.91 / 69.91 | 70.35 / 69.86 | +0.11 [−0.49, +0.71] | not in original run |
| HellaSwag, validation (10042) | 40.96 / 53.79 | 41.21 / 53.62 | 41.55 / 54.27 | −0.17 [−0.45, +0.10] | not in original run |

Original run, for reference (raw / acc_norm): ARC-Easy 65.49 / 58.00 → 67.26 / 61.15 → 65.57 / 58.12 (baseline → damped → heavy-ball); ARC-Challenge 33.28 / 38.40 → 34.90 / 39.16 → 34.30 / 37.97.

## Block mode

Same update, applied to layers 12–15 as one unit.

| Test set (n) | Baseline | Damped Euler | Heavy-ball | Damped − baseline (acc_norm) |
|---|---|---|---|---|
| ARC-Easy (2376) | 65.57 / 57.87 | 67.59 / 60.94 | 66.46 / 59.68 | +3.07 [+2.19, +3.96] |
| ARC-Challenge (1172) | 33.70 / 38.57 | 33.45 / 37.88 | 34.04 / 37.97 | −0.68 [−1.88, +0.51] |
| PIQA, validation (1838) | 69.80 / 69.80 | 69.91 / 69.59 | 70.46 / 69.64 | −0.22 [−0.87, +0.44] |
| HellaSwag, validation (10042) | 40.96 / 53.79 | 40.47 / 53.08 | 41.25 / 53.74 | −0.72 [−1.01, −0.44] |

## Notes

- **Reproduction check.** Per-question agreement with the original `results.json` (baseline / damped / heavy-ball), layer mode: ARC-Easy 98.1% / 97.5% / 97.9%; ARC-Challenge 97.3% / 97.4% / 97.5%. The small gaps come from the software and hardware stack (torch 2.6.0 on H100 vs. the original Docker image).
- **ARC-Easy gain reproduces** in layer mode (about +3.3 points) and is slightly smaller in block mode (+3.1).
- **HellaSwag** drops slightly in block mode (−0.7, interval below zero).
- **PIQA and HellaSwag** use validation splits because their test labels are hidden.

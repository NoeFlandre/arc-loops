# Results: ARC layer-looping on Qwen3-0.6B-Base

Method and original results: **burtenshaw**, https://huggingface.co/buckets/burtenshaw/qwen-arc-loop/tree/eval

This reproduction: one NVIDIA H100 NVL (Grid'5000 abaca, Sophia `musa`), bf16, batch 8. Full result JSON and run log are in the HF bucket `NoeFlandre/arc-loops` under `runs/` (job 3427685).

Cells are accuracy in % as `raw acc / length-normalized acc` (acc_norm). The last column is damped Euler minus baseline on acc_norm, with a paired 95% bootstrap interval.

| Test set (n) | Baseline | Damped Euler | Heavy-ball | Damped − baseline (acc_norm) | Original run (damped − baseline) |
|---|---|---|---|---|---|
| ARC-Easy (2376) | 65.57 / 57.87 | 67.09 / 61.20 | 65.61 / 58.29 | +3.32 [+2.40, +4.25] | +3.16 [+2.23, +4.08] |
| ARC-Challenge (1172) | 33.70 / 38.57 | 34.98 / 38.82 | 34.47 / 37.88 | +0.26 [−1.02, +1.62] | +0.76 (38.40 → 39.16) |
| PIQA, validation (1838) | 69.80 / 69.80 | 69.91 / 69.91 | 70.35 / 69.86 | +0.11 [−0.49, +0.71] | not in original run |
| HellaSwag, validation (10042) | 40.96 / 53.79 | 41.21 / 53.62 | 41.55 / 54.27 | −0.17 [−0.45, +0.10] | not in original run |

Original run, for reference (raw / acc_norm, baseline → damped → heavy-ball): ARC-Easy 65.49/58.00 → 67.26/61.15 → 65.57/58.12; ARC-Challenge 33.28/38.40 → 34.90/39.16 → 34.30/37.97.

## Notes

- **Reproduction check.** Per-question agreement with the original `results.json` (baseline / damped / heavy-ball): ARC-Easy 98.1% / 97.5% / 97.9%; ARC-Challenge 97.3% / 97.4% / 97.5%. The small gaps come from the software and hardware stack (torch 2.6.0 on H100 vs. the original Docker image).
- **ARC-Easy gain reproduces:** about +3.3 points on acc_norm, with an interval close to the original.
- **PIQA and HellaSwag** use validation splits because their test labels are hidden. Looping makes no clear difference on either.
- A CPU float32 backup run was planned but cancelled after the GPU run completed; it is not part of these results.

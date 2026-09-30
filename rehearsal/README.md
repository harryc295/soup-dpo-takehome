# Rehearsal on an RTX 4060 Ti (8 GB, bf16)

The full pipeline ran here first to debug it before the T4 run. These are supporting
evidence only. Everything in REPORT.md comes from `../results/` (T4) unless marked
*rehearsal*.

- `train_4060ti.log`, `verify.*`, `silent_checks.json`, `memory_budget.json`: the same
  config as `soup.yaml` (`local_full.yaml`). It reached the same conclusions: held-out 84%,
  random control 43%, 94% vs 59% by length.
- `memory_calibration/`: two 4-step runs on 32 pairs (`batch1.yaml`, `batch2.yaml`),
  identical except for batch size. The peaks (3.22 GB and 6.01 GB) give the ~18 bytes per
  logit element used in the corrected memory estimate. The 32-pair file was the first 32
  rows of `data/train.jsonl`.

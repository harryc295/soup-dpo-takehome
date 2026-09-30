# How I used AI tools

I used Claude Code heavily. It did most of the code reading and wrote the first drafts of
the scripts and patches. I set the scope, reviewed what it produced, and checked the
numbers in the report against the raw logs before submitting.

**Where it did the work**
- Reading Soup 0.75.1's source (layer streaming, the DPO trainer path, `data lint`,
  `soup ship`) and citing the lines that decide behaviour.
- Drafting `verify_training.py`, `silent_checks.py`, `memory_budget.py`, the notebook,
  and the Soup patches, and driving the Colab run.

**Where its first answer was wrong, and how that was caught**
- The first memory budget treated the DPO reference pass as transient. A rehearsal run
  came in about 23% over it. Reading TRL's `_compute_loss` showed the ref forward runs
  while the policy logits are still alive. The estimate got a correction term, and two
  measured points (batch 1 and 2) put the real cost at about 18 bytes per logit element.
- The first task-eval file for `soup ship` had "expected answers" like "To complete this
  task, I will:". Those were filtered out, and the report calls the leg-1 metric weak
  rather than claiming it measures DPO quality.
- It planned three `soup ship` runs on the full default suite, about an hour each. Leg 2
  was cut to `mini_arithmetic`, and the report says so.
- It assumed Colab's Python would satisfy `soup-cli`'s `<3.13` pin. The first T4 attempt
  failed at install. The notebook now uses the same `--ignore-requires-python` fallback
  as Soup's own proof notebook.
- Several comments in Soup's code are out of date (DPO "adapters disabled /
  `null_ref_context`", stream_layers "batch 1, no accumulation"). Claims were checked
  against the code that runs, not the comments.

**What I checked myself**
- The headline numbers against the raw files: memory peaks in `soup.db` and
  `results/nvidia_smi.csv`, the skipped fp16 steps in `results/trainer_state.json`, and the
  held-out accuracy and length split in `results/verify.json`.
- Both runs, rehearsal (RTX 4060 Ti) and T4, telling the same story.
- The Soup patches: read the diffs and the test results, including the check that the new
  tests fail without the fix.

"""Inputs for the `soup ship` experiment (Part 3).

1. task_eval.jsonl: held-out prompts whose chosen answer opens with a short line
   (a letter, a name, a number). expected = that line, scoring = contains. The DPO
   data has no gold labels, so this is the closest honest task metric available.
2. Two fake adapters that must NOT ship:
     ctrl-zero/    trained adapter with every lora_B zeroed  (a run that trained nothing)
     ctrl-random/  lora_B replaced by random noise with the trained per-tensor norms
                   (a run that changed the model without learning the preference)
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="out-dpo")
ap.add_argument("--heldout", default="data/heldout.jsonl")
args = ap.parse_args()

tasks = []
for line in open(args.heldout, encoding="utf-8"):
    row = json.loads(line)
    first = row["chosen"][0]["content"].strip().split("\n")[0].strip()
    if 0 < len(first) <= 60 and not first.endswith(":"):
        tasks.append({"prompt": row["prompt"][-1]["content"], "expected": first, "scoring": "contains"})
with open("task_eval.jsonl", "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(f"task_eval.jsonl: {len(tasks)} tasks")

sd = load_file(str(Path(args.adapter) / "adapter_model.safetensors"))
g = torch.Generator().manual_seed(0)
for name in ("zero", "random"):
    out = Path(f"ctrl-{name}")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    shutil.copy(Path(args.adapter) / "adapter_config.json", out)
    new = {}
    for k, v in sd.items():
        if "lora_B" in k:
            if name == "zero":
                v = torch.zeros_like(v)
            else:
                r = torch.randn(v.shape, generator=g)
                v = (r * (v.float().norm() / r.norm())).to(v.dtype)
        new[k] = v.contiguous()
    save_file(new, str(out / "adapter_model.safetensors"))
    print(f"{out}/ written")

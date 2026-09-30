"""Generates run_colab.ipynb. Edit this file, not the notebook JSON."""

import json

REPO = "https://github.com/harryc295/soup-dpo-takehome"

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip()})


md(f"""
# Soup DPO + layer streaming on a T4: ship or not?

Runtime -> Change runtime type -> **T4 GPU**, then **Runtime -> Run all**. About 45-75 min.
Every command's output is written with timestamps to `results/logs/`, `nvidia-smi` is
sampled every 500 ms into `results/nvidia_smi.csv`, and the last cell zips `results/`
for download.

Nothing here is edited after the fact: if a step fails, its log and exit code stay in
`results/` and the notebook carries on so the failure is recorded, not hidden.
""")

code(f"""
import os, subprocess
if not os.path.exists("soup-dpo-takehome"):
    subprocess.run(["git", "clone", "{REPO}"], check=True)
os.chdir("soup-dpo-takehome")
os.makedirs("results/logs", exist_ok=True)

# nvidia-smi sampler for the whole session (raw output, untouched)
smi = subprocess.Popen(
    "nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,"
    "utilization.gpu,temperature.gpu,power.draw,clocks.sm --format=csv -lms 500 "
    "> results/nvidia_smi.csv",
    shell=True,
)
subprocess.run("nvidia-smi > results/nvidia_smi_start.txt", shell=True)
print(open("results/nvidia_smi_start.txt").read())
""")

code("""
import datetime, subprocess, sys

def run(cmd, name, env=None):
    \"\"\"Run a shell command, stream output with UTC timestamps to screen and to
    results/logs/<name>.log. Never raises: the exit code is logged instead.\"\"\"
    path = f"results/logs/{name}.log"
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", "SOUP_DB_PATH": os.path.abspath("soup.db"),
                "HF_HUB_DISABLE_PROGRESS_BARS": "1", **(env or {})}
    with open(path, "a", encoding="utf-8") as log:
        def emit(line):
            stamped = f"{datetime.datetime.utcnow().isoformat(timespec='milliseconds')}Z {line}"
            print(stamped, flush=True)
            log.write(stamped + "\\n")
            log.flush()
        emit(f"$ {cmd}")
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, env=full_env, bufsize=1)
        for line in p.stdout:
            emit(line.rstrip("\\n"))
        p.wait()
        emit(f"[exit code {p.returncode}]")
    return p.returncode
""")

md("## 0. Install (pinned) and confirm the card")

code("""
run("pip uninstall -q -y torchao", "00_install")  # peft raises on Colab's old torchao
run('pip install -q "soup-cli[train]==0.75.1"', "00_install")
run("python -c \\"import torch,transformers,trl,peft,soup_cli;"
    "print('torch',torch.__version__,'transformers',transformers.__version__,"
    "'trl',trl.__version__,'peft',peft.__version__,'soup',soup_cli.__version__)\\"", "00_versions")
run("python -c \\"import torch;from soup_cli.utils.gpu import bf16_fp16_flags;"
    "print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0));"
    "print('bf16 in hardware:', torch.cuda.is_bf16_supported(including_emulation=False));"
    "print('soup precision (bf16, fp16):', bf16_fp16_flags('cuda'))\\"", "00_versions")
run("soup doctor", "01_doctor")
""")

md("## 1. Data: build 500 pairs, then Soup's own data checks")

code("""
run("python prepare_data.py", "02_prepare_data")
# data doctor refuses DPO data by design; lint is Soup's preference-data checker
run("soup data doctor data/train.jsonl --model Qwen/Qwen2.5-0.5B-Instruct", "03_data_doctor")
run("soup data lint data/raw_500.jsonl --format dpo --model Qwen/Qwen2.5-0.5B-Instruct "
    "-o results/lint_raw.json", "03_data_lint")
run("soup data lint data/train.jsonl --format dpo --model Qwen/Qwen2.5-0.5B-Instruct "
    "-o results/lint_train.json", "03_data_lint")
""")

md("## 2. Pre-flight and the memory budget (before training)")

code("""
run("soup train --config soup.yaml --dry-run", "04_preflight_dry_run")
run("python memory_budget.py --config soup.yaml --out results/memory_budget_pre.json", "05_memory_budget")
""")

md("## 3. The DPO run")

code("""
subprocess.run("nvidia-smi > results/nvidia_smi_before_train.txt", shell=True)
rc = run("soup train --config soup.yaml -y", "06_train")
subprocess.run("nvidia-smi > results/nvidia_smi_after_train.txt", shell=True)
print("train exit code", rc)
""")

md("## 4. Memory: prediction vs measured")

code("""
run("python memory_budget.py --config soup.yaml --soup-db soup.db --smi results/nvidia_smi.csv "
    "--out results/memory_budget_post.json", "07_memory_measured")
""")

md("## 5. Prove the model trained (Part 2)")

code("""
run("python verify_training.py --adapter out-dpo --out results/verify.json", "08_verify")
""")

md("## 6. Silent-failure evidence (Part 3)")

code("""
run("python silent_checks.py --run out-dpo --db soup.db --out results/silent_checks.json", "09_silent_checks")
# the trainer's own DPO metrics, copied out of the checkpoint before anything else touches them
run("cp $(ls -d out-dpo/checkpoint-* | sort -t- -k2 -n | tail -1)/trainer_state.json results/trainer_state.json",
    "09_silent_checks")
""")

code("""
subprocess.run("cp -r out-dpo/adapter_config.json out-dpo/adapter_model.safetensors soup.db soup.yaml results/ "
               "&& zip -qr results_core.zip results", shell=True)
from google.colab import files
files.download("results_core.zip")
""")

md("""
## 6b. Control run: length-balanced data

The main set has the chosen answer shorter in ~65% of pairs, and DPO sums log-probs over
tokens, so "prefer shorter" is a cheap shortcut. Same config, same held-out set, but the
training pairs are length-balanced. If the held-out accuracy gap between "chosen shorter"
and "chosen longer" shrinks here, the main run's margin was partly length.
""")

code("""
run("soup train --config soup_lenbal.yaml -y", "06b_train_lenbal",
    env={"SOUP_DB_PATH": os.path.abspath("soup_lenbal.db")})
run("python verify_training.py --adapter out-lenbal --train data/train_lenbal.jsonl "
    "--out results/verify_lenbal.json", "08b_verify_lenbal")
run("python silent_checks.py --run out-lenbal --train data/train_lenbal.jsonl --db soup_lenbal.db "
    "--out results/silent_checks_lenbal.json", "09b_silent_checks_lenbal")
""")

md("""
## 7. `soup ship` against the real adapter and two adapters that must not ship

`ctrl-zero` is the trained adapter with every `lora_B` zeroed (a run that trained nothing).
`ctrl-random` has random `lora_B` with the trained per-tensor norms (changed, but learned nothing).

Leg 2 is limited to `mini_arithmetic`: the full default suite generates ~500 answers of up
to 256 tokens one at a time, about an hour per adapter. Everything up to here is already
zipped, so if this section runs long you can stop it and still have the core results.
""")

code("""
run("python make_ship_inputs.py --adapter out-dpo", "10_ship_inputs")
for name, adapter in [("trained", "out-dpo"), ("zero", "ctrl-zero"), ("random", "ctrl-random")]:
    run(f"soup ship --base Qwen/Qwen2.5-0.5B-Instruct --adapter {adapter} --task-eval task_eval.jsonl "
        f"--general-suite mini_arithmetic --device cuda -o results/ship_{name}.json", f"11_ship_{name}")
""")

md("## 8. Re-run lint with the optional `[data]` extra (was a check skipped?)")

code("""
run('pip install -q "soup-cli[data]==0.75.1"', "12_lint_with_data_extra")
run("soup data lint data/raw_500.jsonl --format dpo --model Qwen/Qwen2.5-0.5B-Instruct "
    "-o results/lint_raw_with_datasketch.json", "12_lint_with_data_extra")
""")

md("## 9. Bundle everything for download")

code("""
smi.terminate()
subprocess.run("nvidia-smi > results/nvidia_smi_end.txt", shell=True)
subprocess.run("cp soup.db results/ && cp -r out-dpo/adapter_config.json out-dpo/adapter_model.safetensors results/ "
               "&& cp soup.yaml results/", shell=True)
subprocess.run("zip -qr results.zip results", shell=True)
from google.colab import files
files.download("results.zip")
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
json.dump(nb, open("run_colab.ipynb", "w"), indent=1)
print("wrote run_colab.ipynb")

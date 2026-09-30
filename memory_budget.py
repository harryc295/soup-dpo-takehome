"""Part 1: VRAM budget, worked out BEFORE training from the model config alone.

Deliberately independent of Soup's own estimator (utils/layer_stream.py), so the two
can be compared. Afterwards, --measured takes the trainer's peak and nvidia-smi's
peak and explains the gap.

Usage
  python memory_budget.py --config soup.yaml --out results/memory_budget.json
  python memory_budget.py --config soup.yaml --soup-db soup.db --smi results/nvidia_smi.csv
"""

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import yaml
from transformers import AutoConfig

GB = 1e9
GiB = 1024**3


def budget(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    mc = AutoConfig.from_pretrained(cfg["base"])
    t, lora = cfg["training"], cfg["training"]["lora"]
    h, i, L, V = mc.hidden_size, mc.intermediate_size, mc.num_hidden_layers, mc.vocab_size
    kv = mc.num_key_value_heads * (h // mc.num_attention_heads)
    fp16 = 2  # T4 has no bf16; Soup streams fp16
    batch, seq = t["batch_size"], cfg["data"]["max_length"]
    rows = 2 * batch  # DPO: chosen + rejected concatenated into one batch
    tokens = rows * seq  # worst case: every row padded to max_length

    # decoder layer: q,k,v,o (+ q/k/v biases in qwen2), gate/up/down MLP, 2 RMSNorms
    layer_params = h * h + 2 * h * kv + h * h + (h + 2 * kv) + 3 * h * i + 2 * h
    embed_params = V * h  # tied: the same matrix is the LM head

    dims = {"q_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (h, h)}
    r = lora["r"]
    adapter_params = L * sum(r * (din + dout) for din, dout in
                             (dims[m] for m in lora["target_modules"]))

    terms = {
        # weights actually resident on the GPU
        "layer_buffers (stream_buffers x one fp16 layer)": t.get("stream_buffers", 2) * layer_params * fp16,
        "embeddings / tied lm_head (fp16, resident)": embed_params * fp16,
        # trainable state: fp32 weight + fp32 grad + Adam m + v = 16 B/param
        "lora weights+grads+adam (16 B/param)": adapter_params * 16,
        # TRL keeps a frozen copy of the starting adapter as the DPO reference ('ref')
        "dpo ref adapter copy (fp32)": adapter_params * 4,
        # activations: every layer is checkpointed, so only its fp16 input is kept,
        # plus one layer's full activations live during recompute/backward
        "checkpointed layer inputs (L x tokens x h x 2B)": L * tokens * h * fp16,
        "one layer recompute (attn+mlp, ~ tokens x (4h + 3i) x 2B)": tokens * (4 * h + 3 * i) * fp16,
        # logits dominate: tokens x vocab. fp16 logits, fp32 upcast for log_softmax,
        # fp32 log-probs kept for backward, fp32 grad of logits = 2 + 4 + 4 + 4 = 14 B
        "policy logits + loss (tokens x vocab x 14 B)": tokens * V * 14,
        # the reference pass runs under no_grad; its fp16 logits + fp32 log-softmax are
        # transient, freed before the policy backward, so they don't stack on the peak
        "ref pass (not counted at peak, transient)": 0,
    }
    total = sum(terms.values())

    # Correction found by reading trl/trainer/dpo_trainer.py::_compute_loss after a local
    # rehearsal came in ~23% over: the reference forward runs AFTER the policy forward,
    # while the policy's logits and their autograd graph are still alive. So the ref
    # pass's fp16 logits and the .contiguous() copy of its shifted logits (2 + 2 B per
    # element) stack on the policy's, instead of being transient.
    ref_overlap = tokens * V * 4
    corrected = total + ref_overlap
    host = {
        "pinned host RAM for streamed base (L x layer fp16)": L * layer_params * fp16,
    }
    return {
        "model": cfg["base"],
        "shape": {"hidden": h, "intermediate": i, "layers": L, "vocab": V, "kv_dim": kv},
        "batch": batch, "rows_per_step": rows, "seq_len": seq, "tokens_per_step": tokens,
        "layer_params": layer_params, "embed_params": embed_params,
        "adapter_params": adapter_params,
        "terms_GB": {k: round(v / GB, 3) for k, v in terms.items()},
        "first_pass_peak_GB": round(total / GB, 3),
        "correction_ref_pass_overlaps_policy_graph_GB (tokens x vocab x 4 B)": round(ref_overlap / GB, 3),
        "predicted_allocator_peak_GB": round(corrected / GB, 3),
        "predicted_allocator_peak_GiB": round(corrected / GiB, 3),
        "logit_bytes_per_element_implied": 18,
        "not_in_allocator_peak": {
            "cuda_context_and_kernels": "~0.3-0.5 GB, visible to nvidia-smi only",
            "allocator_cache (reserved - allocated)": "fragmentation, visible to nvidia-smi only",
        },
        "host_GB": {k: round(v / GB, 3) for k, v in host.items()},
    }


def measured(soup_db, smi_csv):
    out = {}
    if soup_db and Path(soup_db).exists():
        con = sqlite3.connect(soup_db)
        run = con.execute("select run_id from runs order by rowid desc limit 1").fetchone()[0]
        vals = [r[0] for r in con.execute("select gpu_mem from metrics where run_id=?", (run,))]
        # Soup logs torch.cuda.max_memory_allocated() formatted in GiB but labelled "GB"
        out["soup_logged_peak_raw"] = max(vals, key=lambda s: float(s.split("/")[0]))
        gib = float(out["soup_logged_peak_raw"].split("/")[0])
        out["soup_logged_peak_as_GB"] = round(gib * GiB / GB, 2)
    if smi_csv and Path(smi_csv).exists():
        used = []
        for row in csv.DictReader(open(smi_csv)):
            key = next(k for k in row if "memory.used" in k)
            used.append(float(row[key].split()[0]))
        out["nvidia_smi_peak_MiB"] = max(used)
        out["nvidia_smi_peak_GB"] = round(max(used) * 1024**2 / GB, 2)
        out["nvidia_smi_samples"] = len(used)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="soup.yaml")
    ap.add_argument("--soup-db")
    ap.add_argument("--smi")
    ap.add_argument("--out", default="results/memory_budget.json")
    args = ap.parse_args()
    res = budget(args.config)
    if args.soup_db or args.smi:
        res["measured"] = measured(args.soup_db, args.smi)
    print(json.dumps(res, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

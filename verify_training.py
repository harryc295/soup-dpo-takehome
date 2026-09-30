"""Part 2: prove the DPO run changed the model in the direction the data asked for.

Falling loss is not proof. It can fall while the adapter is dead (for example the
loss is computed against a policy that is not the one saved), and a SHIP message
does not look at the adapter at all. So this script checks the SAVED ARTIFACT,
through a code path independent of Soup's streamed trainer: a plain resident
transformers + PEFT model loads adapter_model.safetensors from disk.

Checks
  A. Weights  every targeted (layer, module) has a non-zero lora_B, the update is
              non-trivial relative to the base weight, and the saved `ref/` adapter
              really is the zero (identity) adapter DPO compared against.
  B. Behaviour implicit DPO reward margin
                 m = beta * [(lp_pi(c) - lp_ref(c)) - (lp_pi(r) - lp_ref(r))]
              on pairs the model trained on AND on 100 held-out pairs it never saw.
              ref = base model (adapter disabled).
  C. Controls  the same numbers for (i) a zero adapter and (ii) a random adapter with
              exactly the trained adapter's per-tensor norms but random direction.
              A real run beats the random control; "the model changed" alone does not.
  D. Confound  correlation of the margin with (len(chosen) - len(rejected)), so a
              model that only learned "prefer the shorter answer" is visible.

Usage
  python verify_training.py --adapter out-dpo --out results/verify.json
Exit code 0 = run is real (checks A-C pass), 1 = it is not.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

BETA = 0.1
MAX_LEN = 512
MAX_PROMPT = MAX_LEN // 2  # Soup caps the prompt at max_length/2


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- A. weights
def weight_checks(adapter_dir, base_model):
    sd = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    base_sd = base_model.state_dict()

    rows = []
    for k in sorted(sd):
        if "lora_B" not in k:
            continue
        a_key = k.replace("lora_B", "lora_A")
        B, A = sd[k].float(), sd[a_key].float()
        delta = scale * (B @ A)
        # base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight -> layers.3...q_proj.weight
        base_key = k.split("base_model.model.", 1)[1].replace(".lora_B.weight", ".weight")
        w = base_sd[base_key].float().cpu()
        layer = int(k.split("layers.")[1].split(".")[0])
        module = k.split(".lora_B")[0].rsplit(".", 1)[1]
        rows.append({
            "layer": layer,
            "module": module,
            "B_norm": B.norm().item(),
            "delta_rel": (delta.norm() / w.norm()).item(),
        })

    n_layers = base_model.config.num_hidden_layers
    expected = {(l, m) for l in range(n_layers) for m in cfg["target_modules"]}
    present = {(r["layer"], r["module"]) for r in rows}
    zero = [r for r in rows if r["B_norm"] == 0.0]

    ref_dir = Path(adapter_dir) / "ref"
    ref_zero = None
    if (ref_dir / "adapter_model.safetensors").exists():
        ref_sd = load_file(str(ref_dir / "adapter_model.safetensors"))
        ref_zero = all(v.abs().max().item() == 0 for k, v in ref_sd.items() if "lora_B" in k)

    per_layer = {}
    for r in rows:
        per_layer.setdefault(r["layer"], []).append(r["delta_rel"])
    per_layer = {l: sum(v) / len(v) for l, v in sorted(per_layer.items())}

    return {
        "lora_B_tensors": len(rows),
        "expected_tensors": len(expected),
        "missing": sorted(expected - present),
        "zero_B_tensors": [(r["layer"], r["module"]) for r in zero],
        "min_delta_rel": min(r["delta_rel"] for r in rows),
        "max_delta_rel": max(r["delta_rel"] for r in rows),
        "mean_delta_rel_per_layer": per_layer,
        "ref_adapter_is_zero": ref_zero,
    }


# ---------------------------------------------------------------- B. behaviour
def encode(tok, row):
    prompt_text = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    out = []
    for side in ("chosen", "rejected"):
        full = tok.apply_chat_template(row["prompt"] + row[side], tokenize=False)
        assert full.startswith(prompt_text), "chat template does not extend the prompt"
        p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"][-MAX_PROMPT:]
        c_ids = tok(full[len(prompt_text):], add_special_tokens=False)["input_ids"]
        c_ids = c_ids[: MAX_LEN - len(p_ids)]
        out.append((p_ids, c_ids))
    return out


@torch.no_grad()
def completion_logp(model, p_ids, c_ids, device):
    ids = torch.tensor([p_ids + c_ids], device=device)
    logits = model(input_ids=ids).logits.float()
    logp = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = ids[0, 1:]
    tok_lp = logp.gather(-1, targets[:, None])[:, 0]
    return tok_lp[len(p_ids) - 1:].sum().item()


def score(model, encoded, device):
    return [
        (completion_logp(model, *c, device), completion_logp(model, *r, device))
        for c, r in encoded
    ]


def margins(pi, ref):
    return [BETA * ((pc - rc) - (pr - rr)) for (pc, pr), (rc, rr) in zip(pi, ref)]


def summarise(m, seed=0):
    n = len(m)
    wins = sum(x > 0 for x in m)
    ties = sum(x == 0 for x in m)
    # two-sided exact sign test vs 50%, ties dropped
    k, nn = wins, n - ties
    p = min(1.0, 2 * sum(math.comb(nn, i) for i in range(max(k, nn - k), nn + 1)) / 2 ** nn) if nn else 1.0
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(m) for _ in range(n)) / n for _ in range(2000))
    return {
        "n": n,
        "mean_margin": sum(m) / n,
        "mean_margin_ci95": [boots[50], boots[1949]],
        "accuracy": wins / n,
        "ties": ties,
        "sign_test_p": p,
    }


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else 0.0


def make_control(model, name, trained_sd, mode, seed=0):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for k, v in trained_sd.items():
        if "lora_B" in k and mode == "zero":
            v = torch.zeros_like(v)
        elif "lora_B" in k and mode == "random":
            r = torch.randn(v.shape, generator=g).to(v)
            v = r * (v.float().norm() / r.float().norm()).to(v.dtype)
        sd[k] = v.clone()
    model.add_adapter(name, model.peft_config["trained"])
    set_peft_model_state_dict(model, sd, adapter_name=name)


def load_jsonl(path, n=None):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return rows[:n] if n else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="out-dpo")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--heldout", default="data/heldout.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/verify.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_id = json.loads((Path(args.adapter) / "adapter_config.json").read_text())[
        "base_model_name_or_path"
    ]
    tok = AutoTokenizer.from_pretrained(base_id)
    # fp32 so the margins measure the adapter, not fp16 rounding
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32).to(device).eval()

    log("== A. weight checks (saved adapter on disk)")
    A = weight_checks(args.adapter, base)
    log(json.dumps({k: v for k, v in A.items() if k != "mean_delta_rel_per_layer"}, indent=1))
    log("mean |dW|/|W| per layer: " + " ".join(
        f"{l}:{v:.2e}" for l, v in A["mean_delta_rel_per_layer"].items()))

    model = PeftModel.from_pretrained(base, args.adapter, adapter_name="trained").eval()
    trained_sd = get_peft_model_state_dict(model, adapter_name="trained")
    make_control(model, "zero", trained_sd, "zero")
    make_control(model, "random", trained_sd, "random")

    sets = {
        "train": [encode(tok, r) for r in load_jsonl(args.train, args.n)],
        "heldout": [encode(tok, r) for r in load_jsonl(args.heldout, args.n)],
    }
    B = {}
    for split, enc in sets.items():
        log(f"== B/C. scoring {split} ({len(enc)} pairs)")
        with model.disable_adapter():
            ref = score(model, enc, device)
        res = {"base_prefers_chosen": sum(c > r for c, r in ref) / len(ref)}
        per_adapter = {}
        for name in ("trained", "zero", "random"):
            model.set_adapter(name)
            per_adapter[name] = margins(score(model, enc, device), ref)
            res[name] = summarise(per_adapter[name])
            log(f"  {name:8s} {json.dumps(res[name])}")
        model.set_adapter("trained")
        m = per_adapter["trained"]
        len_diff = [len(c[1]) - len(r[1]) for c, r in enc]
        res["margin_vs_len_diff_pearson"] = pearson(m, len_diff)
        res["accuracy_when_chosen_is_shorter"] = (
            sum(x > 0 for x, d in zip(m, len_diff) if d < 0) / max(1, sum(d < 0 for d in len_diff))
        )
        res["accuracy_when_chosen_is_longer"] = (
            sum(x > 0 for x, d in zip(m, len_diff) if d > 0) / max(1, sum(d > 0 for d in len_diff))
        )
        log(f"  length confound: pearson(margin, len_c-len_r)={res['margin_vs_len_diff_pearson']:.3f}"
            f"  acc|chosen shorter={res['accuracy_when_chosen_is_shorter']:.2f}"
            f"  acc|chosen longer={res['accuracy_when_chosen_is_longer']:.2f}")
        B[split] = res

    checks = {
        "every_target_module_has_nonzero_B": not A["missing"] and not A["zero_B_tensors"],
        "ref_adapter_is_identity": A["ref_adapter_is_zero"] is True,
        "zero_control_margin_is_zero": all(B[s]["zero"]["mean_margin"] == 0 for s in B),
        "learned_train_pairs": B["train"]["trained"]["mean_margin_ci95"][0] > 0,
        "beats_random_control_on_train": (
            B["train"]["trained"]["mean_margin"] > B["train"]["random"]["mean_margin_ci95"][1]
        ),
        "generalises_heldout": (
            B["heldout"]["trained"]["mean_margin_ci95"][0] > 0
            and B["heldout"]["trained"]["sign_test_p"] < 0.05
        ),
    }
    real = all(v for k, v in checks.items() if k != "generalises_heldout")
    verdict = {
        "run_is_real": real,
        "generalises": checks["generalises_heldout"],
        "checks": checks,
    }
    log("== verdict\n" + json.dumps(verdict, indent=1))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"weights": A, "behaviour": B, "verdict": verdict},
                                         indent=1, default=str))
    raise SystemExit(0 if real else 1)


if __name__ == "__main__":
    main()

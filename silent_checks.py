"""Part 3: evidence for the silent-failure analysis.

Every check here targets a way the run could finish "green" and still be wrong.
Nothing needs a GPU except what the trainer already wrote.

  1. truncation    what max_length=512 (prompt capped at 256) does to the pairs
  2. length        chosen vs rejected length, since DPO sums log-probs over tokens
  3. log_history   the DPO metrics TRL logged: start at ln2? NaN/inf? skipped fp16
                   steps? both log-probs falling (likelihood displacement)?
  4. soup db       what Soup's own run history kept (and what it dropped)
  5. russian       token cost of Cyrillic vs English under this tokenizer, because
                   the real dataset is Russian and truncation scales with it

Usage
  python silent_checks.py --run out-dpo --db soup.db --out results/silent_checks.json
"""

import argparse
import glob
import json
import math
import sqlite3
from pathlib import Path

from transformers import AutoTokenizer

MAX_LEN = 512
MAX_PROMPT = MAX_LEN // 2


def pct(a, b):
    return round(100 * a / b, 1) if b else 0.0


def truncation_and_length(tok, path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    n = len(rows)
    p_trunc = c_trunc = r_trunc = both_trunc = identical_after = chosen_shorter = 0
    c_lens, r_lens = [], []
    for row in rows:
        prompt_text = tok.apply_chat_template(row["prompt"], tokenize=False,
                                              add_generation_prompt=True)
        p = tok(prompt_text, add_special_tokens=False)["input_ids"]
        p_kept = p[-MAX_PROMPT:]
        p_trunc += len(p) > MAX_PROMPT
        budget = MAX_LEN - len(p_kept)
        sides = {}
        for side in ("chosen", "rejected"):
            full = tok.apply_chat_template(row["prompt"] + row[side], tokenize=False)
            ids = tok(full[len(prompt_text):], add_special_tokens=False)["input_ids"]
            sides[side] = ids
        c, r = sides["chosen"], sides["rejected"]
        c_lens.append(len(c))
        r_lens.append(len(r))
        chosen_shorter += len(c) < len(r)
        ct, rt = len(c) > budget, len(r) > budget
        c_trunc += ct
        r_trunc += rt
        both_trunc += ct and rt
        identical_after += c[:budget] == r[:budget]
    return {
        "pairs": n,
        "prompt_truncated_pct": pct(p_trunc, n),
        "chosen_truncated_pct": pct(c_trunc, n),
        "rejected_truncated_pct": pct(r_trunc, n),
        "both_truncated_pct (end-of-turn token lost on both)": pct(both_trunc, n),
        "identical_after_truncation": identical_after,
        "chosen_mean_tokens": round(sum(c_lens) / n, 1),
        "rejected_mean_tokens": round(sum(r_lens) / n, 1),
        "chosen_shorter_pct": pct(chosen_shorter, n),
    }


def log_history(run_dir):
    ckpts = sorted(glob.glob(f"{run_dir}/checkpoint-*/trainer_state.json"),
                   key=lambda p: int(Path(p).parent.name.split("-")[1]))
    if not ckpts:
        return {"error": "no checkpoint trainer_state.json - DPO metrics were never written"}
    state = json.loads(Path(ckpts[-1]).read_text())
    hist = [h for h in state["log_history"] if "loss" in h]
    steps = [h["step"] for h in hist]

    def series(key):
        return [h.get(key) for h in hist]

    def bad(xs):
        return sum(1 for x in xs if x is None or (isinstance(x, float) and not math.isfinite(x)))

    loss, gn = series("loss"), series("grad_norm")
    lpc, lpr = series("logps/chosen"), series("logps/rejected")
    acc, marg = series("rewards/accuracies"), series("rewards/margins")
    rc, rr = series("rewards/chosen"), series("rewards/rejected")
    q = max(1, len(hist) // 4)

    def mean(xs):
        xs = [x for x in xs if x is not None and math.isfinite(x)]
        return sum(xs) / len(xs) if xs else None

    return {
        "checkpoint_used": ckpts[-1],
        "logged_steps": len(hist),
        "max_step": state.get("max_steps"),
        "global_step": state.get("global_step"),
        "missing_steps": sorted(set(range(1, (state.get("global_step") or 0) + 1)) - set(steps)),
        "first_loss": loss[0],
        "first_loss_minus_ln2": loss[0] - math.log(2),
        "nonfinite_loss": bad(loss),
        "nonfinite_grad_norm (fp16 GradScaler skipped steps)": bad(gn),
        "grad_norm_max": max((g for g in gn if g is not None and math.isfinite(g)), default=None),
        "steps_with_loss_exactly_ln2": sum(1 for x in loss if x is not None
                                           and abs(x - math.log(2)) < 1e-4),
        "loss_first_quarter_mean": mean(loss[:q]),
        "loss_last_quarter_mean": mean(loss[-q:]),
        "reward_acc_first_quarter": mean(acc[:q]),
        "reward_acc_last_quarter": mean(acc[-q:]),
        "reward_margin_last_quarter": mean(marg[-q:]),
        "rewards_chosen_last_quarter": mean(rc[-q:]),
        "rewards_rejected_last_quarter": mean(rr[-q:]),
        "logps_chosen_first_vs_last_quarter": [mean(lpc[:q]), mean(lpc[-q:])],
        "logps_rejected_first_vs_last_quarter": [mean(lpr[:q]), mean(lpr[-q:])],
        "likelihood_displacement (chosen reward < 0 at end)": (mean(rc[-q:]) or 0) < 0,
    }


def soup_db(db_path):
    if not Path(db_path).exists():
        return {"error": f"{db_path} not found"}
    con = sqlite3.connect(db_path)
    run = con.execute("select run_id from runs order by rowid desc limit 1").fetchone()[0]
    cols = [d[0] for d in con.execute("select * from metrics limit 1").description]
    rows = con.execute("select step, loss, grad_norm, speed, gpu_mem from metrics "
                       "where run_id=? order by id", (run,)).fetchall()
    return {
        "run_id": run,
        "metric_columns": cols,
        "dpo_reward_metrics_stored": any("reward" in c for c in cols),
        "rows": len(rows),
        "gpu_mem_values_seen": sorted({r[4] for r in rows}),
        "speed_nonzero_rows": sum(1 for r in rows if r[3]),
    }


RUSSIAN_PROBE = [
    ("My order has not arrived yet and the tracking number does not work. "
     "Please help me find out where the parcel is.",
     "Мой заказ до сих пор не пришёл, а трек-номер не работает. "
     "Помогите, пожалуйста, узнать, где посылка."),
    ("I was charged twice for one subscription. I want a refund for the second payment.",
     "С меня дважды списали деньги за одну подписку. Хочу вернуть второй платёж."),
    ("The app crashes every time I try to log in after the latest update.",
     "Приложение вылетает каждый раз, когда я пытаюсь войти после последнего обновления."),
]


def russian_cost(tok):
    en = sum(len(tok(e)["input_ids"]) for e, _ in RUSSIAN_PROBE)
    ru = sum(len(tok(r)["input_ids"]) for _, r in RUSSIAN_PROBE)
    return {"english_tokens": en, "russian_tokens": ru, "ru_over_en": round(ru / en, 2),
            "note": "same meaning, hand-translated; more tokens = more truncation at 512"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out-dpo")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--db", default="soup.db")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="results/silent_checks.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    out = {
        "truncation_and_length": truncation_and_length(tok, args.train),
        "log_history": log_history(args.run),
        "soup_db": soup_db(args.db),
        "russian_tokens": russian_cost(tok),
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

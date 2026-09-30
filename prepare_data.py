"""Build the preference dataset.

Soup's package (500 Russian support-ticket pairs) was not available, so this uses
argilla/distilabel-intel-orca-dpo-pairs as the email allowed. That dataset is useful
here because it is messy in known ways: it keeps ties and its relabelling swapped
some pairs, so there is real label noise to find.

Outputs (data/):
  raw_500.jsonl     the first 500 rows sampled as-is (seed 0), before any cleaning
  train.jsonl       400 cleaned rows used for training
  heldout.jsonl     100 cleaned rows never trained on, used by verify_training.py
  prep_report.json  what was dropped and why

Cleaning drops ~30% of rows (mostly ties), so sampling continues past the first
500 until there are 500 clean pairs, matching the ~500 the brief describes.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 0
N_RAW = 500
N_CLEAN = 500
N_HELDOUT = 100
OUT = Path("data")


def to_row(r):
    prompt = []
    if (r["system"] or "").strip():
        prompt.append({"role": "system", "content": r["system"]})
    prompt.append({"role": "user", "content": r["input"]})
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": r["chosen"]}],
        "rejected": [{"role": "assistant", "content": r["rejected"]}],
        "_status": r["status"],
        "_chosen_score": r["chosen_score"],
        "_in_gsm8k_train": r["in_gsm8k_train"],
    }


def write(path, rows, keep_meta=False):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if not keep_meta:
                r = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    ds = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")
    order = list(range(len(ds)))
    random.Random(SEED).shuffle(order)
    raw = [to_row(ds[i]) for i in order[:N_RAW]]
    write(OUT / "raw_500.jsonl", raw)

    dropped = {"tie": 0, "identical": 0, "empty": 0, "gsm8k_train_overlap": 0}
    clean, seen = [], 0
    for i in order:
        if len(clean) == N_CLEAN:
            break
        r = to_row(ds[i])
        seen += 1
        c, j = r["chosen"][0]["content"].strip(), r["rejected"][0]["content"].strip()
        if r["_status"] == "tie":
            dropped["tie"] += 1
        elif not c or not j:
            dropped["empty"] += 1
        elif c == j:
            dropped["identical"] += 1
        elif r["_in_gsm8k_train"]:
            dropped["gsm8k_train_overlap"] += 1
        else:
            clean.append(r)

    random.Random(SEED + 1).shuffle(clean)
    heldout, train = clean[:N_HELDOUT], clean[N_HELDOUT:]
    write(OUT / "train.jsonl", train)
    write(OUT / "heldout.jsonl", heldout)

    status_counts = {}
    for r in raw:
        status_counts[r["_status"]] = status_counts.get(r["_status"], 0) + 1
    report = {
        "source": "argilla/distilabel-intel-orca-dpo-pairs",
        "seed": SEED,
        "raw_rows": len(raw),
        "raw_status_counts": status_counts,
        "rows_scanned_for_500_clean": seen,
        "dropped": dropped,
        "train_rows": len(train),
        "heldout_rows": len(heldout),
    }
    (OUT / "prep_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

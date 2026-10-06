"""Golden fixtures for the Rust port, straight from the `laya` package.

    uv run python -m ollaya_convert.goldens en --out ../crates/ollaya-decision/tests/fixtures
    uv run python -m ollaya_convert.goldens en --out out/goldens --all      # every case, local only

One JSON line per case:
    {"id", "state", "questions",
     "items": [{"qid", "qtype", "ids", "markers"}],   # the exact encoder input per question
     "logits": [[...]], "act_logits": [[...]],         # network outputs, markers only
     "answers": {...}}                                 # laya's own system_one output

Expected answers come from laya's own `Agent.system_one` on CPU, where laya runs in fp32 without
autocast, so they are the reference numbers, not a re-implementation of them.

`--precision fp64` runs the checkpoint's network (same weights, same code) in float64 instead:
`laya_ref.Exact`, with ModernBERT's rotary embedding kept in float64 too, and `system_one` on that
network for the answers. It measures how far the fp32 goldens sit from the exact network (#6); the
shipped goldens stay fp32.

`--device cuda` is the fast path: the network runs on the GPU (TF32 off for fp32). In fp32 `answers`
is null there, since `system_one` would autocast to bf16; in fp64 the answers are the exact
network's wherever it runs, and the large `--all` sets leave them null as the fp32 ones did.

Questions whose criteria dict uses non-string keys (e.g. the noul True/False keys) are stored with
the keys laya normalises them to, since JSON has only string keys.
"""
import argparse
import json
import os

import torch

from . import cases, laya_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", choices=sorted(laya_ref.CHECKPOINTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=laya_ref.DEFAULT_ROOT)
    ap.add_argument("--all", action="store_true", help="all 400 typed-decisions rows, not 20")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--precision", choices=["fp32", "fp64"], default="fp32",
                    help="reference numbers: upstream's fp32 forward (default, the shipped goldens), or the network in float64")
    a = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fp64 = a.precision == "fp64"
    agent = laya_ref.load(a.checkpoint, root=a.root, device="cpu" if fp64 else a.device)
    exact = laya_ref.Exact(agent, a.device) if fp64 else None
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "laya-%s.jsonl" % a.checkpoint)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for cid, state, questions in cases.all_cases(0 if a.all else 20):
            enc = laya_ref.encode(agent, state, questions)
            logits, act = laya_ref.forward(agent, enc["batch"], exact)
            answers = None
            if a.device == "cpu" or (fp64 and not a.all):
                answers = (laya_ref.system_one_exact(agent, exact, state, questions) if fp64
                           else agent.system_one(state, questions))["answers"]
            items = []
            for r, it in enumerate(enc["items"]):
                k = len(it["markers"])
                items.append({"qid": it["qid"], "qtype": it["qtype"], "ids": it["ids"],
                              "markers": it["markers"],
                              "logits": [float(x) for x in logits[r, :k]],
                              "act_logits": [float(x) for x in act[r]]})
            rec = {"id": cid, "state": state, "questions": questions, "items": items, "answers": answers}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            n += 1
    print("wrote %d cases to %s (%.1f MB)" % (n, path, os.path.getsize(path) / 2**20))


if __name__ == "__main__":
    main()

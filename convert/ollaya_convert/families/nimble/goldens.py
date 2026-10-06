"""Golden fixtures for the Rust port of `nimble-codes-v1`, straight from the author's code in fp32.

    uv run --with peft==0.21.0 python -m ollaya_convert.families.nimble.goldens out/nimble-9b-v2 \
        [--device offload] [--td-limit 20]

Writes out/goldens-nimble-9b-v2.jsonl, one JSON line per request:
    {"id", "state", "questions",
     "error": null | "<message>",                       # upstream rejects the request (HTTP 422)
     "rows": [{"ids": [...], "candidates": [...]}],    # upstream full_ids + candidate ids, request order
     "plan": [{"qid", "type", "k",
               "option_logits": [...],                  # upstream candidate_logits, fp32, TF32 off
               "probabilities": [...]}],                # softmax(option_logits / temperature)
     "answers": {...}}                                  # openjev scoring.answer at that temperature
A rejected request is followed by a line "<id>#valid" with the questions upstream accepts on their own.
Every record's rows are also checked against the layout port (layout.py) before they are written.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import tokenizers

from ..llm_common import cases
from . import ref
from .layout import LayoutError, NimbleLayout


def softmax(z, t):
    z = np.asarray(z, dtype=np.float64) / t
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def record(tok, model, lay, T, adapter, cid, state, questions):
    prepared, meta = ref.prepare(tok, state, questions, adapter)
    rows = [{"ids": [int(x) for x in ids], "candidates": [int(x) for x in cand]}
            for ids, cand in zip(prepared.full_ids, prepared.candidate_ids)]
    port, _ = lay.encode(state, questions)
    assert port == rows, cid
    logits = ref.forward(model, prepared, adapter)
    plan = [{"qid": m["qid"], "type": m["type"], "k": m["k"], "option_logits": [float(x) for x in z],
             "probabilities": softmax(z, T)} for m, z in zip(meta, logits)]
    return {"id": cid, "state": state, "questions": questions, "error": None, "rows": rows, "plan": plan,
            "answers": ref.answers(meta, logits, T)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--td-limit", type=int, default=20)
    ap.add_argument("--device", default="offload", help="cpu, cuda (needs 40 GB) or offload (GPU + CPU memory)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true", help="keep the records already written")
    a = ap.parse_args()

    adapter, base = (a.adapter, a.base) if a.adapter and a.base else ref.snapshot()
    tok, model = ref.load(adapter, base, device=a.device)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json"), encoding="utf-8"))
    T = json.load(open(os.path.join(a.model_dir, "calibration.json"), encoding="utf-8"))["temperature"][0]
    lay = NimbleLayout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")), decision)
    path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.model_dir)), "goldens-nimble-9b-v2.jsonl")
    n, t0 = 0, time.time()
    done = set()
    if a.resume and os.path.exists(path):
        done = {json.loads(line)["id"].split("#")[0] for line in open(path, encoding="utf-8") if line.strip()}
    with open(path, "a" if a.resume else "w", encoding="utf-8") as f:
        for cid, state, questions in cases.all_cases(a.td_limit):
            if cid in done:
                continue
            try:
                rec = record(tok, model, lay, T, adapter, cid, state, questions)
            except ref.RequestError as e:
                try:
                    lay.encode(state, questions)
                    raise AssertionError("%s: upstream rejects (%s), the port accepts" % (cid, e))
                except LayoutError:
                    pass
                f.write(json.dumps({"id": cid, "state": state, "questions": questions, "error": str(e)},
                                   ensure_ascii=False) + "\n")
                valid = {}
                for qid, q in questions.items():
                    try:
                        ref.prepare(tok, state, {qid: q}, adapter)
                        valid[qid] = q
                    except ref.RequestError:
                        pass
                if not valid:
                    continue
                rec = record(tok, model, lay, T, adapter, cid + "#valid", state, valid)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            print("%4d %-50s %5.0fs" % (n, cid, time.time() - t0), flush=True)
    print("wrote %d records to %s (%.1f MB)" % (n, path, os.path.getsize(path) / 2**20))


if __name__ == "__main__":
    main()

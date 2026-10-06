"""Golden fixtures for the Rust port of `jeeves-markers-v1`, from the authors' own code in fp32 (no thinking).

    JEEVES_SRC=<checkout> uv run --with safetensors python -m ollaya_convert.families.jeeves.goldens out/jeeves-9b \
        [--model SNAPSHOT] [--td-limit 10] [--device cpu]

Writes out/goldens-jeeves-9b.jsonl, one JSON line per request (the kev fixture format):
    {"id", "state", "questions", "error": null | "<message>",
     "rows": [{"ids", "decide", "opts"}], "plan": [{"qid", "type", "k", "option_logits", "probabilities"}]}
A rejected request is followed by "<id>#valid" with the questions upstream accepts on their own. Every
record's rows are checked against the layout port before they are written; --resume keeps finished records.
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
from .layout import JeevesLayout, LayoutError


def softmax(z, t):
    z = np.asarray(z, dtype=np.float64) / t
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--model", default=None)
    ap.add_argument("--td-limit", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    snap = a.model or ref.snapshot()
    model, head, enc = ref.load(snap, device=a.device)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json"), encoding="utf-8"))
    T = json.load(open(os.path.join(a.model_dir, "calibration.json"), encoding="utf-8"))["temperature"][0]
    lay = JeevesLayout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")), decision)
    path = os.path.join(os.path.dirname(os.path.abspath(a.model_dir)), "goldens-jeeves-9b.jsonl")
    done = set()
    if a.resume and os.path.exists(path):
        done = {json.loads(line)["id"].split("#")[0] for line in open(path, encoding="utf-8") if line.strip()}

    def record(cid, state, questions):
        rows, meta = ref.encode(enc, state, questions)
        port, _ = lay.encode(state, questions)
        assert port == rows, cid
        scores = ref.forward(model, head, rows)
        plan = [{"qid": m["qid"], "type": m["type"], "k": m["k"], "option_logits": [float(x) for x in z],
                 "probabilities": softmax(z, T)} for m, z in zip(meta, scores)]
        return {"id": cid, "state": state, "questions": questions, "error": None, "rows": rows, "plan": plan}

    n, t0 = 0, time.time()
    with open(path, "a" if a.resume else "w", encoding="utf-8") as f:
        for cid, state, questions in cases.all_cases(a.td_limit):
            if cid in done:
                continue
            try:
                rec = record(cid, state, questions)
            except ref.RequestError as e:
                try:
                    lay.encode(state, questions)
                    raise AssertionError("%s: upstream rejects (%s), the port accepts" % (cid, e))
                except LayoutError:
                    pass
                f.write(json.dumps({"id": cid, "state": state, "questions": questions, "error": str(e)}, ensure_ascii=False) + "\n")
                valid = {}
                for qid, q in questions.items():
                    try:
                        ref.encode(enc, state, {qid: q})
                        valid[qid] = q
                    except ref.RequestError:
                        pass
                if not valid:
                    continue
                rec = record(cid + "#valid", state, valid)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            print("%4d %-50s %5.0fs" % (n, cid, time.time() - t0), flush=True)
    print("wrote %d records to %s" % (n, path))


if __name__ == "__main__":
    main()

"""Golden fixtures for the Rust port of `arbiter-fixed-v1`, from the reference model in fp32 (ref.py).

    uv run --with peft==0.19.1 --with transformers==4.57.6 \
        python -m ollaya_convert.families.arbiter.goldens out/arbiter-4b --requests shared.jsonl \
        [--run RUN --base BASE] [--device cuda] [--resume]

`out/arbiter-4b` is export.py's output (decision.json, calibration.json, tokenizer.json). The requests are
the shared set (`--requests` JSONL, else `llm_common.cases` with `--td` typed-decisions rows). Writes
out/goldens-arbiter-4b.jsonl, one JSON line per request:
    {"id", "state", "questions", "error": null | "<message>",
     "rows": [{"ids", "last_pos", "slots"}],                    # one row per question, request order
     "plan": [{"qid", "type", "k", "slots",
               "scores": [24 raw slot scores],                   # fp32 reference, LoRA merged, TF32 off
               "option_logits": [scores at slots],               # Ollaya's option order (noul: false, true)
               "probabilities": softmax(option_logits / T)}]}
A rejected request is followed by "<id>#valid" with the questions the reference accepts on their own. Every
record's rows are checked against the layout port before they are written.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import tokenizers

from . import ref
from .check import requests
from .layout import ArbiterLayout, LayoutError


def softmax(z, t):
    z = np.asarray(z, dtype=np.float64) / t
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--requests", default=None, help="the shared request set as JSONL")
    ap.add_argument("--td", type=int, default=40, help="typed-decisions rows when --requests is not given")
    ap.add_argument("--run", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    run, base = (a.run, a.base) if a.run and a.base else ref.snapshot("arbiter-4b")
    tok = ref.tokenizer(base)
    model, head = ref.load(run, base, device=a.device, merge=True)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json")))
    T = json.load(open(os.path.join(a.model_dir, "calibration.json")))["temperature"][0]
    lay = ArbiterLayout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")), decision)
    path = os.path.join(os.path.dirname(os.path.abspath(a.model_dir)), "goldens-arbiter-4b.jsonl")
    done = set()
    if a.resume and os.path.exists(path):
        done = {json.loads(line)["id"].split("#")[0] for line in open(path, encoding="utf-8") if line.strip()}

    def record(cid, state, questions):
        rows, meta = ref.encode(tok, state, questions)
        port, _ = lay.encode(state, questions)
        assert port == rows, cid
        plan = []
        for m, r, s in zip(meta, rows, ref.forward(model, head, rows)):
            z = [float(s[i]) for i in r["slots"]]
            plan.append({"qid": m["qid"], "type": m["type"], "k": m["k"], "slots": r["slots"],
                         "scores": [float(x) for x in s], "option_logits": z, "probabilities": softmax(z, T)})
        return {"id": cid, "state": state, "questions": questions, "error": None, "rows": rows, "plan": plan}

    n, t0 = 0, time.time()
    with open(path, "a" if a.resume else "w", encoding="utf-8") as f:
        for cid, state, questions in requests(a.requests, a.td):
            if cid in done:
                continue
            try:
                rec = record(cid, state, questions)
            except ref.RequestError as e:
                try:
                    lay.encode(state, questions)
                    raise AssertionError("%s: the reference rejects (%s), the port accepts" % (cid, e))
                except LayoutError:
                    pass
                f.write(json.dumps({"id": cid, "state": state, "questions": questions, "error": str(e)},
                                   ensure_ascii=False) + "\n")
                valid = {}
                for qid, q in (questions.items() if isinstance(questions, dict) else []):
                    try:
                        ref.encode(tok, state, {qid: q})
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

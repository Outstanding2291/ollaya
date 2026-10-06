"""Golden fixtures for the Rust port of `clef-joint-v1`, from the authors' own code (`joint_schema_model.py`) in fp32.

    uv run python -m ollaya_convert.families.clef.goldens OUT_DIR [--model SNAPSHOT] [--td-limit 40] [--device cpu]

OUT_DIR is the export (its decision.json and tokenizer.json). Writes goldens-clef-flash.jsonl next to it, one JSON
line per request:
    {"id", "state", "questions", "error": null | "<message>",
     "row": {"ids", "questions": [{"qid", "span", "options", "option_ids"}]},
     "plan": [{"qid", "type", "option_ids", "option_logits", "probabilities"}]}
Logits are in upstream's option order (noul [true, false], choice labels sorted, score levels); probabilities are
their softmax. A rejected request is followed by "<id>#valid" with the questions upstream accepts on their own.
Requests Ollaya's shared question parser rejects are skipped. Every row is checked against the layout port before
it is written; --resume keeps finished records.
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
from .check import port_row, upstream_row
from .layout import ClefLayout, LayoutError, ollaya_rule


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--model", default=None)
    ap.add_argument("--td-limit", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    snap = a.model or ref.snapshot()
    model, tok = ref.load(snap, device=a.device)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json"), encoding="utf-8"))
    lay = ClefLayout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")), decision)
    path = os.path.join(os.path.dirname(os.path.abspath(a.model_dir)), "goldens-clef-flash.jsonl")
    done = set()
    if a.resume and os.path.exists(path):
        done = {json.loads(line)["id"].split("#")[0] for line in open(path, encoding="utf-8") if line.strip()}

    def record(cid, state, questions):
        enc = ref.encode(tok, state, questions)
        row = upstream_row(enc)
        assert port_row(lay.encode(state, questions)) == row, cid
        logits = ref.forward(model, tok.pad_token_id, enc)
        plan = [{"qid": q.question_id, "type": questions[q.question_id]["type"], "option_ids": list(q.option_ids),
                 "option_logits": [float(x) for x in z], "probabilities": softmax(z)}
                for q, z in zip(enc.questions, logits)]
        return {"id": cid, "state": state, "questions": questions, "error": None, "row": row, "plan": plan}

    n, t0 = 0, time.time()
    with open(path, "a" if a.resume else "w", encoding="utf-8") as f:
        for cid, state, questions in cases.all_cases(a.td_limit):
            if cid in done or (isinstance(questions, dict) and any(ollaya_rule(k, v) for k, v in questions.items())):
                continue
            try:
                rec = record(cid, state, questions)
            except ref.RequestError as e:
                try:
                    lay.encode(state, questions)
                    raise AssertionError("%s: upstream rejects (%s), the port accepts" % (cid, e))
                except LayoutError:
                    pass
                f.write(json.dumps({"id": cid, "state": state, "questions": questions, "error": str(e)},
                                   ensure_ascii=False) + "\n")
                valid = {}
                for qid, q in (questions.items() if isinstance(questions, dict) else ()):
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

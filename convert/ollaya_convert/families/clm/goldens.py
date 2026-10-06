"""Golden fixtures for the Rust port of `clm-v1`, from the reference in ref.py (upstream `clm.schema`
and heads, the Qwen3-8B encoder in fp32 on its BF16 weights).

    CLM_SRC=/path/to/CLM/src uv run --with requests python -m ollaya_convert.families.clm.goldens \
        out/clm-8b --base BASE --head HEAD.pt

Writes out/goldens-clm-8b.jsonl, one JSON line per request:
    {"id", "state", "questions",
     "error": null | "<exception class>",          # the whole request is rejected (HTTP 422)
     "texts": [{"qid", "state", "keys", "options",  # upstream build_pairs, request order
                "state_ids", "option_ids"}],        # token ids of each text
     "plan": [{"qid", "type", "k", "option_logits", "probabilities"}],
     "answers": {...}}                              # upstream answer_from_logits
A rejected request is followed by a line "<id>#valid" with the questions accepted on their own.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..llm_common import cases
from . import ref


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def record(r, cid, state, questions):
    pairs, logits, answers = r.answer(state, questions)
    texts = [{"qid": qid, "state": st, "keys": keys, "options": opts, "state_ids": r.ids(st),
              "option_ids": [r.ids(o) for o in opts]} for qid, (st, keys, opts) in pairs.items()]
    plan = [{"qid": qid, "type": questions[qid]["type"], "k": len(pairs[qid][1]),
             "option_logits": [float(x) for x in logits[qid]], "probabilities": softmax(logits[qid])}
            for qid in pairs]
    return {"id": cid, "state": state, "questions": questions, "error": None, "texts": texts, "plan": plan,
            "answers": answers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--td-limit", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = ref.Reference(a.base, a.head, device=a.device)
    path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.model_dir)), "goldens-clm-8b.jsonl")
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for cid, state, questions in cases.all_cases(a.td_limit):
            try:
                rec = record(r, cid, state, questions)
            except (ValueError, TypeError, AttributeError) as e:
                f.write(json.dumps({"id": cid, "state": state, "questions": questions, "error": type(e).__name__},
                                   ensure_ascii=False) + "\n")
                valid = {}
                for qid, q in questions.items():
                    try:
                        r.s.build_pairs(state, {qid: q})
                        valid[qid] = q
                    except (ValueError, TypeError, AttributeError):
                        pass
                if not valid:
                    continue
                try:
                    rec = record(r, cid + "#valid", state, valid)
                except ValueError:
                    continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print("wrote %d records to %s (%.1f MB)" % (n, path, os.path.getsize(path) / 2**20))


if __name__ == "__main__":
    main()

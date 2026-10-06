"""Parity of the weightless arbiter ONNX export (and of the layout port) against the fp32 reference.

    uv run --with peft==0.19.1 --with transformers==4.57.6 \
        python -m ollaya_convert.families.arbiter.parity out/arbiter-4b --requests shared.jsonl \
        [--run RUN --base BASE] [--device cuda] [--report parity.json]

Per request of the shared set (`--requests` JSONL, else `llm_common.cases` with `--td` typed-decisions rows):
  1. reference rows (`ref.encode`: the training templates through transformers) vs the port (`layout.py` +
     `tokenizers` + the export's tokenizer.json and decision.json): token ids, last position and slots must
     be identical, and a request the reference rejects must be rejected by the port; a rejected request
     continues with the questions both accept on their own;
  2. raw 24-slot scores: the reference in fp32 (LoRA merged, TF32 off, one row at a time) vs ONNX Runtime on
     the CPU (LoRA unmerged, rows batched and right-padded);
  3. probabilities through the runtime contract (the row's slots, / temperature, softmax): same argmax.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import tokenizers

from ..llm_common.ort_rows import run_rows, session
from . import ref
from .check import requests
from .goldens import softmax
from .layout import ArbiterLayout, LayoutError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--requests", default=None, help="the shared request set as JSONL")
    ap.add_argument("--td", type=int, default=40, help="typed-decisions rows when --requests is not given")
    ap.add_argument("--run", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = all)")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    run, base = (a.run, a.base) if a.run and a.base else ref.snapshot("arbiter-4b")
    tok = ref.tokenizer(base)
    model, head = ref.load(run, base, device=a.device, merge=True)
    decision = json.load(open(os.path.join(a.model_dir, "decision.json")))
    T = json.load(open(os.path.join(a.model_dir, "calibration.json")))["temperature"][0]
    lay = ArbiterLayout(tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json")), decision)
    sess = session(os.path.join(a.model_dir, "model.onnx"), a.threads)
    pad = decision["special_tokens"]["pad"]

    st = defaultdict(list)
    n_q = agree = n_rows = mism = both_rej = 0
    rej_mismatch, worst = [], []
    case_list = requests(a.requests, a.td)
    for ci, (cid, state, questions) in enumerate(case_list):
        up_err = port_err = None
        try:
            ref_rows, meta = ref.encode(tok, state, questions)
        except ref.RequestError as e:
            up_err = str(e)
        try:
            rows, _ = lay.encode(state, questions)
        except LayoutError as e:
            port_err = str(e)
        if up_err or port_err:
            if up_err and port_err:
                both_rej += 1
            else:
                rej_mismatch.append((cid, up_err, port_err))
            valid = {}
            for qid, q in (questions.items() if isinstance(questions, dict) else []):
                try:
                    ref.encode(tok, state, {qid: q})
                    lay.encode(state, {qid: q})
                    valid[qid] = q
                except (ref.RequestError, LayoutError):
                    pass
            if not valid:
                continue
            ref_rows, meta = ref.encode(tok, state, valid)
            rows, _ = lay.encode(state, valid)
        for u, r in zip(ref_rows, rows):
            n_rows += 1
            mism += int(u != r)
        want = ref.forward(model, head, ref_rows)
        got, _ = run_rows(sess, [r["ids"] for r in rows],
                          lambda idx, L: {"last_pos": np.array([rows[i]["last_pos"] for i in idx], dtype=np.int64)},
                          pad, a.budget)
        for m, r, z_ref, z_got in zip(meta, rows, want, got):
            st["slots/" + m["type"]].append(float(np.abs(z_ref - z_got).max()))
            p_ref = np.array(softmax([z_ref[s] for s in r["slots"]], T))
            p_got = np.array(softmax([z_got[s] for s in r["slots"]], T))
            dp = float(np.abs(p_ref - p_got).max())
            st["prob/" + m["type"]].append(dp)
            same = int(p_ref.argmax() == p_got.argmax())
            n_q += 1
            agree += same
            worst.append((dp, cid, m["qid"], same))
        if (ci + 1) % 10 == 0:
            print("  %d/%d requests, %d questions" % (ci + 1, len(case_list), n_q), flush=True)

    summary = {
        "requests": len(case_list), "questions": n_q, "rows": n_rows, "row_mismatches": mism,
        "rejected_by_both": both_rej, "rejection_mismatches": rej_mismatch,
        "argmax_agreement_onnx_vs_fp32": agree / max(n_q, 1),
        "stats": {k: {"max": float(np.max(v)), "p99": float(np.quantile(v, 0.99)), "mean": float(np.mean(v))}
                  for k, v in sorted(st.items())},
        "worst": sorted(worst, reverse=True)[:8],
    }
    print(json.dumps(summary, indent=1))
    if a.report:
        with open(a.report, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()

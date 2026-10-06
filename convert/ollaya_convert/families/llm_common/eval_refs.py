"""Typed-decisions quality of the ONNX families (via their fp32 upstream references, which the exports
match to ~1e-5), in the same metrics as the llm-logits demo, for a like-for-like catalog comparison.

    uv run python -m ollaya_convert.families.llm_common.eval_refs decider decider-0.8b out/decider-0.8b --root ROOT
    KEV_SRC=... uv run --with peft==0.21.0 --with pydantic==2.12.5 \
        python -m ollaya_convert.families.llm_common.eval_refs kev kev-0.8b out/kev-0.8b --run RUN --base BASE

    CLM_SRC=... uv run --with requests python -m ollaya_convert.families.llm_common.eval_refs clm clm-8b out/clm-8b \
        --base BASE --head HEAD.pt

Writes <model_dir>/typed-decisions-quality.json (as shipped = the model's calibration.json, plus the
cross-fitted per-type temperatures) and typed-decisions-logits.jsonl. A question the model cannot answer (arbiter:
a score with other than 6 levels, a choice over 16 options) has no logits and is left out of the metrics;
"questions" and "answered" give the coverage.
"""
from __future__ import annotations

import argparse
import json
import os

import tokenizers

from . import cases, quality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["decider", "kev", "clm", "arbiter"])
    ap.add_argument("model")
    ap.add_argument("model_dir")
    ap.add_argument("--root", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--head", default=None, help="clm: the projection-head checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--td-limit", type=int, default=400)
    a = ap.parse_args()

    decision = json.load(open(os.path.join(a.model_dir, "decision.json")))
    calib = json.load(open(os.path.join(a.model_dir, "calibration.json")))["temperature"]
    tok = tokenizers.Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    if a.family == "decider":
        from ..decider import ref
        from ..decider.layout import DeciderLayout

        d = ref.load(a.root or ref.snapshot(a.model), device=a.device)
        lay = DeciderLayout(tok, decision)

        def scorer(state, questions):
            items, _ = ref.encode(d, state, questions)
            rows, plan = lay.encode(state, questions)
            logits = ref.forward(d, items)
            return {it["qid"]: lay.option_logits(it, logits) for it in plan}
    elif a.family == "clm":
        from ..clm import ref

        r = ref.Reference(a.base, a.head, device=a.device)

        def scorer(state, questions):
            try:
                return {qid: lg for qid, lg in r.answer(state, questions)[1].items()}
            except ValueError:   # a text over the model's 2,048 tokens: no answer, as the runtime
                return {}
    elif a.family == "arbiter":
        from ..arbiter import ref

        run, base = (a.run, a.base) if a.run and a.base else ref.snapshot(a.model)
        atok = ref.tokenizer(base)
        model, head = ref.load(run, base, device=a.device, merge=True)

        def scorer(state, questions):
            out = {}
            for qid, q in questions.items():   # one at a time: a request with one unanswerable question is a 422
                try:
                    rows, _ = ref.encode(atok, state, {qid: q})
                except ref.RequestError:
                    continue
                z = ref.forward(model, head, rows)[0]
                out[qid] = [z[i] for i in rows[0]["slots"]]
            return out
    else:
        from ..kev import ref

        run, base = (a.run, a.base) if a.run and a.base else ref.snapshot(a.model)
        ck, ktok, m = ref.load(run, base, device=a.device, merge=True)

        def scorer(state, questions):
            enc, meta = ref.encode(ktok, m, state, questions)
            return {q["id"]: s for q, s in zip(meta, ref.forward(m, enc))}

    rows = cases.typed_decisions_gold(a.td_limit)
    items = quality.collect(scorer, rows, progress=100)
    quality.dump(items, os.path.join(a.model_dir, "typed-decisions-logits.jsonl"))
    shipped = dict(zip(quality.TYPES, calib))
    report = {"model": a.model, "rows": len(rows), "questions": sum(len(c[2]) for c, _ in rows), "answered": len(items),
              "as_shipped": {"temperatures": shipped, "metrics": quality.metrics(items, shipped)}}
    report.update(quality.cross_fit(items))
    with open(os.path.join(a.model_dir, "typed-decisions-quality.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report["as_shipped"], indent=1))
    print(json.dumps({k: report[k]["metrics"]["all"] for k in ("fit_even_eval_odd", "fit_odd_eval_even")}, indent=1))


if __name__ == "__main__":
    main()

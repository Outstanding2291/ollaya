"""Typed-decisions quality of a model as Ollaya's Rust runtime runs it (any engine).

    python -m ollaya_convert.families.llm_common.quality_runtime requests out/td-requests.jsonl
    cargo run --release -p ollaya-runner --example logits -- <model-dir> out/td-requests.jsonl out/td-logits.jsonl cuda
    python -m ollaya_convert.families.llm_common.quality_runtime report <model-dir> out/td-logits.jsonl [--device cuda]

`requests` writes the 400 typed-decisions test rows as the runtime reads them; `report` joins the runtime's logits
(the order answers use: a choice's criteria order, noul [false, true], score levels) with the gold labels and writes
typed-decisions.json next to decision.json: accuracy against the gold label (T-independent), and cross-entropy and
ECE at the shipped temperatures (`calibration.json`), at T=1, and cross-fitted (`quality.cross_fit`), as
`quality_llama.py` does for the GGUF families.
"""
from __future__ import annotations

import argparse
import json
import os

from . import cases, quality


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("requests")
    r.add_argument("out")
    r.add_argument("--limit", type=int, default=0, help="typed-decisions rows (0: all 400)")
    p = sub.add_parser("report")
    p.add_argument("model_dir")
    p.add_argument("logits")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = ap.parse_args()

    rows = cases.typed_decisions_gold(a.limit)
    if a.cmd == "requests":
        with open(a.out, "w", encoding="utf-8") as f:
            for (cid, state, questions), _ in rows:
                f.write(json.dumps({"id": cid, "state": state, "questions": questions}, ensure_ascii=False) + "\n")
        print("wrote %d requests to %s" % (len(rows), a.out))
        return

    got = {}
    for line in open(a.logits, encoding="utf-8"):
        x = json.loads(line)
        got[x["id"]] = x.get("logits")
    missing = [cid for (cid, _, _), _ in rows if got.get(cid) is None]
    if missing:
        raise SystemExit("%d requests have no logits (rejected or not run), e.g. %s" % (len(missing), missing[:3]))
    order = iter([cid for (cid, _, _), _ in rows])   # collect calls the scorer once per row, in order
    items = quality.collect(lambda state, questions: got[next(order)], rows)
    t = json.load(open(os.path.join(a.model_dir, "calibration.json"), encoding="utf-8"))["temperature"]
    shipped = {"choice": t[0], "score": t[1], "noul": t[2]}
    report = {"rows": len(rows), "questions": len(items), "device": a.device, "engine": "ollaya-runner",
              "shipped_temperatures": shipped, "shipped": quality.metrics(items, shipped), **quality.cross_fit(items)}
    quality.dump(items, os.path.join(a.model_dir, "typed-decisions-logits.jsonl"))
    with open(os.path.join(a.model_dir, "typed-decisions.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    s = report["shipped"]
    print(json.dumps({"acc": s["all"]["acc"], "by_type": {k: s[k]["acc"] for k in quality.TYPES if k in s},
                      "ece_shipped": s["all"]["ece"], "ece_T1": report["raw_T1"]["all"]["ece"], "n": report["questions"]}))


if __name__ == "__main__":
    main()

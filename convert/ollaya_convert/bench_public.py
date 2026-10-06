"""Bespoke Labs' public decision benchmark, run against any TypeSafe-compatible `/v1/systemone` server.

The suite is Bespoke Labs' (https://github.com/bespokelabsai/nimble, `docs/PUBLIC_BENCHMARKS.md`):
13 subsets of 11 human-labeled public datasets, 3,880 records, one question each. Build it with
their converters and check every subset's `dataset_sha256` against their committed manifests; then

    NIMBLE_SRC=/path/to/nimble python -m ollaya_convert.bench_public run --url http://127.0.0.1:11435 \
        --model winnow:e4b --data DIR --out runs/ollaya/winnow-e4b
    NIMBLE_SRC=/path/to/nimble python -m ollaya_convert.bench_public report runs/*/* > report.md

Requests, scoring and metrics are Bespoke's own code (`nimble.evaluation.evaluate_public_jev`, the
runner they use for TypeSafe's hosted Jev): each request carries exactly the record's `state` and
`questions`, and the human label is joined afterwards. One change: their response check accepts
probabilities rounded to Jev's two decimals; Ollaya and Ollama round to four, so the sum-to-one
tolerance follows that grid. And Ollaya names the model that answered (`nli:latest`, or `laya:en` for the
`laya` router), which TypeSafe allows, so that name is accepted for the requested one. Requests run one at a time, so `elapsed_seconds` is the latency a
single client sees, HTTP included.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

SUBSETS = ["vitaminc-dev", "massive-en-US", "massive-de-DE", "boolq", "squad2", "paws", "multinli",
           "civil_comments", "aegis2", "helpsteer2", "summeval-relevance", "summeval-consistency", "pubmedqa"]


def _nimble():
    src = os.environ.get("NIMBLE_SRC")
    if not src:
        raise SystemExit("set NIMBLE_SRC to a checkout of https://github.com/bespokelabsai/nimble")
    if src not in sys.path:
        sys.path.insert(0, src)
    import nimble.datasets.dataset_io as dio
    import nimble.evaluation.evaluate_public_jev as jev

    return dio, jev


def _validate_4dp(dio):
    """dataset_io.validate_teacher with the rounding grid of a server that rounds to four decimals."""
    original = dio.validate_teacher

    def validate(row, response, requested_model):
        # Ollaya answers with the canonical name of the model that answered (`nli:latest` for `nli`, the
        # route target for a router), which TypeSafe allows; the check is about which model answered.
        got = response.get("model", "")
        if got in (requested_model + ":latest",) or (requested_model == "laya" and got.startswith("laya:")):
            response = {**response, "model": requested_model}
        answer = response["answers"].get("decision", {})
        probs = answer.get("probabilities")
        if isinstance(probs, dict) and probs and all(
                isinstance(p, (int, float)) and math.isclose(p, round(p, 4), abs_tol=1e-10) for p in probs.values()):
            total = sum(probs.values())
            if math.isclose(total, 1.0, abs_tol=len(probs) * 0.00005 + 1e-8):
                response = {**response, "answers": {**response["answers"], "decision": {
                    **answer, "probabilities": {k: v / total for k, v in probs.items()}}}}
        return original(row, response, requested_model)

    return validate


def transport(url, timeout=600):
    def call(payload):
        req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(payload).encode(),
                                     headers={"Authorization": "Bearer local", "Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            from nimble.evaluation.evaluate_public_jev import ApiError

            body = e.read()[:300].decode(errors="replace")
            print("HTTP %d: %s" % (e.code, body), file=sys.stderr, flush=True)
            raise ApiError(e.code) from None
    return call


def run(a):
    dio, jev = _nimble()
    jev.validate_teacher = _validate_4dp(dio)
    jev.API_URL = a.url
    subsets = a.subsets or SUBSETS
    # One untimed request first, so the model load is not in the first record's latency.
    first = json.loads(open(Path(a.data) / subsets[0] / "all.jsonl", encoding="utf-8").readline())
    transport(a.url)(dio.request_for(first, a.model))
    for subset in subsets:
        data = Path(a.data) / subset / "all.jsonl"
        out = Path(a.out) / subset
        report = jev.run(data, out, model=a.model, concurrency=1, transport=transport(a.url), attempts=2)
        s = report["summary"]["all"]
        print("%-22s %-28s acc %.3f  ece %.3f  median %.0f ms  errors %d" % (
            subset, a.model, s["accuracy"], report["calibration"]["ece10"] or float("nan"),
            1000 * report["timing"]["median_seconds"], report["errors"]), flush=True)


def _ece(pairs, bins=10):
    pairs = list(pairs)
    total, err = len(pairs), 0.0
    for b in range(bins):
        sel = [(p, c) for p, c in pairs if (b / bins < p <= (b + 1) / bins) or (b == 0 and p == 0)]
        if sel:
            err += len(sel) / total * abs(sum(c for _, c in sel) / len(sel) - sum(p for p, _ in sel) / len(sel))
    return err


def summarize(run_dir):
    """One run (a model on a server): per subset and pooled accuracy, ECE, Brier, latency."""
    per, rows = {}, []
    for subset in SUBSETS:
        path = Path(run_dir) / subset / "summary.json"
        if not path.exists():
            continue
        s = json.load(open(path, encoding="utf-8"))
        with open(Path(run_dir) / subset / "rows.jsonl", encoding="utf-8") as f:
            sub_rows = [json.loads(line) for line in f if line.strip()]
        rows += sub_rows
        per[subset] = {"n": s["count"], "errors": s["errors"], "accuracy": s["summary"]["all"]["accuracy"],
                       "ece10": s["calibration"]["ece10"], "type": sub_rows[0]["type"]}
    valid = [r for r in rows if "error" not in r]
    times = sorted(r["elapsed_seconds"] for r in rows)
    by_type = {}
    for t in ("choice", "noul", "score"):
        subs = [v["accuracy"] for v in per.values() if v["type"] == t]
        by_type[t] = statistics.mean(subs) if subs else None
    return {
        "run": str(run_dir), "subsets": len(per), "records": len(rows), "errors": len(rows) - len(valid),
        "macro_accuracy": statistics.mean(v["accuracy"] for v in per.values()) if per else None,
        "micro_accuracy": sum(r["student"]["correct"] for r in rows) / len(rows) if rows else None,
        "macro_ece10": statistics.mean(v["ece10"] for v in per.values() if v["ece10"] is not None) if per else None,
        "pooled_ece10": _ece((r["student"]["top_probability"], r["student"]["correct"]) for r in valid) if valid else None,
        "mean_brier": statistics.mean(r["student"]["multiclass_brier"] for r in valid if "multiclass_brier" in r["student"]) if valid else None,
        "median_ms": 1000 * statistics.median(times) if times else None,
        "p95_ms": 1000 * times[math.ceil(0.95 * len(times)) - 1] if times else None,
        "by_type": by_type, "per_subset": per,
    }


def report(a):
    _nimble()
    out = [summarize(d) for d in a.runs]
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=2)
    print("| Run | Records | Errors | Macro acc | Micro acc | Choice | Noul | Score | ECE (macro) | ECE (pooled) | Brier | Median | p95 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    f = lambda x, d=3: "n/a" if x is None else ("%.*f" % (d, x))  # noqa: E731
    for r in sorted(out, key=lambda r: -(r["macro_accuracy"] or 0)):
        print("| %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s ms | %s ms |" % (
            r["run"], r["records"], r["errors"], f(r["macro_accuracy"]), f(r["micro_accuracy"]),
            f(r["by_type"]["choice"]), f(r["by_type"]["noul"]), f(r["by_type"]["score"]),
            f(r["macro_ece10"]), f(r["pooled_ece10"]), f(r["mean_brier"]), f(r["median_ms"], 0), f(r["p95_ms"], 0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--data", required=True, help="directory holding <subset>/all.jsonl")
    r.add_argument("--out", required=True)
    r.add_argument("--subsets", nargs="*")
    p = sub.add_parser("report")
    p.add_argument("runs", nargs="+")
    p.add_argument("--json")
    a = ap.parse_args()
    (run if a.cmd == "run" else report)(a)


if __name__ == "__main__":
    main()

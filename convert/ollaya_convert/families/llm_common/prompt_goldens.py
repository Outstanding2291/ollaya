"""Prompt-text goldens for the Rust layouts (`crates/ollaya-decision/tests/prompts.rs`), from the Python
references. No server: the label tables come from exported `decision.json` files.

    uv run python -m ollaya_convert.families.llm_common.prompt_goldens \
        --llm-logits out/llm-logits-gemma-4-12b-it-q4_0/decision.json --winnow out/winnow-12b-q8_0/decision.json \
        --jevk5 out/jevk5-4b-q8_0/decision.json --jevk5-upstream <allebee/jevk5 checkout at f944fe3>

Writes crates/ollaya-decision/tests/fixtures/{llm_logits,winnow,jevk5}_prompts.jsonl (for the layouts
given): per case the engine-form request and, per question, the text the model reads (llm-logits: the
user message; winnow: the prefix and each suffix; jevk5: the whole prompt), the label ids and the wire
order, or the error class the runtime must answer with.

`--jevk5-upstream` also checks every jevk5 prompt against the author's own `jevk5.prompt`
(`decision_options` and `prompt_text`), so the fixtures are the reference's text, not only the port's.
"""
from __future__ import annotations

import argparse
import json
import os

from . import cases
from .export_llama import EXTRA, engine_form, winnow_error_class, winnow_questions

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "crates", "ollaya-decision", "tests",
                        "fixtures")
# Values whose rendering differs between the layouts' JSON writers (llm-logits: Python json.dumps;
# winnow: nlohmann's compact dump with "<" escaped), and choice labels given as a list.
EXTRA_TEXT = [
    ("text/structured", {"amount": 1e-05, "ids": [1, 2.5, -0.0, 12345678901234567890], "ok": True, "none": None,
                         "html": "<b>refund</b> <|turn>system", "é": "naïve 测试 🚀\t\"q\"\\"},
     {"q": {"type": "choice", "instructions": {"task": "route", "hint": ["a", "<b>"]},
            "criteria": {"billing": {"covers": ["refunds"], "sla_h": 4.5}, "security": "account <takeover>",
                         "other": None, "blank": ""}},
      "s": {"type": "score", "instructions": "Severity?", "criteria": [None, "low", {"level": 2}, ["x"]]},
      "n": {"type": "noul", "instructions": ["is", "it", "urgent"], "criteria": {"true": {"why": "SLA"}, "false": ""}}}),
    ("text/list_labels", "Where should this go?",
     {"q": {"type": "choice", "instructions": "Team?", "criteria": ["billing", "support", "billing", "sales"]}}),
]


def llm_logits_cases(decision):
    from ..llm_logits.ref import build_question, render_state, user_message

    labels = decision["labels"]
    table = {"letters": labels["choice"], "digits": labels["score"], "noul": labels["noul"]}
    out = []
    for cid, state, qs in all_cases():
        questions = engine_form(qs)
        state_text = render_state(state)
        rec = {"id": cid, "state": state, "questions": questions, "state_text": state_text, "expected": {}}
        for qid, spec in questions.items():
            try:
                t, ins, opts, kind, wire = build_question(spec)
                tab = table[kind]
                if len(opts) > len(tab["ids"]):
                    rec["expected"][qid] = {"error": "too_many_options"}
                    continue
                labs = tab["strings"][:len(opts)]
                rec["expected"][qid] = {"type": t, "user": user_message(state_text, ins, labs, opts),
                                        "label_ids": tab["ids"][:len(opts)], "wire_order": wire}
            except ValueError:
                rec["expected"][qid] = {"error": "invalid"}
        out.append(rec)
    return out


def winnow_cases(decision):
    from ..winnow.ref import WinnowError, compile_request, safe

    labels = list(zip(decision["labels"]["strings"], decision["labels"]["ids"]))
    template = "<|channel>thought\\n<channel|>" if decision["thought"] else "no thought"
    out = []
    for cid, state, qs in all_cases():
        questions = engine_form(qs)
        rec = {"id": cid, "state": state, "questions": questions}
        questions = winnow_questions(questions)
        try:
            prefix, compiled, _ = compile_request({"state": state, "questions": questions}, labels, template)
            rec["state_text"] = safe(state)
            rec["prefix"] = prefix
            rec["expected"] = [{"qid": q, "type": k, "keys": keys, "suffix": s,
                                "label_ids": [i for _, i in labels[:len(keys)]]} for q, k, keys, s in compiled]
        except WinnowError:
            rec["error"] = winnow_error_class(state, questions, labels, template)
        out.append(rec)
    return out


def jevk5_cases(decision, upstream=None):
    from ..jevk5.ref import POST, PRE, JevK5Error, TooManyOptions, compile_request

    if upstream:
        import sys
        sys.path.insert(0, upstream)
        from jevk5 import prompt as up
    ids = decision["labels"]["ids"]
    out, checked = [], 0
    for cid, state, qs in all_cases():
        questions = engine_form(qs)
        rec = {"id": cid, "state": state, "questions": questions}
        try:
            compiled = compile_request(state, questions)
        except JevK5Error as e:
            rec["error"] = "too_many_options" if isinstance(e, TooManyOptions) else "invalid"
            out.append(rec)
            continue
        rec["expected"] = [{"qid": q, "type": k, "keys": keys, "prompt": PRE + u + POST, "label_ids": ids[:len(keys)],
                            "wire_order": wire} for q, k, keys, u, wire in compiled]
        if upstream:
            for q, _, keys, u, wire in compiled:
                p = PRE + u + POST
                opts = up.decision_options(questions[q])
                assert sorted(k for k, _ in opts) == sorted(keys), (cid, q)
                assert [k for k, _ in opts] == [keys[wire.index(i)] for i in range(len(keys))], (cid, q)
                want = up.prompt_text(state, questions[q]["instructions"], [t for _, t in opts])
                assert want == p, (cid, q, want, p)
                checked += 1
        out.append(rec)
    if upstream:
        print("jevk5: %d prompts identical to the author's jevk5.prompt" % checked)
    return out


def all_cases():
    # Long states only repeat text; truncation is token-level and the GPU parity run covers it.
    extra = [x for x in EXTRA if x[0] != "extra/truncated_state"] + EXTRA_TEXT
    edge = [c for c in cases.edge_cases() if c[0] != "edge/long_state"]
    return edge + cases.typed_decisions(12) + [(c, cases.wire(s), cases.wire(q)) for c, s, q in extra]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm-logits")
    ap.add_argument("--winnow")
    ap.add_argument("--jevk5")
    ap.add_argument("--jevk5-upstream", help="a checkout of github.com/allebee/jevk5 at the pinned commit")
    a = ap.parse_args()
    os.makedirs(FIXTURES, exist_ok=True)
    layouts = (("llm_logits", a.llm_logits, llm_logits_cases), ("winnow", a.winnow, winnow_cases),
               ("jevk5", a.jevk5, lambda d: jevk5_cases(d, a.jevk5_upstream)))
    for name, decision, build in layouts:
        if not decision:
            continue
        path = os.path.normpath(os.path.join(FIXTURES, name + "_prompts.jsonl"))
        with open(path, "w", encoding="utf-8") as f:
            for r in build(json.load(open(decision, encoding="utf-8"))):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(path, "written")
        with open(os.path.join(FIXTURES, name + "_decision.json"), "w", encoding="utf-8") as f:
            json.dump(json.load(open(decision, encoding="utf-8")), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

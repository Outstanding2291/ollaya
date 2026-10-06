"""Goldens for `snap-v1`, with snap's own prompts as the reference.

    python -m ollaya_convert.families.snap.goldens <requests.jsonl> <snap-export.jsonl> \\
        --server <b11146>/llama-server --gguf snap1-2b-q8_0.gguf --repo logitlab/snap1-2b-GGUF \\
        --revision <commit> --file snap1-2b-q8_0.gguf --slug snap1-2b-q8_0 [--device CUDA0|cpu]

snap is Rust, so there is no Python reference: the prompt token ids come from `snap export-prompts`
(snap v0.5.0, `PROMPT_VERSION` 6) on the shared request set, one case per question, as the author
posted it on issue #46 (`fixture.py` checks the port's text against the same export). This writes
`out/snap-v1-<slug>/` with `decision.json` (the chat template around the user message, checked
against this GGUF's own template, and the letters' tokens), `calibration.json` (temperature 1.0:
snap1-2b ships raw probabilities) and `goldens-<device>.jsonl`: the author's ids evaluated on the
pinned llama-server with the fixed plan (`llm_common.plan`), one cold pass per question.

Every case must be what the daemon hands the engine (`wire.engine_form`) unchanged; a question the
API rejects before any engine sees it is left out. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os

from ..llm_common.export_llama import (LLAMA_BUILD, OUT, device_class, engine_form, gguf_metadata, post_to,
                                       server_version, sha256_file)
from ..llm_common.llama_server import LlamaServer
from ..llm_common.plan import FixedPlan, server_args
from .fixture import split

LAYOUT = "snap-v1"
SYSTEM = ("You are a decision engine. Given a state and a question, you evaluate the options and reply with "
          "only the letter of the best option. Never explain.")
LETTERS = [chr(65 + i) for i in range(26)]
REPLY = "\n\nReply with one letter only."
UPSTREAM = {"repo": "https://github.com/emnlmn/snap", "tag": "v0.5.0",
            "commit": "5e6a65ccef36cc9e0cd44653f52cbb83c73bb3fb", "prompt_version": 6}


def state_text(user, layout, letters):
    """The rendered state inside snap's user message. The option lines are known from the export,
    so the question block's end is found by walking them rather than by searching the state."""
    def options_end(start):
        pos = start
        for i, s in enumerate(letters):
            line = "%s) %s" % (s["letter"], s["text"])
            if not user.startswith(line, pos):
                return None
            pos += len(line) + (1 if i < len(letters) - 1 else 0)
        return pos

    if layout == "question_first":
        at = user.find("\nOPTIONS\n")
        while at >= 0:
            end = options_end(at + len("\nOPTIONS\n"))
            if end is not None and user.startswith("\n\nSTATE\n", end) and user.endswith(REPLY):
                return user[end + len("\n\nSTATE\n"):len(user) - len(REPLY)]
            at = user.find("\nOPTIONS\n", at + 1)
    elif layout == "state_first":
        at = user.rfind("\n\nQUESTION\n")
        while at >= 0:
            opts = user.find("\nOPTIONS\n", at)
            end = options_end(opts + len("\nOPTIONS\n")) if opts >= 0 else None
            if user.startswith("STATE\n") and end == len(user) - len(REPLY):
                return user[len("STATE\n"):at]
            at = user.rfind("\n\nQUESTION\n", 0, at)
    raise SystemExit("cannot find the state in a %s message" % layout)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("requests")
    ap.add_argument("export")
    ap.add_argument("--server", required=True, help="the pinned llama-server build the runtime ships")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--n-ctx", type=int, default=8192, help="snap serve's default context")
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--device", default="CUDA0", help="llama.cpp device (CUDA0, MTL0), or cpu")
    a = ap.parse_args()

    requests = {r["id"]: r for r in map(json.loads, open(a.requests, encoding="utf-8"))}
    exported = [json.loads(l) for l in open(a.export, encoding="utf-8")]
    frames = {split(r["prompt"])[0::2] for r in exported if "prompt" in r and "expand" not in r}
    if len(frames) != 1:
        raise SystemExit("expected one template frame in the export, found %d" % len(frames))
    (pre, post), = frames

    out = os.path.abspath(os.path.join(OUT, "%s-%s" % (LAYOUT, a.slug)))
    os.makedirs(out, exist_ok=True)
    meta = gguf_metadata(a.gguf, ["general.architecture", "sliding_window"])
    swa = any(k.endswith(".attention.sliding_window") for k in meta)
    dev = None if a.device == "cpu" else a.device
    argv = server_args(a.n_ctx, swa, dev)
    srv = LlamaServer.start(a.server, a.gguf, port=a.port, argv=argv,
                            log=os.path.join(out, "llama-server-%s.log" % device_class(a.device)), timeout=1800)
    try:
        tok = lambda t, special: srv.tokenize(t, add_special=False, parse_special=special)  # noqa: E731
        labels = []
        for s in LETTERS:
            ids = tok(s, False)
            if len(ids) != 1 or srv.pieces(ids)[0] != s:
                raise SystemExit("label %r is not one token in this GGUF: %s" % (s, ids))
            labels.append(ids[0])
        tpl = srv.apply_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "USER"}],
                                 enable_thinking=False)
        if tpl != pre + "USER" + post:
            raise SystemExit("this GGUF's chat template does not give snap's frame:\n%r\n%r" % (tpl, pre + "USER" + post))
        sha = sha256_file(a.gguf)
        props = srv.props()
        decision = {
            "engine": "llama", "family": "snap", "layout": LAYOUT,
            "template": {"pre": pre, "post": post},
            "labels": {"strings": LETTERS, "ids": labels}, "upstream": UPSTREAM,
            "gguf": {"repo": a.repo, "revision": a.revision, "path": a.file, "sha256": sha,
                     "size": os.path.getsize(a.gguf), "quantization": props.get("model_ftype", ""),
                     "architecture": meta.get("general.architecture", ""),
                     "url": "https://huggingface.co/%s/resolve/%s/%s" % (a.repo, a.revision, a.file)},
            "llama": {"n_ctx": a.n_ctx, "swa_full": swa, "plan": "cold", "build": LLAMA_BUILD},
        }
        with open(os.path.join(out, "decision.json"), "w", encoding="utf-8") as f:
            json.dump(decision, f, indent=1, ensure_ascii=False)
        with open(os.path.join(out, "calibration.json"), "w", encoding="utf-8") as f:
            json.dump({"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
                       "source": "uncalibrated (temperature 1.0): snap1-2b's card reports raw probabilities"}, f, indent=2)

        plan = FixedPlan(post_to(srv.url))
        name = "goldens-%s" % device_class(a.device)
        with open(os.path.join(out, name + ".meta.json"), "w", encoding="utf-8") as f:
            json.dump({"server": server_version(a.server), "device": a.device, "args": argv, "gguf_sha256": sha,
                       "reference": "snap export-prompts (snap %s, prompt v%d) on the shared request set, "
                       "evaluated with llm_common.plan (the fixed evaluation plan)"
                       % (UPSTREAM["tag"], UPSTREAM["prompt_version"])}, f, indent=1)
        n_rows = left_out = 0
        seen = set()
        with open(os.path.join(out, name + ".jsonl"), "w", encoding="utf-8") as f:
            for r in exported:
                rid, _, qid = r["id"].partition("#")
                qid = qid.split("#")[0]
                if (rid, qid) in seen:
                    continue  # the further probes of one expanded choice
                seen.add((rid, qid))
                req = requests[rid]
                questions = engine_form({qid: req["questions"][qid]})
                if qid not in questions:
                    left_out += 1  # the API rejects it before any engine sees it
                    continue
                if questions[qid] != req["questions"][qid]:
                    raise SystemExit("%s: the engine form differs from what snap read" % r["id"])
                rec = {"id": "%s#%s" % (rid, qid), "state": req["state"], "questions": questions}
                if "expand" in r:
                    rec["error"] = "too_many_options"
                elif "error" in r:
                    rec["error"], rec["message"] = "invalid", r["error"]
                else:
                    ids = r["token_ids"]
                    _, user, _ = split(r["prompt"])
                    if tok(pre, True) + tok(user, False) + tok(post, True) != ids:
                        raise SystemExit("%s: this GGUF tokenizes snap's prompt differently" % r["id"])
                    n = len(r["letters"])
                    keys = [s["key"] for s in r["letters"]]
                    noul = req["questions"][qid]["type"] == "noul"
                    if noul and keys != ["yes", "no"]:
                        raise SystemExit("%s: noul keys %s" % (r["id"], keys))
                    wire = [1, 0] if noul else list(range(n))
                    lp = plan.ask(ids, 0, labels[:n])
                    state = state_text(user, r["layout"], r["letters"])
                    rec.update({"state_tokens": len(tok(state, False)), "state_truncated": False,
                                "rows": [{"qid": qid, "ids": ids, "P": 0, "candidates": labels[:n],
                                          "wire_order": wire, "option_logits": [lp[j] for j in wire]}]})
                    n_rows += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("%s/%s.jsonl: %d questions, %d left out (the API rejects them)" % (out, name, n_rows, left_out))
    finally:
        srv.stop()


if __name__ == "__main__":
    main()

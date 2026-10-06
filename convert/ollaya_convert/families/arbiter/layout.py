"""`arbiter-fixed-v1`: the request -> token rows layout of hiteshluke/arbiter-4b, written the way the Rust
port is (`crates/ollaya-decision/src/arbiter.rs`: HF `tokenizers` and `decision.json` only). `check.py`
compares it id for id with the reference prompt in `ref.py` (the training script's templates, tokenized by
transformers as in training).

One causal row per question, the training prompt with Gemma's `<bos>` in front:

    ids = [bos] + tok("State: {state}\\n\\nQuestion: {instructions}\\n\\nOptions:\\n{block}\\n\\nAnswer:")
    last_pos = len(ids) - 1                     # where the 24-slot head is read

    block   noul    "T. Yes / True\\nF. No / False"
            choice  "A. {option 0}\\nB. {option 1}\\n..."     (1..16 options, letters A..P)
            score   "0\\n1\\n2\\n3\\n4\\n5"                   (exactly 6 levels)

`state` and `instructions` are `render(...)` of the JSON values; a choice option is `name` or
`name: render(description)`. The noul descriptions and the score level descriptions are not part of the
prompt: the model was trained on the fixed blocks above.

The graph returns 24 raw slot scores per row. A question's option logits are the scores at its slots, in
Ollaya's option order:

    noul    [1, 0]           false = F (slot 1), true = T (slot 0)
    choice  [2, ..., 1 + k]  option j = letter j (A..P)
    score   [18, ..., 23]    level j = digit j (0..5)

Requests the fixed head cannot answer are rejected (HTTP 422), never truncated: a choice with more than 16
options, a score with other than 6 levels, a row over `max_row_tokens`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
MAX_CHOICE = len(CHOICE_LETTERS)
SCORE_LEVELS = 6

# Slot order of the head (`head_meta.json` "verbalizer"). Slot 1 is the F of T/F, slot 7 the F of A..P.
VERBALIZERS: List[str] = ["T", "F"] + CHOICE_LETTERS + [str(i) for i in range(SCORE_LEVELS)]
NUM_SLOTS = len(VERBALIZERS)                            # 24
NOUL_SLOTS = [1, 0]                                     # Ollaya's noul order: [false, true]
CHOICE_SLOTS = list(range(2, 2 + MAX_CHOICE))           # 2..17
SCORE_SLOTS = list(range(2 + MAX_CHOICE, NUM_SLOTS))    # 18..23

NOUL_BLOCK = "T. Yes / True\nF. No / False"
SCORE_BLOCK = "\n".join(str(i) for i in range(SCORE_LEVELS))


class LayoutError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


def render(v, indent: int = 0) -> str:
    """None -> '', scalars -> Python str() (True, 1.0, 1e-05), lists -> '- item' lines, dicts -> 'k: v'
    lines, two spaces per nesting level."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join("%s- %s" % (pad, render(x, indent + 1).lstrip()) for x in v)
    return "\n".join(
        "%s%s:\n%s" % (pad, k, render(x, indent + 1)) if isinstance(x, (dict, list))
        else "%s%s: %s" % (pad, k, render(x))
        for k, x in v.items()
    )


def option_text(name: str, desc) -> str:
    return name if desc is None or desc == "" else "%s: %s" % (name, render(desc))


def to_record(state, questions: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str, List[str]]], List[Dict[str, Any]]]:
    """Validate the request and render its texts: -> (state_text, [(qtype, instructions, options)], meta).
    `options` are the texts a choice lists (empty for noul and score, whose blocks are fixed); meta[q]["k"]
    is the number of option logits."""
    if not isinstance(questions, dict) or not questions:
        raise LayoutError("questions must contain at least one question")
    qs, meta = [], []
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise LayoutError("question %r must be an object" % qid)
        t = q.get("type")
        if "instructions" not in q:
            raise LayoutError("question %r: no 'instructions'" % qid)
        crit = q.get("criteria")
        if t == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise LayoutError("question %r: noul criteria must be an object" % qid)
            opts, k = [], 2
        elif t == "choice":
            if not isinstance(crit, dict) or not crit:
                raise LayoutError("question %r: choice criteria must be an object of label -> description" % qid)
            if len(crit) > MAX_CHOICE:
                raise LayoutError("question %r: %d options; this model answers at most %d" % (qid, len(crit), MAX_CHOICE))
            opts = [option_text(name, d) for name, d in crit.items()]
            k = len(opts)
        elif t == "score":
            if not isinstance(crit, list) or len(crit) != SCORE_LEVELS:
                raise LayoutError("question %r: score criteria must be a list of exactly %d levels" % (qid, SCORE_LEVELS))
            opts, k = [], SCORE_LEVELS
        else:
            raise LayoutError("question %r: unknown type %r" % (qid, t))
        qs.append((t, render(q.get("instructions")), opts))
        meta.append({"qid": qid, "type": t, "k": k})
    return render(state), qs, meta


def options_block(qtype: str, options: List[str]) -> str:
    if qtype == "noul":
        return NOUL_BLOCK
    if qtype == "score":
        return SCORE_BLOCK
    return "\n".join("%s. %s" % (CHOICE_LETTERS[i], o) for i, o in enumerate(options))


def prompt_text(state_text: str, instructions: str, block: str) -> str:
    return "State: %s\n\nQuestion: %s\n\nOptions:\n%s\n\nAnswer:" % (state_text, instructions, block)


def decision_fields(bos: int, pad: int, max_row_tokens: int) -> Dict[str, Any]:
    """The fields of decision.json this layout reads (export.py writes them; check.py builds them)."""
    return {
        "max_row_tokens": max_row_tokens,
        "special_tokens": {"bos": bos, "pad": pad},
        "num_slots": NUM_SLOTS,
        "slots": {"noul": list(NOUL_SLOTS), "choice": list(CHOICE_SLOTS), "score": list(SCORE_SLOTS)},
    }


class ArbiterLayout:
    """Encode a request into one causal row per question."""

    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        # The runtime loads tokenizer.json without its truncation and padding settings; so does the port.
        if hasattr(tokenizer, "no_truncation"):
            tokenizer.no_truncation()
            tokenizer.no_padding()
        self.bos = decision["special_tokens"]["bos"]
        self.max_row = decision["max_row_tokens"]
        sl = decision["slots"]
        self.slots = {"noul": sl["noul"], "choice": sl["choice"], "score": sl["score"]}

    def _encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode(self, state, questions):
        """-> (rows, meta): rows[q] = {"ids", "last_pos", "slots"}, one row per question in request order."""
        state_text, qs, meta = to_record(state, questions)
        rows = []
        for (t, instr, opts), m in zip(qs, meta):
            ids = [self.bos] + self._encode(prompt_text(state_text, instr, options_block(t, opts)))
            if len(ids) > self.max_row:
                raise LayoutError("question %r: the row is %d tokens; this model reads up to %d"
                                  % (m["qid"], len(ids), self.max_row))
            rows.append({"ids": ids, "last_pos": len(ids) - 1, "slots": self.slots[t][:m["k"]]})
        return rows, meta

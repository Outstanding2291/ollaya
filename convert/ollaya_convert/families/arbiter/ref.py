"""The PyTorch reference for hiteshluke/arbiter-4b in fp32: the reference prompt and the reference model.

Arbiter is a LoRA (r 16, alpha 32) on `unsloth/gemma-3-4b-it` plus a 24-slot head, `nn.Linear(2560, 24,
bias=False)`, read at the last position of the prompt's final hidden state (after Gemma's final norm).
The training script is `training_v3_6/train.py` in https://github.com/CodekinsTech/arbiter (pinned below);
the model repository holds the adapter, `head.pt` (the EMA-averaged head, BF16) and `head_meta.json`.

**Reference prompt.** The three templates below are copied verbatim from the training script, and a prompt
is tokenized the way training tokenizes it: transformers' tokenizer of the base repository with its
defaults, which put `<bos>` in front. The training data was text, so the request -> (state, question,
options) texts are Ollaya's (`layout.to_record`: `render` and the option texts), shared with the port.
The port (`layout.ArbiterLayout`) instead fills the template with plain string formatting and tokenizes
with HF `tokenizers`, as the Rust runtime does; `check.py` compares the two id for id.

**Reference model.** transformers' `Gemma3ForConditionalGeneration` in fp32 (eager attention, TF32 off), the
adapter through peft (merged for the goldens), one unpadded row at a time: the readout is
`hidden_states[-1][0, -1]`, as in the repository's benchmark script (`training_v3_6/bench.py`).

    tok = ref.tokenizer(base)
    rows, meta = ref.encode(tok, state, questions)       # reference prompt rows
    model, head = ref.load(run, base, device="cuda")
    scores = ref.forward(model, head, rows)               # one [24] array of raw slot scores per row
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from .layout import LayoutError, NUM_SLOTS, VERBALIZERS, to_record

ARBITER_GIT = {"repo": "https://github.com/CodekinsTech/arbiter", "commit": "e1cb30fd3019cf91d2631a2d1004abd858b8e4fa",
               "script": "training_v3_6/train.py"}
MODELS = {
    "arbiter-4b": {
        "repo": "hiteshluke/arbiter-4b",
        "revision": "0c44271c59f89758e3cae17b032e98a9140093e9",
        "base": "unsloth/gemma-3-4b-it",
        "base_revision": "bf46152c47f5dd20b896357cb51abc4c03b8ee8c",
        "base_files": ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
    },
}
# Ollaya's limit per row (a longer row is rejected). Training truncated prompts at 768 tokens instead.
MAX_ROW_TOKENS = 8192


class RequestError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


# ---- verbatim from training_v3_6/train.py (curate_data) ----
def _make_choice_prompt(state, question, options):
    letters = [chr(ord('A') + i) for i in range(len(options))]
    opt_lines = '\n'.join(f'{L}. {o}' for L, o in zip(letters, options))
    return f'State: {state}\n\nQuestion: {question}\n\nOptions:\n{opt_lines}\n\nAnswer:'


def _make_noul_prompt(state, question):
    return (f'State: {state}\n\nQuestion: {question}\n\n'
            f'Options:\nT. Yes / True\nF. No / False\n\nAnswer:')


def _make_score_prompt(state, question):
    return (f'State: {state}\n\nQuestion: {question}\n\n'
            f'Options:\n0\n1\n2\n3\n4\n5\n\nAnswer:')


NOUL_SLOTS = [0, 1]                       # T, F
def choice_slots(n): return list(range(2, 2 + n))
SCORE_SLOTS = list(range(18, 24))
# ---- end of the training script's definitions ----


def slots(qtype: str, k: int) -> List[int]:
    """The head slots of a question's option logits, in Ollaya's option order (noul: false, true)."""
    if qtype == "noul":
        return [NOUL_SLOTS[1], NOUL_SLOTS[0]]
    if qtype == "choice":
        return choice_slots(k)
    return list(SCORE_SLOTS)


def snapshot(slug: str):
    """Local snapshots of the checkpoint and of the base at their pinned revisions."""
    from huggingface_hub import snapshot_download

    m = MODELS[slug]
    run = os.environ.get("ARBITER_RUN") or snapshot_download(m["repo"], revision=m["revision"])
    base = os.environ.get("ARBITER_BASE") or snapshot_download(
        m["base"], revision=m["base_revision"],
        allow_patterns=["*.json", "*.jinja", "tokenizer.model"] + m["base_files"])
    return run, base


def tokenizer(base_dir: str):
    """The tokenizer as training loads it (`AutoTokenizer.from_pretrained(BASE_MODEL)`)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def encode(tok, state, questions):
    """-> (rows, meta): rows[q] = {"ids", "last_pos", "slots"} from the training templates."""
    try:
        state_text, qs, meta = to_record(state, questions)
    except LayoutError as e:
        raise RequestError(str(e)) from e
    rows = []
    for (t, instructions, options), m in zip(qs, meta):
        if t == "noul":
            prompt = _make_noul_prompt(state_text, instructions)
        elif t == "choice":
            prompt = _make_choice_prompt(state_text, instructions, options)
        else:
            prompt = _make_score_prompt(state_text, instructions)
        ids = list(tok(prompt)["input_ids"])
        if len(ids) > MAX_ROW_TOKENS:
            raise RequestError("question %r: the row is %d tokens; this model reads up to %d"
                               % (m["qid"], len(ids), MAX_ROW_TOKENS))
        rows.append({"ids": ids, "last_pos": len(ids) - 1, "slots": slots(t, m["k"])})
    return rows, meta


def head_meta(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "head_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("num_slots") != NUM_SLOTS or meta.get("verbalizer") != VERBALIZERS:
        raise SystemExit("head_meta.json: %d slots %r, expected %d slots %r"
                         % (meta.get("num_slots", 0), meta.get("verbalizer"), NUM_SLOTS, VERBALIZERS))
    return meta


def text_model(conditional_generation):
    """The Gemma 3 text decoder (`Gemma3TextModel`) inside `Gemma3ForConditionalGeneration`."""
    inner = getattr(conditional_generation, "model", None)
    lm = getattr(inner, "language_model", None)
    if lm is None:
        raise SystemExit("unexpected Gemma 3 layout in this transformers version: no model.language_model")
    return lm


def load(run_dir: str, base_dir: str, device: str = "cpu", merge: bool = True):
    """Gemma 3 4B IT + the adapter (peft) + the 24-slot head, all fp32. Returns (model, head); `model` is the
    peft model (merged into a plain `Gemma3ForConditionalGeneration` when `merge`)."""
    import torch
    from peft import PeftModel
    from transformers import Gemma3ForConditionalGeneration

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    head_meta(run_dir)
    base = Gemma3ForConditionalGeneration.from_pretrained(base_dir, torch_dtype=torch.float32,
                                                          attn_implementation="eager")
    model = PeftModel.from_pretrained(base, run_dir)
    if merge:
        model = model.merge_and_unload()
    model.to(device).eval()

    head = torch.nn.Linear(base.config.text_config.hidden_size, NUM_SLOTS, bias=False)   # 2560
    state = torch.load(os.path.join(run_dir, "head.pt"), map_location="cpu", weights_only=True)
    head.load_state_dict({"weight": state["proj.weight"].float()})
    head.to(device).eval()
    for p in list(model.parameters()) + list(head.parameters()):
        p.requires_grad_(False)
    return model, head


def forward(model, head, rows) -> List[np.ndarray]:
    """Raw 24-slot scores (float64 copies of the fp32 values), one unpadded row at a time."""
    import torch

    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for r in rows:
            o = model(input_ids=torch.tensor([r["ids"]], device=device), output_hidden_states=True,
                      use_cache=False, logits_to_keep=1)
            h = o.hidden_states[-1][0, -1]
            out.append(head(h.float()).double().cpu().numpy())
    return out

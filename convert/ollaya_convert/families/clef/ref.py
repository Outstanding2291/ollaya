"""The PyTorch reference for Cloudflare/clef-flash: the authors' own `joint_schema_model.py` from the model
repository at the pinned revision, run in fp32 on its text-only path.

The repository holds the whole post-trained Qwen3.5-9B (vision tower included) as four BF16 shards, the joint
schema head (`joint_head.safetensors`, BF16) and `joint_schema_model.py`, which encodes a request
(`encode_record`), runs the backbone and the head (`ClefModel`) and answers a `/v1/systemone` body
(`systemone`). This module imports that file from the snapshot, so the prompt and the head are upstream's code.

    model, tokenizer = ref.load(snapshot, device="cpu")
    encoded = ref.encode(tokenizer, state, questions)               # systemone's checks + encode_record
    logits = ref.forward(model, tokenizer.pad_token_id, encoded)    # [k] raw option logits per question

Every question gets one logit per option, in upstream's option order: noul [true, false], a choice's labels
sorted, score levels in order. Probabilities are a plain softmax (no fitted temperature).
"""
from __future__ import annotations

import os
import sys

import torch

REPO = "Cloudflare/clef-flash"
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
SHARDS = ["model-%05d-of-00004.safetensors" % i for i in range(1, 5)]
HEAD = "joint_head.safetensors"
# encode_record's max_length. Upstream defaults to 16,384 tokens; Ollaya passes 4,096, so a request's activations fit
# next to the 18 GB of BF16 weights on a 24 GB GPU (longer sequences there spill into host memory and take minutes).
# The state is cut to fit, the way upstream cuts it at its own limit.
MAX_LEN = 4096


class RequestError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


def snapshot():
    from huggingface_hub import snapshot_download

    return os.environ.get("CLEF_MODEL") or snapshot_download(REPO, revision=REVISION)


def upstream(path):
    """The authors' `joint_schema_model` module from the snapshot."""
    if path not in sys.path:
        sys.path.insert(0, path)
    import joint_schema_model

    return joint_schema_model


def load(path, device="cpu"):
    """Upstream load_release_model in fp32 (TF32 off), line for line, with the tokenizer in place of the processor:
    -> (ClefModel, tokenizer). The processor's image and video halves need torchvision and Pillow, and only
    requests with media reach them; its tokenizer is this AutoTokenizer."""
    import json

    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    jsm = upstream(path)
    # upstream passes device_map={"": device}, which needs accelerate; loading, then moving, is the same model.
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.float32).to(device)
    backbone.config.use_cache = False
    head = jsm.JointSchemaHead(**json.load(open(os.path.join(path, "joint_head_config.json"), encoding="utf-8")))
    head.load_state_dict(load_file(os.path.join(path, HEAD)), strict=True)
    head = head.to(device=device, dtype=torch.float32)
    return jsm.ClefModel(backbone, head).eval(), AutoTokenizer.from_pretrained(path)


def encode(tokenizer, state, questions, max_length=MAX_LEN):
    """upstream systemone's request checks, then encode_record. Anything upstream raises on is a rejection."""
    jsm = sys.modules["joint_schema_model"]
    request = {"model": "clef-flash", "state": state, "questions": questions}
    try:
        if not isinstance(questions, dict) or not questions:
            raise ValueError("at least one question is required")
        for qid, q in questions.items():
            if q.get("type") not in jsm.QUESTION_TYPES:
                raise ValueError("%s: type must be noul, choice, or score" % qid)
            if q["type"] != "noul" and not q.get("criteria"):
                raise ValueError("%s: criteria must not be empty" % qid)
        return jsm.encode_record(tokenizer, request, max_length=max_length)
    except (ValueError, TypeError, AttributeError, KeyError) as e:
        raise RequestError("%s: %s" % (type(e).__name__, e)) from e


@torch.no_grad()
def forward(model, pad_id, encoded):
    """Raw option logits, one [k] float64 array per question, in upstream's option order."""
    jsm = sys.modules["joint_schema_model"]
    device = next(model.parameters()).device
    batch = jsm.collate_records([encoded], pad_id, device)
    return [z.double().cpu().numpy() for z in model(batch)[0]]

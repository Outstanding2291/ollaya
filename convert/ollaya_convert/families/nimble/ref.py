"""The PyTorch reference for bespokelabs/Bespoke-Nimble-9B-v2: the author's own serving code, run in fp32.

The model repo is a LoRA adapter on Qwen/Qwen3.5-9B plus the reference scorer: `serving_schema.py`
(prompts for 1..255 choices), `extended_schema.py`, `parallel_schema.py` (the training prompt contract)
and `inference.py` (`candidate_logits`, `decision_result`). This module imports those files from the
pinned snapshot, so the prompts and the readout are upstream's code, not a copy.

TypeSafe questions become a Nimble schema the way the author's `/v1/systemone` server does it
(`nimble/serving/compiler.py` at NIMBLE_GIT, with openjev-sglang's `SystemOneRequest` models at
OPENJEV_GIT for the defaults): `_schema` below is that mapping, line for line.

    tok, model = ref.load(adapter_dir, base_dir, device="cpu")
    prepared, meta = ref.prepare(tok, state, questions)     # upstream prepare_prompts on the mapped schema
    logits = ref.forward(model, prepared)                   # [k] raw candidate logits per question
    answers = ref.answers(meta, logits, T)                  # openjev scoring.answer at temperature T

The reference loads the base in fp32 with the adapter unmerged (upstream `PeftModel.from_pretrained`),
TF32 off. Upstream serves in BF16 autocast; parity is measured in fp32 like every other family.
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

REPO = "bespokelabs/Bespoke-Nimble-9B-v2"
REVISION = "4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe"
BASE = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
BASE_FILES = ["model.safetensors-%05d-of-00004.safetensors" % i for i in range(1, 5)]
NIMBLE_GIT = {"repo": "https://github.com/bespokelabsai/nimble", "commit": "62076b4f2d365b5879dafcf7f6dd072a1fe76df7",
              "file": "nimble/serving/compiler.py"}
OPENJEV_GIT = {"repo": "https://github.com/ekzhang/openjev-sglang", "commit": "7f84bedc169439f03379c2fa8d00ada220af2295",
               "file": "src/openjev/models.py"}
# openjev NoulCriteria defaults ("true" -> yes, "false" -> no).
NOUL_DEFAULTS = {"true": "Yes", "false": "No"}


class RequestError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


def snapshot():
    from huggingface_hub import snapshot_download

    adapter = os.environ.get("NIMBLE_ADAPTER") or snapshot_download(REPO, revision=REVISION)
    base = os.environ.get("NIMBLE_BASE") or snapshot_download(BASE, revision=BASE_REVISION)
    return adapter, base


def upstream(adapter_dir):
    """The author's modules from the snapshot: (serving_schema, inference)."""
    if adapter_dir not in sys.path:
        sys.path.insert(0, adapter_dir)
    import inference  # noqa: F401
    import serving_schema  # noqa: F401

    return sys.modules["serving_schema"], sys.modules["inference"]


def contract(adapter_dir):
    """(schema_config, serving_config, temperature_config) as shipped."""
    read = lambda name: json.load(open(os.path.join(adapter_dir, name), encoding="utf-8"))  # noqa: E731
    return read("schema_config.json"), read("serving_config.json"), read("temperature_config.json")


def load(adapter_dir, base_dir, device="cpu"):
    """Upstream ParallelScorer's model, in fp32: Qwen3_5ForConditionalGeneration + PeftModel (unmerged)."""
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    schema_config, serving_config, _ = contract(adapter_dir)
    if (schema_config["model"], schema_config["revision"]) != (BASE, BASE_REVISION):
        raise SystemExit("schema_config.json names %s@%s, not the pinned base" % (schema_config["model"], schema_config["revision"]))
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    _, inference = upstream(adapter_dir)
    inference.validate_serving_config(adapter_dir, tok)
    if device == "offload":
        # fp32 9B (36 GB) does not fit a 24 GB GPU: load on the CPU with the adapter, then let accelerate keep
        # what fits on the GPU and stream the rest from CPU memory, executing every layer on the GPU (TF32 off,
        # so the arithmetic stays fp32).
        from accelerate import dispatch_model, infer_auto_device_map

        base = Qwen3_5ForConditionalGeneration.from_pretrained(base_dir, dtype=torch.float32, attn_implementation="sdpa")
        base.config.use_cache = False
        model = PeftModel.from_pretrained(base, adapter_dir).eval()
        device_map = infer_auto_device_map(model, max_memory={0: os.environ.get("NIMBLE_GPU_MEM", "20GiB"), "cpu": "200GiB"},
                                           no_split_module_classes=base._no_split_modules)
        model = dispatch_model(model, device_map=device_map)
        model.offload_device_map = device_map
        return tok, model
    base = Qwen3_5ForConditionalGeneration.from_pretrained(base_dir, dtype=torch.float32, attn_implementation="sdpa")
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()
    return tok, model


def serialize(value):
    """nimble.serving.compiler.serialize."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


def _schema(questions):
    """NimbleCompiler.prepare's TypeSafe -> Nimble schema mapping, with openjev's defaults.

    -> (schema, meta). Type errors the wire validation already rejects are not repeated here."""
    schema, meta = {}, []
    for name, q in questions.items():
        t = q.get("type")
        if "instructions" not in q:
            raise RequestError("question %r: instructions are required" % name)
        field = {"description": serialize(q["instructions"])}
        crit = q.get("criteria")
        if t == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise RequestError("question %r: noul criteria must be an object" % name)
            c = crit or {}
            extra = sorted(set(c) - set(NOUL_DEFAULTS))
            if extra:  # openjev NoulCriteria: extra="forbid", aliases "true" and "false" only
                raise RequestError("question %r: noul criteria take only \"true\" and \"false\", got %s" % (name, extra))
            desc = {k: serialize(c[k]) if c.get(k) is not None else NOUL_DEFAULTS[k] for k in ("false", "true")}
            field.update(type="boolean", choices=[False, True], choice_descriptions=desc)
            k = 2
        elif t == "choice":
            labels = list(crit) if isinstance(crit, dict) else list(dict.fromkeys(crit))
            descs = crit if isinstance(crit, dict) else {}
            field.update(type="enum", choices=labels, choice_descriptions={
                key: serialize(descs[key]) if descs.get(key) is not None else key for key in labels})
            k = len(labels)
        elif t == "score":
            if not isinstance(crit, list):  # openjev ScoreQuestion: criteria is a list
                raise RequestError("question %r: score criteria must be a list of levels" % name)
            field.update(type="enum", choices=[str(i) for i in range(len(crit))],
                         choice_descriptions={str(i): serialize(text) for i, text in enumerate(crit)})
            k = len(crit)
        else:
            raise RequestError("question %r: unknown type %r" % (name, t))
        schema[name] = field
        meta.append({"qid": name, "type": t, "k": k, "criteria": crit})
    return schema, meta


def prepare(tok, state, questions, adapter_dir=None):
    """-> (PreparedPrompts, meta): upstream serving_schema.prepare_prompts at the shipped 8,192-token contract."""
    adapter_dir = adapter_dir or snapshot()[0]
    serving_schema, _ = upstream(adapter_dir)
    _, serving_config, _ = contract(adapter_dir)
    schema, meta = _schema(questions)
    try:
        prepared = serving_schema.prepare_prompts(tok, serialize(state), schema, serving_config["max_prompt_tokens"])
    except ValueError as e:
        raise RequestError(str(e)) from e
    return prepared, meta


@torch.no_grad()
def forward(model, prepared, adapter_dir=None):
    """Raw candidate logits (fp32), one [k] array per question: upstream candidate_logits, one row at a time."""
    _, inference = upstream(adapter_dir or snapshot()[0])
    collate = inference.CandidateCollator(0)
    device = "cuda:0" if getattr(model, "offload_device_map", None) else next(model.parameters()).device
    out = []
    for ids, cand in zip(prepared.full_ids, prepared.candidate_ids):
        batch = {k: v.to(device) for k, v in collate([{"input_ids": ids, "candidate_ids": cand}]).items()}
        out.append(inference.candidate_logits(model, batch)[0].double().cpu().numpy())
    return out


def _softmax(z, t):
    peak = max(z)
    w = [math.exp((x - peak) / t) for x in z]
    s = math.fsum(w)
    return [x / s for x in w]


def _confidence(p):
    """openjev scoring.confidence: 1 - H(p)/ln k (upstream's own statistic, not TypeSafe's)."""
    if len(p) < 2:  # one option: nothing to be uncertain about (openjev itself takes 2 or more)
        return 1.0
    h = -math.fsum(x * math.log(x) for x in p if x > 0)
    return min(1.0, max(0.0, 1 - h / math.log(len(p))))


def answers(meta, logits, temperature):
    """openjev scoring.answer on the candidate logits at `temperature` (what the author's server returns)."""
    out = {}
    for m, z in zip(meta, logits):
        p = _softmax([float(x) for x in z], temperature)
        if m["type"] == "noul":
            out[m["qid"]] = {"type": "noul", "noul": p[1]}
        elif m["type"] == "score":
            out[m["qid"]] = {"type": "score", "score": math.fsum(i * x for i, x in enumerate(p)),
                             "probabilities": {str(i): x for i, x in enumerate(p)}, "confidence": _confidence(p)}
        else:
            keys = list(m["criteria"]) if isinstance(m["criteria"], dict) else list(dict.fromkeys(m["criteria"]))
            dist = dict(zip(keys, p))
            out[m["qid"]] = {"type": "choice", "choice": max(dist, key=dist.__getitem__), "probabilities": dist,
                             "confidence": _confidence(p)}
    return out

"""Export bespokelabs/Bespoke-Nimble-9B-v2 (Qwen3.5-9B + LoRA) to a weightless ONNX graph.

    uv run --with peft==0.21.0 python -m ollaya_convert.families.nimble.export --out out/nimble-9b-v2 \
        [--adapter ADAPTER_SNAPSHOT --base BASE_SNAPSHOT]

Graph (layout `nimble-codes-v1`, see layout.py and docs/families/nimble.md):
    inputs   input_ids   int64   [rows, seq]  one row per question; seq a multiple of 64, right-padded
             last_pos    int64   [rows]       index of the row's last real token
    outputs  cand_logits float32 [rows, 255]  next-token logits of the 255 option codes (A..Z, AA, AB, ...)

The LoRA is NOT merged: every adapted Linear runs as `x W^T + 2.0 * (x A^T) B^T`, so the graph references
the upstream files byte for byte: the base shards `model.safetensors-0000i-of-00004.safetensors` (BF16,
Qwen/Qwen3.5-9B; its untied `lm_head` is read only at the 255 code rows) and the adapter
`adapter_model.safetensors` (F32).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil

import torch

from ..llm_common import onnx_export as ox
from ..llm_common.qwen35 import CHUNK, Qwen35Trunk
from ...weightless_sharded import safetensors_source
from . import ref
from .layout import NOUL_DEFAULTS, NimbleLayout

INPUT_NAMES = ["input_ids", "last_pos"]
OUTPUT_NAMES = ["cand_logits"]
SENTINEL = "\u0000NIMBLE_CONTENT\u0000"


class NimbleGraph(torch.nn.Module):
    def __init__(self, text_model, lm_head, code_ids):
        super().__init__()
        self.trunk = Qwen35Trunk(text_model)
        self.lm_head = lm_head
        self.register_buffer("code_ids", torch.tensor(code_ids, dtype=torch.long), persistent=False)

    def forward(self, input_ids, last_pos):
        h = self.trunk(input_ids).float()
        hs = h[torch.arange(h.shape[0], device=h.device), last_pos]
        w = self.lm_head.weight[self.code_ids]           # the untied LM head, only the code rows
        return hs @ w.transpose(0, 1)


def rename(name):
    """Graph initializer name -> checkpoint tensor names (base shards or the adapter)."""
    if name.startswith("trunk.m."):
        x = name[len("trunk.m."):]
        if ".lora_" in x:
            return ["base_model.model.model.language_model." + x.replace(".default", "")]
        return ["model.language_model." + x.replace(".base_layer", "")]
    if name.startswith("lm_head."):
        return [name.replace(".base_layer", "")]
    return [name]


def templates(tok, system_prompt):
    """{"letter"|"short": {"pre", "post"}}: the chat template rendered around the user content."""
    out = {}
    for key, system in (("letter", system_prompt), ("short", system_prompt.replace("one-letter", "short"))):
        text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": SENTINEL}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pre, post = text.split(SENTINEL)
        assert post.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), repr(post)
        out[key] = {"system": system, "pre": pre, "post": post}
    return out


def export(out_dir, adapter_dir, base_dir):
    schema_config, serving_config, temperature_config = ref.contract(adapter_dir)
    tok, model = ref.load(adapter_dir, base_dir, device="cpu")
    code_ids = serving_config["candidate_token_ids"]
    assert len(code_ids) == serving_config["max_choices"] == 255
    peft_root = model.base_model.model                     # Qwen3_5ForConditionalGeneration with LoRA wrappers
    graph = NimbleGraph(peft_root.model.language_model, peft_root.lm_head, code_ids).eval()

    # Two rows (a real request): every dynamic axis > 1.
    prepared, _ = ref.prepare(tok, "The customer was charged twice for order A-104 and wants the duplicate refunded. " * 4,
                              {"team": {"type": "choice", "instructions": "Which team?",
                                        "criteria": {"billing": "charges", "tech": "bugs", "other": None}},
                               "upset": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]}},
                              adapter_dir)
    lens = [len(x) for x in prepared.full_ids]
    T = -(-max(lens) // CHUNK) * CHUNK
    ids = torch.zeros((2, T), dtype=torch.long)
    for i, x in enumerate(prepared.full_ids):
        ids[i, :len(x)] = torch.tensor(x)
    args = (ids, torch.tensor([n - 1 for n in lens]))
    with torch.no_grad():
        want = ref.forward(model, prepared, adapter_dir)
        got = graph(*args)
    print("eager graph vs upstream candidate_logits: %.2e"
          % max(float((got[i, :len(w)].double() - torch.tensor(w)).abs().max()) for i, w in enumerate(want)))

    R = torch.export.Dim("rows", min=1, max=4096)
    N = torch.export.Dim("chunks", min=1, max=4096)
    dyn = {"input_ids": {0: R, 1: CHUNK * N}, "last_pos": {0: R}}
    tmp = ox.scratch_dir("nimble-export-")
    secs = ox.export_graph(graph, args, INPUT_NAMES, OUTPUT_NAMES, dyn, os.path.join(tmp, "model.onnx"))
    print("exported in %.0fs" % secs)
    tmpl = templates(tok, schema_config["system_prompt"])
    del graph, peft_root, model, want, got
    gc.collect()

    base_ckpts = [os.path.join(base_dir, f) for f in ref.BASE_FILES]
    sources = [
        safetensors_source(f, p, repo=ref.BASE, revision=ref.BASE_REVISION, filename=f)
        for f, p in zip(ref.BASE_FILES, base_ckpts)
    ] + [
        safetensors_source("adapter_model.safetensors", os.path.join(adapter_dir, "adapter_model.safetensors"),
                           repo=ref.REPO, revision=ref.REVISION, filename="adapter_model.safetensors"),
    ]
    report = ox.weightless(tmp, out_dir, sources, rename)
    ox.cleanup(tmp)
    shutil.copy(os.path.join(adapter_dir, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))

    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8"))
    temperature = float(temperature_config["temperature"])
    decision = {
        "engine": "onnx",
        "family": "nimble",
        "layout": "nimble-codes-v1",
        "upstream": {"repo": ref.REPO, "revision": ref.REVISION, "base": ref.BASE, "base_revision": ref.BASE_REVISION,
                     "prompt_code_sha256": schema_config["prompt_code_sha256"],
                     "serving_sources_sha256": serving_config["source_sha256"],
                     "typesafe_mapping": ref.NIMBLE_GIT, "typesafe_defaults": ref.OPENJEV_GIT,
                     "lora": {"r": cfg["r"], "alpha": cfg["lora_alpha"], "scaling": cfg["lora_alpha"] / cfg["r"],
                              "merged": False}},
        "contract": {
            "inputs": {
                "input_ids": {"dtype": "int64", "shape": ["rows", "seq"],
                              "note": "one row per question; seq a multiple of 64; right-pad with any id"},
                "last_pos": {"dtype": "int64", "shape": ["rows"], "note": "index of the row's last real token"},
            },
            "outputs": {"cand_logits": {"dtype": "float32", "shape": ["rows", 255],
                                        "note": "next-token logits of codes.ids; a question reads its first k"}},
            "seq_multiple": CHUNK,
            "positions": "0..seq-1, implicit",
            "attention": "causal; no mask input (right padding cannot reach earlier positions)",
        },
        "templates": tmpl,
        "row": "pre + safe_json({context, schema}) + \"\\n\\nRequested field: \" + safe_json(qid) + post; "
               "tokenized whole, add_special_tokens=False, special tokens parsed",
        "codes": {"strings": serving_config["candidate_codes"], "ids": code_ids,
                  "letter_codes": serving_config["legacy_max_choices"]},
        "noul_defaults": NOUL_DEFAULTS,
        "max_prompt_tokens": serving_config["max_prompt_tokens"],
        "min_options": 1,
        "max_options": serving_config["max_choices"],
        "special_tokens": {"pad": tok.pad_token_id, "add_special_tokens": False},
        "option_logits": {"choice": "cand_logits[row, :k]", "noul": "cand_logits[row, :2] (0 = false, 1 = true)",
                          "score": "cand_logits[row, :levels]"},
        "opset": ox.OPSET,
        "precision": "fp32 compute; base weights BF16, widened by Cast; adapter F32",
        "weights_in_memory": ox.weights_in_memory(report),
    }
    calibration = {"temperature": [temperature] * 3, "temperature_by_options": {},
                   "source": "temperature_config.json (%s; the author's recommended default, fitted on the "
                             "original Bespoke-Nimble-9B and transferred to v2)" % temperature_config["method"]}
    files = {
        "model": "nimble-9b-v2",
        "layers": [
            {"role": "graph", "path": "model.onnx", "hosted_by": "ollaya", "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
             "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))},
            *[ox.file_entry("weights/base", ref.BASE, ref.BASE_REVISION, f, p, location=f)
              for f, p in zip(ref.BASE_FILES, base_ckpts)],
            ox.file_entry("weights/adapter", ref.REPO, ref.REVISION, "adapter_model.safetensors",
                          os.path.join(adapter_dir, "adapter_model.safetensors"), location="adapter_model.safetensors"),
            ox.file_entry("tokenizer", ref.REPO, ref.REVISION, "tokenizer.json", os.path.join(adapter_dir, "tokenizer.json")),
            {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
            {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
            ox.file_entry("license", ref.REPO, ref.REVISION, "LICENSE", os.path.join(adapter_dir, "LICENSE"))
            | {"note": "Apache-2.0 (adapter); the base Qwen3.5-9B is Apache-2.0"},
        ],
        "weightless": {k: v for k, v in report.items() if k != "unused"},
        "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()},
    }
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    # the layout port must reproduce upstream's prompt for the sample request
    import tokenizers

    lay = NimbleLayout(tokenizers.Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json")), decision)
    rows, _ = lay.encode("The customer was charged twice for order A-104 and wants the duplicate refunded. " * 4,
                         {"team": {"type": "choice", "instructions": "Which team?",
                                   "criteria": {"billing": "charges", "tech": "bugs", "other": None}},
                          "upset": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]}})
    assert [r["ids"] for r in rows] == [list(x) for x in prepared.full_ids], "layout port differs from upstream"
    assert [r["candidates"] for r in rows] == [list(x) for x in prepared.candidate_ids]
    print(json.dumps(files["weightless"]["stats"]), "inline bytes", files["weightless"]["inline_bytes"],
          "graph MB %.1f" % (files["layers"][0]["bytes"] / 2**20), "unused", files["unused_checkpoint_tensors"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None, help="local snapshot of the Nimble repo at the pinned revision")
    ap.add_argument("--base", default=None, help="local snapshot of Qwen/Qwen3.5-9B at the pinned revision")
    a = ap.parse_args()
    adapter, base = (a.adapter, a.base) if a.adapter and a.base else ref.snapshot()
    export(a.out, adapter, base)


if __name__ == "__main__":
    main()

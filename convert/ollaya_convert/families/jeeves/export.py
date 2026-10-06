"""Export PostHog/jeeves (fused Qwen3.5-9B + pointer head) to a weightless ONNX graph, `jeeves-markers-v1`.

    JEEVES_SRC=<checkout> uv run python -m ollaya_convert.families.jeeves.export --out out/jeeves-9b \
        [--model SNAPSHOT --qwen QWEN3.5-9B SNAPSHOT --tokenizer KEV-9B tokenizer.json]

Graph (the kev contract, see docs/families/jeeves.md):
    inputs   input_ids   int64   [rows, seq]  one row per question; seq a multiple of 64, right-padded
             decide_pos  int64   [rows]       the row's last position (the decide marker)
             opt_pos     int64   [rows, k]    the option-close markers after </think> (pad with 0)
    outputs  scores      float32 [rows, k]    raw pointer scores k(h_opt) . q(h_decide) / 16

The trunk is transformers' Qwen3.5 text model (config: Qwen/Qwen3.5-9B's text_config) holding Jeeves'
fused weights, recomputed by llm_common.qwen35; every initializer references Jeeves' five BF16 shards or
`head.pt` (a torch zip, tensors stored uncompressed) by byte offset.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import shutil

import torch

from ..llm_common import onnx_export as ox
from ..llm_common.qwen35 import CHUNK, Qwen35Trunk
from ...weightless_sharded import safetensors_source, torchzip_source
from . import ref
from .check import decision_for
from .layout import MAX_OPTIONS, JeevesLayout

INPUT_NAMES = ["input_ids", "decide_pos", "opt_pos"]
OUTPUT_NAMES = ["scores"]


class JeevesGraph(torch.nn.Module):
    def __init__(self, text_model, head):
        super().__init__()
        self.trunk = Qwen35Trunk(text_model)
        self.head = head

    def forward(self, input_ids, decide_pos, opt_pos):
        h = self.trunk(input_ids).float()
        rows = torch.arange(h.shape[0], device=h.device)
        q = self.head.q(h[rows, decide_pos])                 # [R, P]
        k = self.head.k(h[rows.unsqueeze(1), opt_pos])       # [R, K, P]
        return (k @ q.unsqueeze(-1)).squeeze(-1) * self.head.scale


def text_model(model_dir, qwen_dir):
    """transformers' Qwen3.5 text model with Jeeves' fused weights (fp32)."""
    from safetensors.torch import load_file
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    cfg = json.load(open(os.path.join(qwen_dir, "config.json"), encoding="utf-8"))["text_config"]
    config = Qwen3_5TextConfig(**cfg)
    # Built on the meta device and filled by assignment, so the fp32 weights (36 GB) exist once.
    with torch.device("meta"):
        m = Qwen3_5TextModel(config)
    state = {}
    for f in ref.SHARDS:
        for k, v in load_file(os.path.join(model_dir, f)).items():
            if k.startswith("model."):
                state[k[len("model."):]] = v.float()
            del v
    missing, unexpected = m.load_state_dict(state, strict=False, assign=True)
    del state
    if missing or unexpected:
        raise SystemExit("weights do not fit the text model: missing %s, unexpected %s" % (missing[:5], unexpected[:5]))
    m.rotary_emb = type(m.rotary_emb)(config=config)   # its inv_freq buffer is not in the checkpoint
    return m.eval()


def rename(name):
    if name.startswith("trunk.m."):
        return ["model." + name[len("trunk.m."):]]
    if name.startswith("head."):
        return [name[len("head."):]]
    return [name]


def export(out_dir, model_dir, qwen_dir, tokenizer_json):
    import tokenizers

    ref._src()
    from model.head import PointerHead

    meta = json.load(open(os.path.join(model_dir, "export.json"), encoding="utf-8"))
    head = PointerHead(4096, meta["head_dim"]).float()
    head.load_state_dict(torch.load(os.path.join(model_dir, "head.pt"), map_location="cpu"))
    head.temperature = 1.0
    head.eval()
    tm = text_model(model_dir, qwen_dir)
    graph = JeevesGraph(tm, head).eval()

    enc = ref.encoder(model_dir)
    decision_layout = decision_for(enc)
    lay = JeevesLayout(tokenizers.Tokenizer.from_file(tokenizer_json), decision_layout)
    state = "The customer was charged twice for order A-104 and wants the duplicate refunded. " * 4
    questions = {"a": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "tech": "bugs", "other": None}},
                 "b": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]}}
    rows, _ = lay.encode(state, questions)
    up_rows, _ = ref.encode(enc, state, questions)
    assert rows == up_rows, "layout port differs from upstream"
    T = -(-max(len(r["ids"]) for r in rows) // CHUNK) * CHUNK
    ids = torch.full((2, T), enc.pad_id, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r["ids"])] = torch.tensor(r["ids"])
    args = (ids, torch.tensor([r["decide"] for r in rows]), torch.tensor([r["opts"] for r in rows]))
    with torch.no_grad():
        got = graph(*args)
    print("graph scores on the sample:", [[round(float(x), 4) for x in row] for row in got])

    R = torch.export.Dim("rows", min=1, max=4096)
    N = torch.export.Dim("chunks", min=1, max=4096)
    K = torch.export.Dim("options", min=1, max=MAX_OPTIONS)
    dyn = {"input_ids": {0: R, 1: CHUNK * N}, "decide_pos": {0: R}, "opt_pos": {0: R, 1: K}}
    tmp = ox.scratch_dir("jeeves-export-")
    secs = ox.export_graph(graph, args, INPUT_NAMES, OUTPUT_NAMES, dyn, os.path.join(tmp, "model.onnx"))
    print("exported in %.0fs" % secs)
    del graph, tm
    gc.collect()

    sources = [safetensors_source(f, os.path.join(model_dir, f), repo=ref.REPO, revision=ref.REVISION, filename=f)
               for f in ref.SHARDS]
    sources.append(torchzip_source("head.pt", os.path.join(model_dir, "head.pt"), repo=ref.REPO,
                                   revision=ref.REVISION, filename="head.pt"))
    report = ox.weightless(tmp, out_dir, sources, rename)
    ox.cleanup(tmp)
    shutil.copy(tokenizer_json, os.path.join(out_dir, "tokenizer.json"))
    decision = {
        "engine": "onnx", "family": "jeeves", "layout": "jeeves-markers-v1",
        "upstream": {"repo": ref.REPO, "revision": ref.REVISION, "code": ref.JEEVES_GIT, "format": meta["format"],
                     "tokenizer": {"repo": ref.TOKENIZER[0], "revision": ref.TOKENIZER[1], "path": ref.TOKENIZER[2]}},
        "contract": {
            "inputs": {"input_ids": {"dtype": "int64", "shape": ["rows", "seq"],
                                     "note": "one row per question; seq a multiple of 64; right-pad with the pad id"},
                       "decide_pos": {"dtype": "int64", "shape": ["rows"], "note": "the row's last position"},
                       "opt_pos": {"dtype": "int64", "shape": ["rows", "k"],
                                   "note": "option-close positions after </think>; shorter rows pad with 0"}},
            "outputs": {"scores": {"dtype": "float32", "shape": ["rows", "k"], "note": "raw pointer scores"}},
            "seq_multiple": CHUNK, "positions": "0..seq-1, implicit", "attention": "causal; no mask input"},
        **decision_layout,
        "pad": enc.pad_id,
        "min_options": 1, "max_options": MAX_OPTIONS,
        "option_logits": {"choice": "scores[row, :k]", "noul": "scores[row, :2] (0 = no = false, 1 = yes = true)",
                          "score": "scores[row, :levels]"},
        "opset": ox.OPSET,
        "precision": "fp32 compute; weights BF16 (widened by Cast); head F32",
        "weights_in_memory": ox.weights_in_memory(report),
    }
    calibration = {"temperature": [float(meta["temperature"])] * 3, "temperature_by_options": {},
                   "source": "export.json temperature (one softmax temperature fitted on the dev set)"}
    files = {"model": "jeeves-9b", "layers": [
        {"role": "graph", "path": "model.onnx", "hosted_by": "ollaya", "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
         "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))},
        *[ox.file_entry("weights", ref.REPO, ref.REVISION, f, os.path.join(model_dir, f), location=f) for f in ref.SHARDS],
        ox.file_entry("weights/head", ref.REPO, ref.REVISION, "head.pt", os.path.join(model_dir, "head.pt"), location="head.pt"),
        ox.file_entry("tokenizer", ref.TOKENIZER[0], ref.TOKENIZER[1], ref.TOKENIZER[2], tokenizer_json),
        {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
        {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
        ox.file_entry("license", ref.REPO, ref.REVISION, "LICENSE", os.path.join(model_dir, "LICENSE"))],
        "weightless": {k: v for k, v in report.items() if k != "unused"},
        "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()}}
    for name, obj in (("decision.json", decision), ("calibration.json", calibration), ("files.json", files)):
        ox.write_json(os.path.join(out_dir, name), obj)
    print(json.dumps(files["weightless"]["stats"]), "graph MB %.1f" % (files["layers"][0]["bytes"] / 2**20),
          "unused", files["unused_checkpoint_tensors"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--qwen", default=None, help="a Qwen/Qwen3.5-9B snapshot (for its text_config)")
    ap.add_argument("--tokenizer", default=None)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download

    model = a.model or ref.snapshot()
    qwen = a.qwen or snapshot_download("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                                       allow_patterns=["config.json"])
    tok = a.tokenizer or hf_hub_download(ref.TOKENIZER[0], ref.TOKENIZER[2], revision=ref.TOKENIZER[1])
    os.makedirs(a.out, exist_ok=True)
    export(a.out, model, qwen, tok)


if __name__ == "__main__":
    main()

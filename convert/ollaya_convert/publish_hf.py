"""Publish Ollaya's derived files for a model to its Hugging Face repository.

    uv run python -m ollaya_convert.publish_hf laya [--org ollaya-dev] [--dry-run out/hf]

Uploads, to `huggingface.co/<org>/<model>`, what Ollaya derives from the upstream checkpoint:
the ONNX graphs (weightless: they reference the author's `model.safetensors` by byte offset),
the decision and calibration configs, and a model card crediting the original authors. A GGUF
family (run on llama.cpp from the author's own GGUF file) has no graph: its repository holds the
configs, and its card names the GGUF file each tag pulls.
Weights are never uploaded; `ollaya pull` fetches them from the author's repository.

The files come from `registry/` (run `package.py` first), so what is published is exactly what
the registry serves, digest for digest.
"""
import argparse
import json
import os
import shutil

from .catalog import CATALOG
from .package import MEDIA, REGISTRY

CARD = """---
{license_yaml}
base_model:
{base_models}
library_name: onnx
tags:
- ollaya
- onnx
- decision-model
- system-one
pipeline_tag: text-classification
---

# {model} for Ollaya

[Ollaya](https://github.com/ollaya-dev/ollaya) package of {base_links}{by}.
Ollaya runs open decision models locally, the way Ollama runs LLMs: typed questions in,
calibrated answers out, behind a TypeSafe-compatible API.

```sh
ollaya run {model}
```

## What is in this repository

This repository holds only the files Ollaya derives, with no weights. Each graph is an ONNX export of the
original model whose weights **reference the authors' own weight files by byte offset**,
so `ollaya pull` downloads the weights from the upstream repositories, unmodified and pinned to a
commit, and verifies their sha256.

| Tag | Upstream | Files |
|---|---|---|
{rows}

{graphs_note} Each tag also has `decision.json` (sequence layout, special tokens) and
`calibration.json` (temperatures).{questions_note}

## Parity

{parity}

## License

{license_text_note}
"""


GGUF_CARD = """---
{license_yaml}
base_model:
{base_models}
tags:
- ollaya
- gguf
- llama.cpp
- decision-model
- system-one
pipeline_tag: text-classification
---

# {model} for Ollaya

[Ollaya](https://github.com/ollaya-dev/ollaya) package of {base_links}{by}.
Ollaya runs open decision models locally, the way Ollama runs LLMs: typed questions in,
calibrated answers out, behind a TypeSafe-compatible API.

```sh
ollaya run {model}
```

## What is in this repository

This repository holds only the files Ollaya derives, with no weights. The model is the authors' own GGUF
file: `ollaya pull` downloads it from their repository, unmodified and pinned to a commit, verifies its
sha256, and Ollaya runs it on llama.cpp.

| Tag | The authors' GGUF | Files |
|---|---|---|
{rows}

Each tag has `decision.json` (the prompt, the option labels Ollaya reads and llama.cpp's settings) and
`calibration.json` (temperatures).{mmproj_note}

## Parity

{parity}

## License

{license_text_note}
"""


LAYA_PARITY = """The exports are checked against the PyTorch reference on 2,383 questions per checkpoint.
The checks use typed-decisions plus multilingual and edge cases:

- **fp32:** the same decision on 100% of questions, with probabilities within 1.1e-4.
- **fp16:** the same decision on 99.1–99.6% of questions. Nearly all of the differences are
  near-ties between the top two options.

The fp32 graphs (CPU) are opset 23: attention runs as fused `Attention` nodes, which ONNX Runtime's
CPU provider runs faster than the decomposed ops, so they need ONNX Runtime 1.23 or newer. The fp16
graphs (GPU) are opset 20."""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(CATALOG))
    ap.add_argument("--org", default="ollaya-dev")
    ap.add_argument("--dry-run", metavar="DIR", help="write the repository to DIR instead of uploading")
    a = ap.parse_args()
    spec = CATALOG[a.model]
    ns, model = spec["namespace"], spec["model"]

    stage = a.dry_run or os.path.join(os.path.dirname(__file__), "..", "out", "hf", model)
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    rows, base_models, any_fp16, any_questions, any_mmproj = [], [], False, False, False
    gguf = all("gguf" in v for v in spec["tags"].values())
    for tag in spec["tags"]:
        with open(os.path.join(REGISTRY, "v2", ns, model, "manifests", tag), encoding="utf-8") as f:
            manifest = json.load(f)
        os.makedirs(os.path.join(stage, tag))
        names = []
        for layer in manifest["layers"]:
            blob = os.path.join(REGISTRY, "blobs", "sha256-" + layer["digest"].split(":", 1)[1])
            kind = layer["mediaType"].rsplit(".", 1)[-1]
            ann = layer.get("annotations", {})
            # A second graph (a vision model's image graph) is named by its role.
            graph = ("%s-%s.onnx" % (ann["org.ollaya.graph"], ann.get("org.ollaya.precision", "fp32"))
                     if "org.ollaya.graph" in ann else "model-%s.onnx" % ann.get("org.ollaya.precision", "fp32"))
            name = {"onnx": graph,
                    "decision": "decision.json", "calibration": "calibration.json",
                    "questions": "questions.json"}.get(kind)
            if name and os.path.exists(blob):
                shutil.copy(blob, os.path.join(stage, tag, name))
                names.append(name)
                any_fp16 = any_fp16 or name == "model-fp16.onnx"
                any_questions = any_questions or name == "questions.json"
        v = spec["tags"][tag]
        # The tag's own repo, then any other repo its weights come from (a LoRA's base model).
        sources = [(v["repo"], v["commit"])]
        for source in (v["weights"].values() if isinstance(v.get("weights"), dict) else []):
            if isinstance(source, tuple) and source[:2] not in sources:
                sources.append(source[:2])
        for repo, _ in sources:
            if repo not in base_models:
                base_models.append(repo)
        upstream = ", ".join("[%s@%s](https://huggingface.co/%s/tree/%s)" % (r, c[:7], r, c) for r, c in sources)
        if gguf:  # the file the tag pulls, and a vision tag's projector from the same revision
            upstream += " `%s`" % v["gguf"] + (" + `%s`" % v["mmproj"] if v.get("mmproj") else "")
            any_mmproj = any_mmproj or bool(v.get("mmproj"))
        rows.append("| `%s:%s` | %s | %s |" % (model, tag, upstream, ", ".join("`%s/%s`" % (tag, n) for n in names)))
    by = spec.get("author", "")
    base_model = base_models[0]
    # An SPDX id goes in as is; anything else (arbiter's "Apache-2.0 (...) and the Gemma Terms of Use (...)")
    # is Hugging Face's `other`, with the name and link the catalog gives.
    lic = spec["license"]
    if " " in lic:
        name, link = spec["hf_license"]
        license_yaml = "license: other\nlicense_name: %s\nlicense_link: %s" % (name, link)
    else:
        license_yaml = "license: " + lic.lower()
    common = dict(
        license_yaml=license_yaml, model=model,
        base_models="\n".join("- " + b for b in base_models),
        base_links=" and ".join("**[%s](https://huggingface.co/%s)**" % (b, b) for b in base_models),
        by=(" by " + by) if by else "", rows="\n".join(rows),
        parity=spec.get("parity", LAYA_PARITY),
        license_text_note="Same as the upstream model (%s). Ollaya itself is Apache-2.0." % spec["license"],
    )
    if "PENDING" in common["parity"]:
        raise SystemExit("%s: the catalog's parity text is still a placeholder" % model)
    if gguf:
        card = GGUF_CARD.format(**common, mmproj_note=(
            " A vision tag also pulls the authors' vision projector from the same revision, which reads the "
            "images." if any_mmproj else ""))
    else:
        card = CARD.format(**common, graphs_note=(
            "Each tag has an fp32 graph (CPU) and an fp16 graph (GPU)." if any_fp16
            else "Each tag has an fp32 graph, used on CPU and GPU."), questions_note=(
            " `questions.json` holds the built-in questions: the model answers those and no "
            "others, so requests leave `questions` out." if any_questions else ""))
    with open(os.path.join(stage, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print("staged %s" % os.path.abspath(stage))
    if a.dry_run:
        return

    from huggingface_hub import HfApi

    api = HfApi()
    repo = "%s/%s" % (a.org, model)
    api.create_repo(repo, repo_type="model", exist_ok=True)
    info = api.upload_folder(repo_id=repo, folder_path=stage, commit_message="Ollaya package for %s" % base_model)
    print("published https://huggingface.co/%s (%s)" % (repo, info.oid[:7] if hasattr(info, "oid") else "ok"))


if __name__ == "__main__":
    main()

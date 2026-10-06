# arbiter (`arbiter-fixed-v1`)

**Arbiter v3.3** ([hiteshluke/arbiter-4b](https://huggingface.co/hiteshluke/arbiter-4b), by Codekins Pvt Ltd ·
Zyot Lab) is Gemma 3 4B IT with a LoRA (r 16, α 32, on q/k/v/o and gate/up/down) and a **fixed 24-slot head**,
`nn.Linear(2560, 24, bias=False)`, read at the last position of the prompt. The head's rows started as the base
LM head's rows for the 24 answer tokens and were trained with the LoRA; the checkpoint ships the average of the
last five evaluation heads (BF16, `head.pt`). One forward pass answers one question. Training code:
`training_v3_6/train.py` in [CodekinsTech/arbiter](https://github.com/CodekinsTech/arbiter) at `e1cb30fd`.

| Tag | Weights | Temperature |
|---|---|---|
| `arbiter:4b`, `arbiter:latest` | `unsloth/gemma-3-4b-it@bf46152c`: two BF16 shards and `tokenizer.json`; `hiteshluke/arbiter-4b@0c44271c`: `adapter_model.safetensors` (F32) and `head.pt` | 1.0 (none fitted) |

The head has three fixed slot ranges, so a question has at most 16 options (choice) and a score exactly 6 levels:

| Slots | Tokens | Read for |
|---|---|---|
| 0, 1 | `T`, `F` | noul: option logits `[F, T]` (Ollaya's order: false, true) |
| 2..17 | `A`..`P` | choice: option j at slot 2 + j |
| 18..23 | `0`..`5` | score: level j at slot 18 + j |

### Base repository

`unsloth/gemma-3-4b-it@bf46152c` is a copy of Google's `google/gemma-3-4b-it@093f9f38`. The Hub reports the same
LFS sha256 and size for every file the runtime reads:

| File | sha256 | Bytes |
|---|---|---|
| `model-00001-of-00002.safetensors` | `eb5fd5e97ddd07b56778733e9653c07312529cb00980a318fc3e1c4e3b5a8f1f` | 4,961,251,752 |
| `model-00002-of-00002.safetensors` | `fdde0e5aa5ced0fa203b3d50f4ab78168b7e3a3e08c6349f5cc9326666e1bb13` | 3,639,026,128 |
| `tokenizer.json` | `4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795` | 33,384,568 |

`model.safetensors.index.json` and `tokenizer.model` are identical too. Only metadata differs (`config.json`,
`generation_config.json`, `tokenizer_config.json`, `special_tokens_map.json`, the README, and the copy's extra
`chat_template.jinja`), and the runtime reads none of it. The manifest points at the copy because
`google/gemma-3-4b-it` is gated (the Hub lists it as `gated: manual`: each account must request access and accept
the Gemma terms before it can download), and Ollaya pulls without a Hugging Face token. It is also the name the
training script loads the base and the tokenizer by.

The tokenizer comes from the base repository, not from `hiteshluke/arbiter-4b`: that repository's `tokenizer.json`
(sha256 `b666c93e…`) is the same tokenizer with truncation to 255 tokens switched on.

## Sequence

`convert/ollaya_convert/families/arbiter/layout.py` (the port) and `crates/ollaya-decision/src/arbiter.rs`. One
causal row per question, the training prompt after Gemma's `<bos>`:

```text
ids = [<bos>] + tok("State: {render(state)}\n\nQuestion: {render(instructions)}\n\nOptions:\n{block}\n\nAnswer:")

block   noul    "T. Yes / True\nF. No / False"
        choice  "A. {option 0}\nB. {option 1}\n..."      one line per option, letters A..P
        score   "0\n1\n2\n3\n4\n5"
```

- `render`: `null` is `""`, a scalar is Python's `str()` (`True`, `1.0`, `1e-05`), a list is `- item` lines and an
  object `key: value` lines, two spaces per level. A choice option is `name`, or `name: render(description)`.
- The noul and score descriptions are not part of the prompt: the model was trained on the fixed blocks.
- The head reads the last position (`last_pos = len(ids) - 1`).
- **Rejected (422), never truncated:** a choice whose criteria are not an object, or hold more than 16 options
  (`TOO_MANY_OPTIONS`); a score with other than 6 levels; a row over 8,192 tokens. A request with one such
  question is rejected whole, like any invalid request.

The reference prompt (`ref.py`) is the training script's three prompt functions, copied verbatim, tokenized by
transformers as training tokenizes (`tok(prompt)`, which adds `<bos>`). The request -> text mapping (`render`,
option texts, validation) is Ollaya's, shared by both: the training data was plain text.

## Graph

| Tensor | dtype | Shape | |
|---|---|---|---|
| `input_ids` | int64 | `[rows, seq]` | one row per question, right-padded; `seq` a multiple of 64 |
| `last_pos` | int64 | `[rows]` | the row's last token |
| `scores` | float32 | `[rows, 24]` | the head's raw slot scores at `last_pos` |

`export.py`: transformers' Gemma 3 text model recomputed by `llm_common/gemma3.py` (positions `0..seq-1`, no mask
input: every layer is causal, and the sliding layers see the last 1,024 positions), the LoRA unmerged
(`x·Wᵀ + 2·(x·Aᵀ)·Bᵀ`), and the head. Weightless: the base shards, the adapter and the head tensor inside
`head.pt` are referenced by byte offset; nothing is re-hosted. The vision tower and its LoRA tensors are unused.
`weights_in_memory` is `bf16`, about 8 GB. The runner (`crates/ollaya-runner/src/arbiter.rs`) batches rows
shortest first under 8,192 padded tokens per `session.run` and returns the scores at each question's slots.

## Differences from upstream

- **Precision.** Training and the model-card benchmarks ran the base 4-bit (NF4, bitsandbytes) with BF16 compute;
  the adapter was trained against that 4-bit base (`unsloth/gemma-3-4b-it-unsloth-bnb-4bit`). Ollaya runs the
  unquantized BF16 base in fp32, as does the reference here. Measured on the same rows, the unquantized base scores
  equal or higher (see Quality).
- **Length.** Training cut prompts at 768 tokens and the benchmark script at 1,024 (from the right, which drops
  `Answer:`). Ollaya reads rows of up to 8,192 tokens whole and rejects longer ones.
- **Requests.** Training saw text; JSON states and instructions go through `render` above. Gemma's control tokens
  written in user text (`<start_of_turn>`, `<bos>`) are tokenized as control tokens, as in training.
- **Fixed head.** Choices over 16 options and scores with other than 6 levels are rejected, not truncated or
  rescaled.
- **Temperature.** The checkpoint has no fitted temperature; `calibration.json` ships 1.0 for every type.

## Parity

The shared request set (the edge cases and 40 typed-decisions rows, as JSONL), plus six arbiter cases in
`check.py` (`EXTRA`): the shared set has no 6-level score and no 16-option choice, so these add both, a long state
past the sliding window, and the two rejections of the fixed head.

- **Prompt** (`check.py`, no model needed): the port against the reference prompt, id for id, on the base's
  tokenizer. Measured 2026-10-05 with transformers 4.57.6 and 5.17.0: 127 requests, 6 accepted whole and 121
  rejected by both, 420 rows identical (every question of a rejected request is also compared on its own), 0
  mismatches.
- **Goldens** (`goldens.py`): the reference model in fp32 (LoRA merged, TF32 off, one unpadded row at a time):
  token ids, last position and slots, all 24 slot scores, option logits and probabilities per question.
- **Export** (`parity.py`): ONNX Runtime (LoRA unmerged, rows batched) against the reference.
- **Runtime** (`crates/ollaya-runner/examples/parity_arbiter.rs`): identical rows and rejections, every decision
  the same, all 24 slot scores within 1e-3.

```sh
cd convert
uv run --no-project --with transformers==4.57.6 --with numpy \
    python -m ollaya_convert.families.arbiter.check BASE --requests shared.jsonl
uv run --with peft==0.19.1 --with transformers==4.57.6 \
    python -m ollaya_convert.families.arbiter.export arbiter-4b --out out/arbiter-4b
uv run --with peft==0.19.1 --with transformers==4.57.6 \
    python -m ollaya_convert.families.arbiter.goldens out/arbiter-4b --requests shared.jsonl
uv run --with peft==0.19.1 --with transformers==4.57.6 \
    python -m ollaya_convert.families.arbiter.parity out/arbiter-4b --requests shared.jsonl
cd .. && cargo run --release -p ollaya-runner --example parity_arbiter -- \
    convert/out/arbiter-4b convert/out/goldens-arbiter-4b.jsonl cuda
```

Measured 2026-10-06 on the RTX 4090 machine (transformers 4.57.6 and peft 0.19.1 for the reference; the export
and the runtime on the shared set without `--requests`: 127 requests, 6 accepted whole, 121 rejected by both, 420
rows):

| Check | Device | Rows | Decisions | Slot scores max | Option logits max | Probabilities max |
|---|---|---|---|---|---|---|
| Export (`parity.py`, ONNX Runtime) | CPU | 420 identical | 420/420 | 5.6e-5 | | 9.3e-6 |
| Runtime (`parity_arbiter`) | x86-64 CPU | 420 identical | 420/420 | 6.0e-5 | 5.9e-5 | 1.0e-5 |
| Runtime (`parity_arbiter`) | CUDA, RTX 4090 | 420 identical | 420/420 | 8.5e-5 | 6.4e-5 | 9.2e-6 |

The runtime runs on CUDA with Microsoft's ONNX Runtime 1.28.2 from the CUDA 13 pack, as the GPU runner does. In
the runner, a request of three or more questions takes 154 ms at the median on the RTX 4090 (p95 303 ms, 117
requests of the shared set, one forward pass per question).

## Quality

- **Typed-decisions** (all 400 test states, argmax against the majority label, from the fp32 reference with
  `eval_refs.py`, which the runtime matches above): the fixed head answers **1,200 of the 2,000 questions**. None
  of the 800 score questions has 6 levels (they have 4 or 5), so all of them are rejected. On the 1,200 it
  answers: **0.620** (choice 0.563 on 600, noul 0.677 on 600), ECE 0.149 with no temperature. These numbers are
  not comparable with the other families' full 2,000-question scores.
- **Published benchmarks** (measured on an NVIDIA T4 with the training prompt, one row at a time, not through
  Ollaya). The model card numbers come from the 4-bit base the adapter was trained on. The same adapter and head
  were then run on the same rows with the unquantized base (fp32 compute, the way Ollaya runs it):

| Benchmark | Type | n | 4-bit base (model card) | Unquantized base |
|---|---|---|---|---|
| BoolQ (validation) | noul | 1,000 | 0.849 | 0.853 |
| ARC-Challenge (test) | 4-way choice | 500 | 0.738 | 0.762 |
| CommonsenseQA (validation) | 5-way choice | 500 | 0.706 | 0.720 |
| OpenBookQA (test) | 4-way choice | 500 | 0.722 | 0.748 |

  The unquantized base is equal or better on all four (per-benchmark differences are within sampling noise, all in
  the same direction), so running the full BF16 base in Ollaya does not cost accuracy relative to the model card.

## Limits

- **Options.** Choices of 1..16 options, scores of exactly 6 levels (0..5).
- **Rows.** Up to 8,192 tokens per question, the whole state included; the model was trained on up to 768.
- **Memory.** About 8 GB with the weights kept BF16.

## License and attribution

- **Arbiter** (the LoRA adapter and the 24-slot head): Apache-2.0, by Codekins Pvt Ltd · Zyot Lab.
- **Base model:** Gemma 3 4B IT by Google DeepMind (https://huggingface.co/google/gemma-3-4b-it), under the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). Ollaya fetches the weights
  unmodified and does not redistribute them.
- The catalog's `license` field says both: "Apache-2.0 (LoRA adapter and head) and the Gemma Terms of Use (Gemma 3
  base model)", and the license layer carries the Gemma notice.
- **Training data:** mainly [SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3);
  see the dataset card for its license.

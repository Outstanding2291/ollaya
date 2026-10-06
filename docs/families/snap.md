# snap (`snap-v1`)

**snap1-2b** by logitlab ([logitlab/snap1-2b-GGUF](https://huggingface.co/logitlab/snap1-2b-GGUF),
Apache-2.0) is `openbmb/MiniCPM5-2B` (Apache-2.0) fine-tuned with a rank-16 LoRA and merged, for one job: a
typed decision read from the option letters' logits after one prompt. It is the model of
[snap](https://github.com/emnlmn/snap) (MIT), the author's engine, and it is trained on snap's exact prompt
bytes. Requested in #46.

| Tag | Weights | Temperature |
|---|---|---|
| `snap:2b`, `snap:latest` | `logitlab/snap1-2b-GGUF@39321915` `snap1-2b-q8_0.gguf` (2.7 GB) | 1.0 (raw probabilities, as snap ships them) |
| `snap:2b-q4_k_m` | the same commit, `snap1-2b-q4_k_m.gguf` (1.6 GB), snap's own default file | 1.0 |

The author measured Q8_0 and Q4_K_M the same on typed-decisions (0.655 and 0.654 with snap 0.5.0); Ollaya
defaults to Q8_0 wherever it fits, and the two tags share the prompt, the readout and `calibration.json`.

## Prompt

`crates/ollaya-decision/src/snap.rs` ports snap v0.5.0's prompt compiler (`PROMPT_VERSION` 6, commit
`5e6a65cc`): `src/prompts.rs`, the question checks of `src/schema.rs`, and `resolve_layout` and `compile`
from `src/engine.rs`. snap is Rust, so the TOON renderer and the slot and message builders are snap's code,
adapted (MIT notice in the file).

```text
question_first: QUESTION / instructions / [Answer yes or no. / Yes: ... / No: ...] / "" / OPTIONS /
                A) text / B) text ... / "" / STATE / state / "" / Reply with one letter only.
state_first:    STATE / state / "" / QUESTION ... OPTIONS ... / "" / Reply with one letter only.
ids = tok(pre, specials) ++ tok(user, text only) ++ tok(post, specials)
```

- **State.** A string as it is; anything else in TOON (spec v4.1 as snap pins it): JSON's data model with
  declared array lengths and per-table field lists instead of repeated keys.
- **Layout.** As snap's `layout: auto`, decided for the whole request: the state goes first when it is longer
  than 2,000 bytes, or when the questions' instructions and criteria are shorter than reading the state once
  per question; otherwise the question goes first.
- **Options.** A choice shows its description, or its key when the description is empty or null; a list entry
  without text reads `option i`. A score level shows its index when undescribed. A yes/no question reads
  `A) Yes`, `B) No`, with `{"true", "false"}` criteria as `Yes:` and `No:` lines under the question.
- **Template.** MiniCPM5's chat template around snap's system message, thinking off (`decision.json`),
  checked against the GGUF's own template on the pinned llama-server. Only the template's text may produce
  control tokens; request text stays text.
- **Readout.** The next-token logits of the question's letters, one cold pass per question. snap pools each
  letter's bare, space- and newline-prefixed tokens; Ollaya reads the bare letter, the token the prompt ends
  on.

`crates/ollaya-decision/tests/prompts.rs` checks the port against snap's own `export-prompts` output on the
shared request set, one question per request as the export asks them
(`convert/ollaya_convert/families/snap/fixture.py`): all 573 user messages, layouts and option keys identical,
and the same 20 rejections (14 one-option choices, 3 scores whose criteria are an object, 3 NUL characters).

## Differences from upstream

- **Wide choices.** snap expands a choice wider than the 26 letters into one yes/no probe per option. That is
  not ported: more than 26 options is `TOO_MANY_OPTIONS`.
- **snap's extensions to the wire.** `boolean` and `numeric` questions, the abstain slot, `layout` and
  `expand` are not part of TypeSafe's wire, so Ollaya rejects them.
- **Missing instructions.** Ollaya reads a question without instructions as its name, for every model; snap
  reads no instructions.
- **Readout.** The bare letter only (above).

## Parity (measured 2026-10-05, the RTX 4090 2026-10-06)

Goldens: `python -m ollaya_convert.families.snap.goldens` takes the prompt token ids from snap's own export
(the author's code, not a port) and evaluates them on a stock `llama-server` of the pinned build (b11146) with
the GGUF, one cold pass per question. It first checks that the GGUF's chat template gives snap's frame and that
the pinned build tokenizes every exported prompt to snap's ids. 593 cases: 573 questions, 17 rejected as
invalid and 3 as `TOO_MANY_OPTIONS`; 3 more are rejected by the API before any engine sees them.

| Model | Device | Token ids | Decisions | Option logits max | Probabilities max |
|---|---|---|---|---|---|
| 2b Q8_0 | x86-64 CPU | 573/573 identical to snap's | 573/573 | 7.4e-6 | 1.9e-6 |
| 2b Q8_0 | CUDA, RTX 5090 | 573/573 identical to snap's | 573/573 | 7.4e-6 | 1.9e-6 |
| 2b Q4_K_M | x86-64 CPU | 573/573 identical to snap's | 573/573 | 7.4e-6 | 1.9e-6 |
| 2b Q4_K_M | CUDA, RTX 4090 | 573/573 identical to snap's | 573/573 | 7.1e-6 | 1.9e-6 |
| 2b Q8_0 | CUDA, RTX 4090 | 573/573 identical to snap's | 573/573 | 7.3e-6 | 1.9e-6 |

The RTX 4090's reference is the same prompts replayed on stock `llama-server` b11146 on that GPU
(`llm_common.replay`), since CUDA rounds differently from one GPU to another: against the RTX 5090's goldens
the RTX 4090 differs by up to 0.28 in option logits and on 8 of 573 decisions, measured and not gated.

Five questions take 84 ms in the runner on the RTX 5090 and 65 ms on the RTX 4090 (p50 of 20 requests,
`parity_llama --latency`), one cold pass per question; Q4_K_M takes 62 ms on the RTX 4090 and 2.1 s on the
x86-64 CPU. snap shares the state's prefix between questions; the
author reports 48 ms on an RTX 4090 with Q4_K_M.

RTX 4090, HTTP API (Ollaya 0.10.0, Q8_0 on CUDA): the triage preset (five questions, 606 prompt tokens) takes
68 ms at the median of 15 warm requests on one message, and 67 ms over 20 different messages
(`results/runs/2026-10-06-latency-rtx4090-cuda-extra.json`).

## Quality

- **Typed-decisions** (all 400 test states, 2,000 questions, argmax against the majority label, through
  Ollaya's runtime: `logits` example and `quality_runtime.py`): **0.648**, ECE 0.062 at temperature 1. By type:
  choice 0.655, score 0.600, noul 0.703. Q4_K_M (`snap:2b-q4_k_m`, measured 2026-10-06 the same way on CUDA):
  **0.660**, ECE 0.047; choice 0.653, score 0.585, noul 0.767. The 25-question gap between the two files is
  within what quantization moves on this set; the author's own runs put them 1 question apart.
- **The author's figure** with snap itself: 0.655 on Q8_0 (snap 0.5.0, Apple M1 Max). Ollaya's runtime is 15
  questions lower. The prompts are identical, so the gap lies in reading the answer: snap pools each letter's
  space- and newline-prefixed tokens and evaluates the state once for all questions, Ollaya reads the bare
  letter in one cold pass per question. The author reports that the checkpoint never trained on typed-decisions.

## Limits

- **Options.** 2 to 26 per choice and per score; a wider choice is `TOO_MANY_OPTIONS`.
- **Context.** 8,192 tokens per question, snap's default; a longer prompt is rejected, not cut.
- **Memory.** About 3 GB for the weights plus the context.
- **Languages.** English and Italian, the model card's.

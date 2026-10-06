# Intel Arc 140T parity measurements

Measured on 2026-09-27 and 2026-09-28 for [PR #27](https://github.com/ollaya-dev/ollaya/pull/27).
Same-backend Vulkan parity passes for both measured models, which is Ollaya's gate for a device.

## Setup

- Windows x64; Intel Core Ultra 9 285H; 64 GiB system RAM.
- Intel Arc 140T integrated GPU, `Vulkan0`, driver `32.0.101.8860`.
  Vulkan reports about 37 GiB accessible memory. This is shared system memory, not a separate VRAM pool.
- Ollaya revision `5ad1f9f61d4d46b15d9f937be536aab597575fa7`, after merging main `91f873c02fd4b8f95649c49e2aeb43f146a7fe7a`.
  CUDA comparison runs used the same runner executable at checkout revision
  `80ffebb898a1c327d011d2750b8f1f03c59319ea`; the intervening commit changed documentation
  and reference-fixture filenames, not Rust inference code.
- Stock `llama-b11146-bin-win-vulkan-x64.zip`: version `0.5.0-dev`, build 11146,
  commit `7fe450e19305b828c199d602c23a8337aaa1f03b`, Clang 20.1.8.
  The native parity runs used the stock libraries, not the local diagnostic build described below.
- Python reference encoder: laya 0.3.7. The default 40 typed-decisions rows and the full edge-case set were used.

| Model | Pinned author GGUF | Context | Plan | Temperature |
| --- | --- | ---: | --- | ---: |
| Winnow-E4B | `EldanRing/Winnow-E4B`, revision `734302fe5fbfeb3f21a7ece62653c9539be4aaf3`, `gguf/Winnow-E4B-Q8_0.gguf` | 8192, full sliding-window cache | prefix | 1.2574172017327816 |
| JevK5 | `alibiserikbay/JevK5-GGUF`, revision `ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a`, `jevk5-4b-v0.3-Q8_0.gguf` | 16384 | cold | 1.22 |

GGUF SHA-256:

- Winnow-E4B: `840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2`.
- JevK5: `aea433883bc7ed399f2fbd539e53d2eac7caf71a946fe6650995a413979d4a30`.

## Same-backend native parity

The reference prompt encoder ran through stock llama-server on **Vulkan0**, using the fixed evaluation plan.
`parity_llama` then compared Ollaya on **Vulkan0** with those fixtures, including prompt IDs,
split points, label candidates, state token counts, truncation and rejected requests.
Logit differences below are measured after log-softmax over the options, as in the parity tool.

| Model | Cases | Rejected cases | Matching decisions | Maximum logit difference | Maximum probability difference | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Winnow-E4B | 123 | 15 | 505/505 | 1.143e-5 | 2.299e-6 | PASS |
| JevK5 | 123 | 4 | 593/593 | 9.521e-6 | 1.563e-6 | PASS |

Fixture SHA-256:

- Winnow Vulkan: `de45022c172f5f590e3ca18e461102babeae48e73868d9eb294ec42aa4a57011`.
- JevK5 Vulkan: `2db2417a9c00ef5da3e7cd92838df6924dc285dbee6a86db1576ee64ff498aa0`.

Winnow used 108 prefix steps, 397 barriers and 505 question steps. JevK5 used 579 cold steps;
single-option questions require no inference. These passes establish fidelity to the same backend's
reference. They do not establish matching outputs between Vulkan and CUDA.

## Against CUDA (information, not a gate)

The maintainer supplied the [exact CUDA reference](https://gist.github.com/cobanov/c808f61976a8210235d8faf8ae91aa15)
in [PR comment 5864110141](https://github.com/ollaya-dev/ollaya/pull/27#issuecomment-5864110141).
Its SHA-256 was verified as `b0b15b5646e8cda17364b6097a9613eb231a88a4813b52c129423313447695fe`.
The supplied decision and calibration JSON files exactly match the registry files used above.
The reference comes from stock b11146 CUDA on an RTX 4090, Linux, 2026-09-26.

| Arc implementation compared with CUDA | Matching decisions | Maximum logit difference | Maximum probability difference |
| --- | ---: | ---: | ---: |
| Stock Vulkan llama-server, full 123 cases | 502/505 | 0.32614444 | 0.05600171 |
| Ollaya Vulkan runner, full 123 cases | 502/505 | 0.3261 | 0.05600 |

Both full comparisons cover the same 505 questions and 15 rejected cases. The stock comparison
found no differences in prompt IDs, split points, candidates, wire order, state token counts or
truncation. The runner reported the same decision changes and numerical range as stock
llama-server, so the difference is the backend's rounding, not Ollaya's. Parity is gated per device
(docs/decisions/0003-llama-cpp-runtime.md, point 7): differences between backends are measured and
documented, as here, not gated. The x86-64 CPU differs from CUDA by a similar amount on the same model.

## Reproduction

From `convert/`, use the pinned author GGUF and the stock server from build b11146:

```powershell
uv run python -m ollaya_convert.families.llm_common.export_llama winnow `
  --server $SERVER --gguf $WINNOW --slug arc-vulkan-native `
  --repo EldanRing/Winnow-E4B --revision 734302fe5fbfeb3f21a7ece62653c9539be4aaf3 `
  --file gguf/Winnow-E4B-Q8_0.gguf --temperature 1.2574172017327816 `
  --upstream-commit 77d14580c6732ca2f3745750c1dc1fd446d8bcee `
  --n-ctx 8192 --device Vulkan0 --port 8098

uv run python -m ollaya_convert.families.llm_common.export_llama jevk5 `
  --server $SERVER --gguf $JEVK5 --slug arc-vulkan-native `
  --repo alibiserikbay/JevK5-GGUF --revision ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a `
  --file jevk5-4b-v0.3-Q8_0.gguf --temperature 1.22 `
  --n-ctx 16384 --device Vulkan0 --port 8097
```

Place the matching GGUF as `model.gguf` beside each generated `decision.json` and `calibration.json`.
From the repository root, set `OLLAYA_LIBRARY_PATH` to the stock install's `lib/ollaya` directory:

```powershell
cargo run --release -p ollaya-runner --example parity_llama -- `
  convert/out/winnow-arc-vulkan-native convert/out/winnow-arc-vulkan-native/goldens-vulkan.jsonl Vulkan0
cargo run --release -p ollaya-runner --example parity_llama -- `
  convert/out/jevk5-arc-vulkan-native convert/out/jevk5-arc-vulkan-native/goldens-vulkan.jsonl Vulkan0
```

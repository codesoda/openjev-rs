# Runtime results

## M2 — exact pinned GGUF smoke

Status: **targeted M2 remediation is implemented and all three exact artifacts now pass the create-only offline Metal and true-CPU smoke captures; final parent/Astra verification remains required before commit.** MiniCPM5-2B and Qwen3.5-4B have `exact` template status. Qwen3-0.6B remains a nonidentical template pair and has the narrower `reviewed-equivalent` status approved by parent/Astra for exactly two string system/user messages, no tools, `add_generation_prompt=true`, and `enable_thinking=false`. This does not relabel the templates identical or approve tool, multimodal, assistant-reasoning, or arbitrary multi-turn behavior.

### Host and native builds

- MacBook Pro `Mac15,6`, Apple M3 Pro (11 CPU / 14 GPU cores), 18 GB RAM, arm64, macOS 26.2 (25C56).
- Rust/Cargo 1.95.0, CMake 3.28.2, Apple clang 17.0.0, Xcode developer directory `/Applications/Xcode.app/Contents/Developer`.
- Native pin: `llama-cpp-2 = llama-cpp-sys-2 = 0.1.156`, bundled llama.cpp `e79e4bf660e19f2ad851e06c6913f7a8c5852621`.
- Metal target: `--features metal`, separate `target-m2-metal`, `GGML_METAL=ON`, `GGML_OPENMP=OFF`, all GPU layers requested. CMake cache confirmed Metal ON. Native logs proved Apple M3 Pro `MTL0` and offload of 29/29, 43/43, and 34/34 layers respectively.
- True CPU target: `--features native`, separate `target-m2-cpu`, `GGML_METAL=OFF`, `GGML_OPENMP=OFF`, zero GPU layers requested, `offload_kqv=false`, `op_offload=false`. CMake cache confirmed Metal OFF; runtime device enumeration contained CPU only.
- The safe wrapper does not expose an actual-offloaded-layer count. Historical M2 smoke rows therefore keep `gpu_layers_actual: null`; actual Metal offload evidence is retained in selected native stderr rather than fabricated in metadata. M3 resolved production schema semantics with nullable actual plus explicit status, while leaving these historical captures unchanged.

Full configuration is in [`results/m2-config.json`](results/m2-config.json). The original Metal run occurred first; the original CPU run then used `--offline` and the already verified canonical artifacts. The remediation captures on both devices also used `--offline` and reused the same target directories and model cache.

### Artifact/cache evidence

Canonical root was `/Users/chrisraethke/.cache/openjev`, with hf-hub's content cache under `hub/` and atomic verified receipts under `openjev/receipts/`. Observed execution history: before the original transfer, the exact revision/file paths were checked under `~/.cache/huggingface/hub` and none was present; each exact artifact was then transferred into the canonical cache (5,213,792,864 manifest bytes combined). The retained offline captures prove verified reuse, not an independent transport-level transfer count.

Every cache resolution and every owner-worker model load recomputed both complete byte length and SHA-256. Cache inputs now reject unsafe repository and filename components before filesystem work. A mismatch returns an integrity error without repair. Explicit repair quarantines corrupt owned bytes and rebuilds owned snapshots, but bypasses corrupt external HF sources without modifying them; offline repair reuses only a valid alternate and otherwise reports `OfflineMiss`. Receipts are published only after final verification, with atomic replacement on Unix, and are invalidated before incomplete repair. Tiny-file tests cover absolute/traversal/Windows/network forms with an outside file unchanged, normal HF `../../blobs/hash` symlink import, dangling destinations, corrupt external repair/fetch, corrupt owned blob plus valid external reuse, failed-repair receipt absence, healthy no-fetch behavior, and true cross-process one-fetch contention behind the OS lock. No test contains weights.

| ID | Pinned bytes | Verified SHA-256 |
|---|---:|---|
| `qwen3-0.6b` | 639,446,688 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| `minicpm5-2b` | 1,561,318,368 | `ec2d5801640099e97d8d7e8003ad4d81f336e757811f03a26173dddf386602fd` |
| `qwen3.5-4b` | 3,013,027,808 | `13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983` |

No GGUF was copied into the repository or a test temporary directory.

### Load/template/token/readout results

Each worker owned backend and model on one thread. Contexts borrowing the model were created and dropped inside that thread for each pass. The direct helper used `AddBos::Never`, exact nondeprecated `token_to_piece_bytes(..., 32, false, None)` round trips, all slot collision/vocabulary/append-boundary checks, final chunk-local `get_logits_ith` indexing, an immediate owned copy of full-vocabulary f32 logits, then the core f64 readout. Each model ran a warmup context and a second reported context at actual `n_ctx=4096`, `n_batch=512`, `n_ubatch=512`.

The original create-only captures are retained unchanged and record the pre-adjudication Qwen mismatch. Their timing columns contain `load_seconds / warmup_forward_seconds` (not the separately reported measured `forward_seconds`):

| ID | Architecture | trained `n_ctx` | vocab | GGUF template SHA-256 | Original template result | Metal load / warmup forward (s) | CPU load / warmup forward (s) | Slots |
|---|---|---:|---:|---|---|---:|---:|---|
| `qwen3-0.6b` | `qwen3` | 40,960 | 151,936 | `57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361` | **mismatch / adjudication required** versus pinned native `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` | 0.278 / 0.0666 | 1.029 / 15.1702 | A/B/C = 32/33/34 |
| `minicpm5-2b` | `llama` | 131,072 | 130,560 | `cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c` | exact match | 0.241 / 0.1711 | 1.932 / 20.0632 | A/B/C = 54/55/56 |
| `qwen3.5-4b` | `qwen35` | 262,144 | 248,320 | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` | exact match | 0.593 / 0.3355 | 4.567 / 28.6039 | A/B/C = 32/33/34 |

All six reported Metal/CPU readouts were finite, selected the first option, and included full-vocabulary normalizers and allowed-token mass. Exact JSONL rows are in [`results/m2-metal-smoke.jsonl`](results/m2-metal-smoke.jsonl) and [`results/m2-cpu-smoke.jsonl`](results/m2-cpu-smoke.jsonl). These original reproducibility captures both used `OPENJEV_INTEGRATION=1 --offline` after the observed initial transfers, and all six rows report `cache_hit=true`; they prove verified reuse, not transfer counts. Selected stderr evidence is in the adjacent `.stderr.txt` files. Large repetitive native logs were not retained in git; each reduced file records the raw capture's SHA-256 and original line/byte count. llama.cpp's 15 Qwen3.5 warnings about unused block-32 MTP/next-token tensors are preserved in the reduced evidence as an observation; M2's finite direct readout passed, while M3 parity remains unclaimed.

The original Qwen mismatch was a metadata-template identity issue, not a load failure: the exact GGUF loaded and scored on both devices, while the restricted hand profile used the pinned native rendering. That history is not erased. Parent/Astra subsequently adjudicated the two nonidentical fixture templates as equivalent only for the restricted profile. The manifest record is keyed to artifact `9465e63a…031`, GGUF template `57f1fd00…d0361`, and native profile `a55ee1b1…74d8`; unseen hashes still fail explicitly. `fixtures/qwen-template-equivalence.json` and the Jinja 3.1.4 oracle record 144 authored + 108 perturbation matches, all 252 reference prompt hashes, and four edge states.

New create-only final captures are [`m2-metal-smoke-final.jsonl`](results/m2-metal-smoke-final.jsonl) and [`m2-cpu-smoke-final.jsonl`](results/m2-cpu-smoke-final.jsonl), with reduced stderr beside them and configuration in [`m2-final-config.json`](results/m2-final-config.json). All six rows have `outcome=passed`, `cache_hit=true`, finite readouts, and `gpu_layers_actual=null` because the safe wrapper cannot expose that count. Qwen reports `reviewed-equivalent`; the other two report `exact`.

| ID | Remediation template status | Metal load / warmup / measured forward (s) | CPU load / warmup / measured forward (s) |
|---|---|---:|---:|
| `qwen3-0.6b` | `reviewed-equivalent` | 0.2883 / 0.0709 / 0.0664 | 1.0246 / 13.8246 / 14.1127 |
| `minicpm5-2b` | `exact` | 0.2880 / 0.1741 / 0.1647 | 1.9301 / 16.6437 / 15.1409 |
| `qwen3.5-4b` | `exact` | 0.7456 / 0.3294 / 0.3108 | 9.3541 / 27.7240 / 26.7048 |

### Commands

```sh
OPENJEV_INTEGRATION=1 GGML_METAL=ON CARGO_TARGET_DIR=target-m2-metal \
  cargo run -p openjev-llama --features metal --example m2_smoke -- \
  --all --offline --device metal --gpu-layers all

OPENJEV_INTEGRATION=1 GGML_METAL=OFF CARGO_TARGET_DIR=target-m2-cpu \
  cargo run -p openjev-llama --features native --example m2_smoke -- \
  --all --offline --device cpu --gpu-layers 0
```

These rows are a small M2 smoke only. They are not M3 authored144 prompt/token/logit parity, not shared/batch execution, and not performance benchmarks. The M3 evidence below supersedes the earlier statement that the exact Qwen gate was unrun; it does not change these M2 captures.

## M3 — production direct readout and strict Qwen parity

Status: **implemented locally; the exact authored144 gate passed and Astra accepted the measured numerical baseline. A targeted `cache_hit` metadata fix now awaits parent/Astra re-review before commit.** M4 CLI, M5 shared/batch execution, and M6 performance/eval commands were not implemented.

Production `EngineHandle::score_direct` now returns a validated full `openjev-readout-v1` from one clean prompt prefill per decision (chunking allowed), with no per-row warmup and no generation. All option slots are single-token/ASCII/unique/in-vocabulary and append-boundary checked before decode. Forward timing spans decode plus synchronized `get_logits_ith`/logit copy; total timing includes prompt rendering, tokenization/slot validation, context construction, f64 readout, and metadata construction. Every direct call creates a fresh context and prefills the complete prompt, so production readouts now report inference `cache_hit=false` regardless of whether the GGUF artifact was already in the download cache. Artifact cache status remains separate cache/runner metadata. The M2 `smoke_direct` path remains deliberately two-pass (warmup plus measured).

Each retained row reports the pinned GGUF repo/revision/file, artifact SHA-256, Q8_0 quantized/mixed dtype, native BF16 source/revision, llama wrapper/native commit, prompt profile/hash/version, token/slot IDs, raw f64-stored logits converted from native f32, probabilities, full-vocabulary statistics, forward/total timing, and actual execution configuration. Metal requests all layers, while `gpu_layers_actual=null` and `gpu_layers_status=unavailable` honestly record the safe-wrapper gap; reduced native stderr proves 29/29 layers offloaded. CPU production metadata uses actual `0` only when the native CPU path requests zero and disables offload. The schema/PLAN now normatively permit integer-or-null actual layers and require a status; no requested layer count is re-labelled as actual.

### Exact gate

Command (offline, existing M2 cache and `target-m2-metal` only):

```sh
OPENJEV_INTEGRATION=1 GGML_METAL=ON CARGO_TARGET_DIR=target-m2-metal RUST_LOG=info \
  cargo run -p openjev-llama --features metal,integration --example m3_parity -- \
  --device metal \
  --authored-output docs/results/m3-qwen3-metal-authored144.predictions.jsonl \
  --perturbations-output docs/results/m3-qwen3-metal-perturbations108.predictions.jsonl \
  --report docs/results/m3-qwen3-metal-report.json
```

All fixture/reference inputs and create-only output paths were validated before model load. The runner indexed all 252 unique committed BF16 prediction rows by semantic ID, selected the exact 144 authored IDs and 108 perturbation IDs, rejected duplicates/missing IDs, and never zipped by file position.

| Set | Rows | prompt SHA exact | input tokens exact | option IDs exact | answer token IDs exact | all-slot boundaries | finite readouts |
|---|---:|---:|---:|---:|---:|---:|---:|
| mandatory authored144 | 144 | 144 | 144 | 144 | 144 | 144 | 144 |
| extended perturbations108 | 108 | 108 | 108 | 108 | 108 | 108 | 108 |

Any mismatch is a hard runner error; there is no threshold, skip, or xfail. An opt-in integration test independently loaded the cached Qwen artifact and passed encoded prompt/hash/token/slot validation for all 144 authored rows. Ordinary workspace tests compile zero M3 integration tests and never access a model or network. Unit regressions prove that altered Python JSON spacing, an inserted BOS token, and an absolute rather than last-chunk-local logits index are detected without modifying production behavior.

### Measured Q8_0 versus native BF16 delta

No guessed 98% criterion is asserted. The measured comparison is:

| Set | first-argmax agreement | Logit MAE / RMSE / max abs | Probability MAE / RMSE / max abs |
|---|---:|---:|---:|
| authored144 | 140/144 = 0.972222 | 0.534313 / 0.664228 / 2.543209 | 0.023046 / 0.068618 / 0.513266 |
| perturbations108 | 107/108 = 0.990741 | 0.565334 / 0.721156 / 2.591896 | 0.014599 / 0.056621 / 0.542852 |

Authored mismatch IDs and BF16/local first-choice probability margins:

- `533d4423d311de82b27d`: BF16 `B`, margin 0.358355; local `A`, margin 0.006306.
- `e1d610bd14e3d16c09b9`: BF16 `B`, margin 0.634824; local `A`, margin 0.391645.
- `6db17cc22bd656d78558`: BF16 `prohibited`, margin 0.194460; local `permitted`, margin 0.126730.
- `b8c5b9b285cbdddc9f8d`: BF16 `B`, margin 0.185146; local `insufficient`, margin 0.339048.

Extended mismatch: `0b580003cbe6b93cefd6`, BF16 `A` margin 0.462085 versus local `B` margin 0.623574. This is an empirical comparison of the pinned llama.cpp backend/Q8_0 GGUF against committed native-BF16 reference rows. It is not a same-artifact backend test, not a numerical-equivalence claim, and does not establish quantization as the only cause of the differences. Astra accepted 140/144 with logit MAE 0.534313, plus the 107/108 extended result, as the measured baseline; no 98% gate or other guessed tolerance applies.

### Retained M3 artifacts

Evidence note: the original 252 create-only production rows are preserved byte-for-byte and all incorrectly contain `cache_hit=true`. That historical field reflects the now-fixed defect that copied download-cache status into inference metadata; it must not be read as prefix reuse. The raw logits, probabilities, exact prompt/token gates, numerical comparison, file sizes, and hashes are unaffected. The corrected production path and runner assertion require `cache_hit=false`; the full numerical benchmark was not rerun or silently rewritten for this metadata-only fix.

- `m3-qwen3-metal-authored144.predictions.jsonl`: 144 rows, 388,242 bytes, SHA-256 `95bff7a6b8a4fcd73deffd2db7de7f88900530726264096266b0a9cf8976bc67`.
- `m3-qwen3-metal-perturbations108.predictions.jsonl`: 108 rows, 291,218 bytes, SHA-256 `4272e100f23774e93bbaeaab740c697562519185bcde4701b0fdf5dd36f7103a`.
- `m3-qwen3-metal-report.json`: 6,500 bytes, SHA-256 `ef131a91c09ab21f2d4017203834436b44f8ef96f8775434d7111aab13e11edf`.
- `m3-qwen3-metal-run.stdout.json`: one JSON summary object only, 615 bytes, SHA-256 `0ac098f920fe69c54f652a53af6dad8cd008247f00bbc330777eba329f3b60a7`.
- `m3-qwen3-metal-run.stderr.txt`: reduced native stderr, 3,119 bytes, SHA-256 `819b7cd85667df45f61ad0cb5b4ec797661e8ceeb4f022b1f68cc14c15057333`, with command/device/offload/context evidence and the original raw capture hash/size; native logs never entered stdout.

The report also records hashes/sizes for authored144, perturbations108, and the 252-row reference; complete model/config metadata; exact-gate counts; mismatch IDs; and both reference/local margins. Files were created with create-new semantics. No model was downloaded, no new native target directory was made, and `reference/` remains unchanged.

# Runtime results

## M2 — exact pinned GGUF smoke

Status: **approved and committed. All three exact artifacts pass the create-only offline Metal and true-CPU smoke captures.** MiniCPM5-2B and Qwen3.5-4B have `exact` template status. Qwen3-0.6B remains a nonidentical template pair and has the narrower `reviewed-equivalent` status approved by parent/Astra for exactly two string system/user messages, no tools, `add_generation_prompt=true`, and `enable_thinking=false`. This does not relabel the templates identical or approve tool, multimodal, assistant-reasoning, or arbitrary multi-turn behavior.

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

Status: **approved and committed. The exact authored144 gate passed, Astra accepted the measured numerical baseline, and the targeted direct `cache_hit=false` correction passed final parent/Astra review.** The sentence that M4 CLI was absent describes the M3 commit boundary; current M4 evidence is recorded below.

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

## M4 — production CLI and serial fallback

Status at the M4 boundary: **approved and committed as `a44b805` after targeted Astra-remediation gates passed.** The evidence below describes the M4 boundary, before M5 added receipt-gated KV copy and independent sequence packing.

The release Metal CLI was built in the existing `target-m2-metal` directory and exercised only the already verified cached Qwen3-0.6B artifact with `--offline`. Seven create-only probe cases covered direct `decide` with confidence, stdin Noul, structured-state Score with explicit finite values, stdin `ask`, ordered two-row `run`, repeated-question automatic shared request with serial fallback, and `--require-shared` refusal. Every success stdout line parsed as exactly one JSON object; native llama logs stayed on stderr. Direct/serial-full-prompt rows reported the production readout string and `cache_hit=false`.

The repeated-question case emitted two rows in order with `requested_mode=shared`, `effective_mode=serial`, fallback reason `shared execution is not implemented or probed until M5`, and an stderr warning despite `--quiet`. `--require-shared` exited 2 with empty stdout and a structured `unsupported` stderr record before native load. No shared/batch probe receipt or performance claim was created.

Primitive checks:

- Noul emitted `primitive=noul`, ordered `yes/no` distribution and numeric `p_yes`.
- Score emitted finite `level_values`, unchanged distribution, finite expectation and argmax level.
- Confidence was opt-in and labelled `normalized margin; uncalibrated`.
- Every observed production row retained `gpu_layers_actual=null` / `gpu_layers_status=unavailable` for Metal rather than copying the all-layers request into an actual count; native stderr remained the offload evidence.

Model commands were separately exercised. `models list` returned `openjev-models-v1` and recomputed complete size/SHA verification for all three canonical cached artifacts; every row was `cached=true`, `verified=true`, with shared/batch status explicitly `not-implemented-or-probed-until-m5`. `models path qwen3-0.6b` and offline `models pull qwen3-0.6b` returned `openjev-model-path-v1` JSON envelopes with the verified path/hash—not bare strings—and no network request. The opt-in process test also scored the cached Qwen file as a caller-hashed custom local artifact and observed `source=local`, `integrity=caller-sha256`, `template_override=true`, `template_status=override-unverified`, and no `native_reference`.

Automated process coverage includes JSON help/version for every command surface, parse-aware nested command/usage metadata even when global option values equal command names, piped examples, usage and backend-disabled exits, explicit-state no-stdin-blocking, fatal JSONL parse before backend, model-list envelopes, broken-pipe nonpanic behavior, deterministic model-free primitive/fallback injection, runtime-row failure continuation, create-only no-overwrite/empty-on-startup-failure policy, and owner-thread clean/panic join handling. Injected scorer/sink tests prove each row is visible and flushed before the next score and that output failure stops later calls while shutting down. The native opt-in suite used a long but valid row under `--max-tokens 128` to produce success/error/success JSONL in input order and exit 1, and a separate 20-row cached-Qwen process observed the first output-file row before the process completed.

Custom Hub cache tests now exercise the shared registered/custom pre-download containment preflight. Tiny files prove hf-hub's root, `.locks`, repository, blob, snapshot/nested-filename and negative-cache parents are canonical owned directories before an injected downloader can run; every escape case records zero downloader calls. Offline resolution rejects an external snapshot target without modifying it, while a normal mock download and offline owned-path reuse pass with caller SHA-256 and discovered byte length. No test contacts the network or downloads weights.

Retained create-only evidence:

- `m4-cli-metal-probes-final.json`: 5,627 bytes, SHA-256 `8bbbfda271fcd5b43d32b15ae8740c5d86de42e141c8a2b82ea6aaad0a4cee0f`; final seven cases, exact stdout/stderr hashes/counts and repeated-question group identity.
- `m4-cli-metal-probes-final.stderr.txt`: 3,252 bytes, SHA-256 `9e909b1bd8ac96b1028af057e0a33ce8ee1542da95f4813ff4b5642a305b2dba`; concise selected warning/load/offload lines plus raw stderr hashes and sizes.
- `m4-models-metal-probes-final.json`: 5,042 bytes, SHA-256 `b3c096b4600384d5e973cf2dffa8a6c8ce6684dd27bbcf2d864d108d035b3a8a`; final list/path/offline-pull envelopes, including honest `registered-runtime-load-not-attempted-by-list` support status.
- `m4-native-cli-tests-final.txt`: 692 bytes, SHA-256 `dfc9a28153c42df9eda77d7a10a563dcf8ef68385ea49f020c09cb95ac9d0329`; three opt-in release process tests passed before the final nested-help validation addition.
- `m4-native-cli-tests-final2.txt`: 593 bytes, SHA-256 `3de4298277c1e9264317ad058ca8fb4a373da9dbd6ac7865fc3d624b258c76c2`; final post-change opt-in release rerun, 3/3 process tests passed in 9.48 seconds.
- `m4-native-cli-tests-final3.txt`: 760 bytes, SHA-256 `1e55eb8c870cc792ca2add8e9b81b9a096726040f2e556518bfd7209b83b4f73`; create-only targeted-remediation rerun, 4/4 release Metal process tests passed in 9.39 seconds, including first-row file visibility before process completion.

The corresponding earlier M4 files are immutable passing captures retained for chronology. The `final3` test capture supersedes but does not overwrite `final2` after custom Hub mutation containment, true streaming run output, and clap-tree help metadata remediation; the earlier probe/model captures remain the latest probe evidence.

No GGUF was downloaded or copied, no new native target directory was created, and no all-model or long true-CPU inference smoke was rerun. The existing `target-m2-metal` and `target-m2-cpu` directories were reused for feature clippy/tests; actual new M4 inference was Qwen/Metal only. Eval, bench, calibration, permutation/temperature transforms, shared KV copy, probe eligibility, and packed batch timing remain explicitly unimplemented.

## M5 — shared/batch eligibility probes

Status: **the real shared-KV and independent packed-batch paths are implemented, but all twelve exact configurations probed on this host failed the frozen numerical gates. No tested profile is enabled; standard scoring visibly uses fresh serial full prompts.** This is a fail-closed result, not a shared/batch performance claim.

The exact configuration key includes artifact SHA-256, `llama-cpp-2/llama.cpp` pin, probe-suite version, device, GPU-layer request, KQV/op-offload booleans, thread count, explicit/automatic context settings, `n_batch`, `n_ubatch`, `n_seq_max`, unified KV and prompt profile. The retained runs used the same host and native pins documented for M2, default 11 threads, automatic context bounded by 4,096 request tokens and 32,768 context tokens, 512 batch/ubatch, and 32 sequences. Metal requested all layers with KQV/op offload; true CPU requested zero layers and disabled both offloads.

Frozen acceptance requires all of:

- maximum absolute selected-slot logit delta `<= 0.001`;
- maximum probability delta `<= 0.0001`;
- identical first argmax.

Each subprocess began with a one-branch binary case, then a ragged two-branch 3-way case that forces multiple chunks (`n_batch=512`). Later planned cases are a long-state 21-branch 16-way case, changed-state isolation, and a `n_seq_max + 1` copy/clear case that forces sequence-ID reuse across waves. After a decisive failure, later cases are explicitly recorded `unrun-after-decisive-failure`; they are not silently marked passed. Failed receipts are retained locally and reject production eligibility.

| Model | Device | Mode | max slot-logit delta | max probability delta | same first argmax | Decisive case |
|---|---|---|---:|---:|---|---|
| Qwen3-0.6B | Metal | shared | 0.03468895 | 0.00002451 | yes | ragged 2-branch |
| Qwen3-0.6B | Metal | batch | 0.05129051 | 0.00007275 | yes | ragged 2-branch |
| MiniCPM5-2B | Metal | shared | 0.01604462 | 0.00046002 | yes | ragged 2-branch |
| MiniCPM5-2B | Metal | batch | 0.02395439 | 0.00065809 | yes | ragged 2-branch |
| Qwen3.5-4B | Metal | shared | 0.00438118 | 0.00007469 | yes | ragged 2-branch |
| Qwen3.5-4B | Metal | batch | 0.00293541 | 0.00051441 | yes | ragged 2-branch |
| Qwen3-0.6B | CPU | shared | 0.81830978 | 0.17236975 | yes | binary 1-branch |
| Qwen3-0.6B | CPU | batch | 0.84081841 | 0.00125196 | yes | ragged 2-branch |
| MiniCPM5-2B | CPU | shared | 0.59481430 | 0.17811387 | yes | binary 1-branch |
| MiniCPM5-2B | CPU | batch | 0.34326744 | 0.03522472 | yes | ragged 2-branch |
| Qwen3.5-4B | CPU | shared | 0.10205460 | 0.01284628 | yes | binary 1-branch |
| Qwen3.5-4B | CPU | batch | 0.49478531 | 0.03544527 | yes | ragged 2-branch |

The exact JSON reports are under [`results/m5/`](results/m5/). The twelve finalized `*-final.json` reports include `probe_suite_version` in the configuration identity; each has `process_status=completed`, `enabled=false`, a self-consistent failed receipt, observed deltas, explicit case statuses, and a failure reason. They total 34,284 bytes. The twelve earlier same-outcome files without `-final` are preserved as pre-final evidence, but their configuration payload omitted the suite-version identity and they are not valid eligibility receipts under the finalized implementation. Finalized SHA-256 values:

| Report | Bytes | SHA-256 |
|---|---:|---|
| `qwen3-0.6b-metal-shared-final.json` | 2,830 | `f66138122a5d4bdc938df8fe6ed643e6ee5620aa27c510405f838b14e952d06b` |
| `qwen3-0.6b-metal-batch-final.json` | 2,821 | `452a97ce9288fe484134e9fbad162e4f0a20f0b5729c4bc9f670f169a810c8f6` |
| `minicpm5-2b-metal-shared-final.json` | 2,818 | `761806695773e358f3ed9b2be0cee17c647a1fdb4934fd1a87f62817c42f36af` |
| `minicpm5-2b-metal-batch-final.json` | 2,825 | `5816b275b8aadcf594560f870f2622f3bd05f8300170ffe5695a716a2ef3683d` |
| `qwen3.5-4b-metal-shared-final.json` | 2,840 | `446c89092061be2d694cc02f950b5020c05ae12e6c6e1e442e944bfdd680b941` |
| `qwen3.5-4b-metal-batch-final.json` | 2,831 | `6296c670b4e071e25b85ff7c396fcf0442846032d081c9bd7cdb1e0f8e31b4bc` |
| `qwen3-0.6b-cpu-shared-final.json` | 2,970 | `4019132169fa877b82359d733633540eea0d20c8eb04327ce519231735e5bbec` |
| `qwen3-0.6b-cpu-batch-final.json` | 2,801 | `c7468de7c67a90ec03e66d88cb2745ab46a2957165e896b2995ea0bb8d7dedb7` |
| `minicpm5-2b-cpu-shared-final.json` | 2,974 | `3673497147cc3b6de0ef88434807f5765358a0e15404b46a60fb16b70aa5b0a4` |
| `minicpm5-2b-cpu-batch-final.json` | 2,797 | `f962d1aba4ec62f53f5c91929c56b2f73a5897eda23ffa11d61fef5450bc6d7f` |
| `qwen3.5-4b-cpu-shared-final.json` | 2,990 | `5499f261269dcba9d99ecb8981c63b81e408e89e0486ca8ef825224e31476687` |
| `qwen3.5-4b-cpu-batch-final.json` | 2,787 | `04234d7eaac0a836eddebca9f2f8d6c595be9c8149f1fef4fc57816becaf2c63` |

The first shared failure on true CPU is large, but code and pinned-source inspection found no concrete token, position, device, offload, context-size, or readout-index mismatch: full prompts and shared suffixes use the same token IDs and absolute positions, CPU disables KQV/op offload on both paths, and every final row uses its batch-local output offset. The execution shapes do differ by design: direct uses one sequence and usually one prompt-processing decode; shared separates prefix and suffix decodes after full sequence copy; ragged shared/batch decodes multiple sequences together. The pinned llama.cpp source builds graphs from microbatch token/sequence shape and dispatches prompt versus token-generation work differently. Different quantized CPU GEMM/GEMV or reduction paths are therefore a plausible explanation, but are not proven as the cause. No tolerance was relaxed and no targeted profile was enabled.

Only the one-branch binary case and, where reached, the ragged two-branch case were executed. The 21-branch 16-way, changed-state-isolation, and repeated copy/clear-cycle cases are `unrun-after-decisive-failure` in every finalized receipt; they are not passed coverage. Because no exact configuration is eligible, M5 establishes safe serial fallback only. It provides no verified shared/batch speedup and no performance ratio is claimed.

Reproduction uses only the existing cache and target directories and runs one model at a time:

```sh
# Build once in the existing Metal target, then probe each model/mode.
CARGO_NET_OFFLINE=true GGML_METAL=ON CARGO_TARGET_DIR=target-m2-metal \
  cargo build --release -p openjev-cli --features metal,integration
for model in qwen3-0.6b minicpm5-2b qwen3.5-4b; do
  for mode in shared batch; do
    OPENJEV_INTEGRATION=1 target-m2-metal/release/openjev \
      --offline --device metal models probe "$model" --mode "$mode"
  done
done

# True CPU, with Metal absent from the native build and runtime offload disabled.
CARGO_NET_OFFLINE=true GGML_METAL=OFF CARGO_TARGET_DIR=target-m2-cpu \
  cargo build --release -p openjev-cli --features native,integration
for model in qwen3-0.6b minicpm5-2b qwen3.5-4b; do
  for mode in shared batch; do
    OPENJEV_INTEGRATION=1 target-m2-cpu/release/openjev \
      --offline --device cpu --gpu-layers 0 models probe "$model" --mode "$mode"
  done
done
```

A probe report exits 0 only when enabled and exits 1 on a completed numerical failure. Native/progress output remains on stderr. The reports are stdout JSON only; large repetitive native stderr was not committed.

Astra accepted the twelve numerical failures as the intended serial-only outcome: they are not a blocker and the frozen gates were not relaxed. The later lifecycle review found that the original forced-abort regression checked only the disabled report: a preexisting passing receipt for the same key could remain eligible after the crash. Reprobe now publishes an exact-key suspension in the parent before child launch and holds a process lock through publication. Crash, malformed/nonzero child output, launch error, identity mismatch, and publication failure leave the key suspended; only a normal fully validated exact passing child result clears suspension after atomic receipt publication. Failed diagnostic receipts remain ineligible. An updated isolated-temporary-cache process regression seeds a clearly labelled synthetic state-machine pass (not parity evidence and never a user-cache receipt), forces the cached-Qwen child abort before model load, and proves the old pass can no longer be loaded afterward. Model-free tests cover successful, failed, malformed/nonpublished, nonzero-after-passing, parent-identity mismatch, and publication-failure transitions.

No finalized numerical report was rerun or overwritten, no GGUF was downloaded or copied, no new native target directory was created, and `reference/` was unchanged.

## Resident HTTP / Jev SDK extension

**Passed on this Apple M3 Pro with cached Qwen3-0.6B and explicit Metal.** The final smoke retained at [`results/serve/20260920T031535Z-final/SUMMARY.md`](results/serve/20260920T031535Z-final/SUMMARY.md) verifies repeated mixed Choice/Noul/Score HTTP requests, a single resident model load, empty stdout, and successful SIGTERM shutdown. The official `@typesafe-ai/sdk` 0.6.0 passed model listing, all three typed response projections, probability/Score rounding and fields, token accounting, and 404/422 error paths. Configured bearer authentication succeeded; a wrong key produced JSON 401. A mixed SDK request reported 338 logical input tokens and zero generated tokens.

Parent code review fixed shared-requirement enforcement, worker failure/readiness/shutdown handling, and excessive repeated-state allocation before final smoke. Final fmt, workspace clippy/tests and CPU/Metal feature clippy/tests all passed; command/log evidence is in [`results/serve/20260920T031934Z-review-gates/`](results/serve/20260920T031934Z-review-gates/). CPU compilation/unit coverage is not a real CPU-server or Linux runtime smoke claim.

This is protocol/residency evidence, **not a performance benchmark or successful shared-mode acceleration**. Shared requests still disclose serial fallback on every tested production profile. See [`SERVE.md`](SERVE.md) for supported wire limits and [`PROGRESS.md`](PROGRESS.md) for the separate, still-incomplete M6 benchmark work.

## v0.1.0 — tagged release and downloaded-artifact acceptance

Status: **accepted for the bounded release contract.** This acceptance covers immutable-source CI, two published binary packages, downloaded-asset integrity, macOS user-local installation, and a real installed-release Metal HTTP/official-SDK smoke. It does not complete M6 or M7.

### Immutable release source and hosted CI

- Tag [`v0.1.0`](https://github.com/codesoda/openjev-rs/releases/tag/v0.1.0), release target, `BUILD-INFO.json`, and both passing workflows identify source commit `bb23406606e423fb35f5e62fdf5f170a6b14ad3f`.
- Main run [35493609481](https://github.com/codesoda/openjev-rs/actions/runs/35493609481) passed on that exact commit before tagging.
- Tag run [35494477837](https://github.com/codesoda/openjev-rs/actions/runs/35494477837) passed the native macOS and Linux formatting/clippy/test/build/configuration/package jobs and the release publication job. The Linux CI package ran extracted help/version and linkage checks; it did not run Linux model inference.
- The GitHub repository remained private. No release acceptance depends on making it public.

The parent downloaded both `.tar.gz` assets and `SHA256SUMS` from the GitHub Release with `gh release download`. `SHA256SUMS`, GitHub's asset digests, safe archive manifests, payload manifests, and macOS linkage all matched:

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `openjev-v0.1.0-aarch64-apple-darwin.tar.gz` | 8,656,257 | `b2b26ee4ed33b584f01b22b5ebec5d38745c4cd53a59e2e9c8fd303031896251` |
| `openjev-v0.1.0-x86_64-unknown-linux-gnu.tar.gz` | 9,471,588 | `4430be0d48e77248b3e170bcb572f521f794fb5a39ea34e99f6285be37a9b8e6` |
| `SHA256SUMS` | 222 | `3cf1f0d65b5379af1cce4c01c27dcc640331a3cb1ff5296431fa609acec34967` |

The Linux archive was verified but not executed on the local Mac. There is no Linux model-inference claim.

### Installed downloaded macOS artifact

No pre-existing `openjev` command or binary was replaced. The unchanged payload was installed under
`~/.local/share/openjev/releases/v0.1.0/openjev-v0.1.0-aarch64-apple-darwin/`, and `~/.local/bin/openjev` points to its executable. `~/.local/bin` was already on `PATH`. The installed executable reports `0.1.0`; its SHA-256 is `b9999f65f936fdd17e193af90e57bd568c2888f98889b15c3ce98d1a319fc66a`, exactly matching packaged `BUILD-INFO.json`.

The installed release smoke started at `2026-09-20T06:37:12Z` and passed using offline cached Qwen3-0.6B from a temporary directory outside the checkout with explicit Metal and without DYLD/library/model-path overrides:

- eight retained raw HTTP checks: health, readiness, three equal decoded JSON mixed Choice/Noul/Score response objects, wrong bearer 401, unknown model 404, and unsupported float 422;
- one resident process and one pinned model load, embedded Metal library, `MTL0 (Apple M3 Pro)`, empty server stdout, and exit 0 after SIGTERM;
- raw mixed-request usage of 313 input tokens and 0 output tokens;
- pinned official `@typesafe-ai/sdk` 0.6.0 smoke passed Choice/Noul/Score fields, distributions and rounding, Score legend, model listing and token usage of 338 input / 0 output tokens;
- every multi-question response disclosed `requested=shared; effective=serial` and fresh serial full-prompt fallback.

This is bounded artifact identity, integrity, linkage, installation, residency, protocol, and SDK evidence. It is not a latency/throughput result, acceleration claim, shared/batch parity result, complete model/OS matrix, hosted Jev parity claim, or Apple Developer ID/notarization claim. M6 remains incomplete and M7 remains unfinished.

Raw captures and the reproducible harness are in [`results/releases/v0.1.0-downloaded-metal/`](results/releases/v0.1.0-downloaded-metal/). The immutable release contains source commit `bb234066…ad3f`; this evidence was collected afterward and belongs to a later documentation commit. Independent Astra final acceptance review passed with no release blockers; the documentation-only follow-up does not change the tagged implementation or published artifacts.

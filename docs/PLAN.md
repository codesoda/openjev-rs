# openjev-rs — Phase A implementation contract

Status: **design only; no Rust implementation, native build, GGUF download, or runtime validation yet.** This document resolves the choices in `todo.md` before M1. The original brief remains the requirements ground truth. This contract corrects source/API assumptions with evidence; it does not override user requirements or relax acceptance gates. Reference material is read-only. Parent/Astra review accepted this plan, with the M2 continue-on-load-failure policy clarified below; commit precedes M1.

## 1. Scope and invariants

Implement a local library and JSON CLI for frozen-model, last-position option-logit Choice, with Noul and Score as transparent Choice adapters. No sampling, generation, grammar, logit bias, model training, server, vision, reranker, or GLiNER integration. A future `System1` trait can wrap the backend-neutral API; do not couple this workspace to gliner2-rs now.

Follow Python `src/core.py`, `direct.py`, `shared.py`, `serial.py`, and `cli.py`, not `webgpu-demo/worker.js`. The browser prompt, 20-option limit, hardcoded label bases and generated-token API are not our scoring contract. Our limit is 2–16 and token IDs are discovered, never assumed.

Hard invariants:

- Prompt bytes and token counts are strict gates, not quantities with a quantization tolerance.
- Every scoring path verifies exact single-token answer slots and append-boundary stability.
- No truncation, implicit BOS insertion, softmax over the full vocabulary, or silent model substitution.
- First maximum wins ties: ascending original option index; full-vocabulary ties use ascending token ID. Do not use an iterator `max_by` that returns the last equal item.
- All numerical JSON values are finite. Reject nonfinite option logits; allow negative infinity only as an explicitly masked full-vocabulary entry, contributing zero mass. Reject vocabulary NaN/+infinity and an all-negative-infinity vocabulary.
- Preserve option ID alignment, model/artifact identity, execution path, and transformation provenance.
- Carry these Python strings verbatim: `native full-vocabulary last-position logits restricted to declared answer slots` and `conditional option score; uncalibrated as decision confidence`.
- Documentation and the `limitations` output array also retain: `A forced typed output can still be semantically wrong.` and `Softmax over allowed tokens is conditional on the supplied alternatives; it is not calibrated operational confidence.`

## 2. Section 8 decisions (closed design choices, conditional runtime gates)

| Brief risk | Decision and explicit failure policy |
|---|---|
| 8.1 architecture support | Start with exact published `llama-cpp-2 = 0.1.156` and matching sys crate, not nonexistent crates.io 0.1.157. Its bundled llama.cpp is `e79e4bf660e19f2ad851e06c6913f7a8c5852621`. Qwen3.5 graph and MiniCPM5 tokenizer support exist in source; MiniCPM5 native config says `LlamaForCausalLM`, not a missing `minicpm5` architecture enum. This is not evidence that our three GGUFs load. M2 must smoke all three on this machine. On failure preserve the model entry with `unavailable` status and error; default remains `minicpm5-2b`, and a failed default returns an error. Never choose Qwen automatically. |
| 8.1 build fallback | Diagnose actual failure first. `LLAMA_CPP_PATH` is **not supported** by inspected build.rs, and `dynamic-link` builds the bundled source as shared libraries rather than selecting an arbitrary external llama.cpp. If necessary, propose a separately reviewed, commit-pinned utilityai/sys fork with its submodule and bindings rebuilt together; rerun M2/M3/M5 after any native revision change. A failing model may be explicitly excluded from a release only with user/reviewer approval and documented reduced support; that is not completion of the original three-model gate. No speculative mistralrs swap; it is a future backend, not a drop-in or guaranteed pure-Rust fix. |
| 8.2 disabled thinking | Use a small, **restricted, model-profile renderer** for exactly two string messages (system/user), no tools, `add_generation_prompt=true`, `enable_thinking=false`. Renderings below have been checked against pinned upstream Jinja for all 252 fixtures. Safe crate `apply_chat_template` does not accept kwargs; common's C++ Jinja API is not exposed by the sys wrapper. Do not add a C++ shim or general Jinja engine in v1. Unknown GGUFs require explicit `--template-profile qwen3|qwen3.5|minicpm5`; arbitrary template inference is rejected. Record that override, and do not claim unknown artifacts are golden-verified. |
| 8.3 serialization | Preserve insertion order at every object depth. Recursively reject floating-point JSON numbers, including lexical `1.0`, `1e0`, and `-0.0`, until a separately reviewed Python-repr implementation exists. Accept integer JSON tokens only in i64/u64 range, normalize integer `-0` to `0`, reject overflow. Null/bools are allowed within structured state. Use a real recursive Python-compatible serializer, never whitespace replacement on JSON text. |
| 8.4 shared/hybrid | Full-sequence copy from immutable seq 0 into clean branch sequences; never roll back a hybrid suffix. Validate actual shared/direct parity per artifact/device/config before enabling shared. Unsupported, unprobed, failed-copy or out-of-tolerance configurations visibly fall back to serial **full-prompt** direct scoring, unless `--require-shared`, which errors. This serial fallback is not the Python prefix-cache optimization. |
| 8.5 accelerators | Explicit `cpu`, `metal`, `cuda` backend feature surfaces; no default OpenMP/common dependency. On macOS use Metal with all layers by default for accelerated builds; CPU build/runtime are explicit. Linux baseline is CPU; CUDA is opt-in with toolkit. Log actual offload/configuration to stderr and output metadata. Never claim Metal/CUDA was used merely because a feature compiled. OOM/load failure does not silently reduce layers or change devices; suggest an explicit retry configuration. |
| 8.6 licensing | Project code MIT. Preserve upstream `reference/semif-py/LICENSE` (Copyright (c) 2026 TheoLeeCJ), include its notice for copied prompts/fixtures/ported algorithms in `THIRD_PARTY.md`, link TheoLeeCJ/openjev and SemIf, and state non-affiliation with TypeSafe/Jev. Credit llama.cpp, utilityai bindings (MIT OR Apache-2.0), hf-hub (Apache-2.0) and model/tokenizer sources under their own licenses. Do not represent model weights as covered by our MIT license; distribute no weights. |

Runtime failures are evidence, not unresolved design questions. Per the user's explicit M2 exception, record each failed model in docs/RESULTS.md and continue to remaining models and milestones; do not stall on a model load failure or claim three-model success. M3's Qwen exact hash/token gate remains mandatory: if it cannot pass, stop parity-dependent advancement with reproduction, logs, attempted paths and required decision. Do not weaken acceptance after seeing results.

## 3. Dependency/source evidence

### 3.1 Registry versus Git

Planning queries used `cargo info`, crates.io API, GitHub contents API and `git ls-remote`; no native compilation was necessary.

- `cargo info llama-cpp-2@0.1.157` failed: not in registry. Published latest stable for **both** llama crates is **0.1.156**, downloaded into `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/`.
- Published `.cargo_vcs_info.json`: utilityai commit `63e549708237b16b39018288582655539e0a9d5b`.
- Remote utilityai `main`/HEAD observed: `992519ec699510e488223d490747cbdb7dbc182f`; its Cargo.toml calls itself 0.1.157, unpublished at inspection. Both commits point to llama.cpp `e79e4bf660e19f2ad851e06c6913f7a8c5852621`.
- Remote ggml-org/llama.cpp HEAD observed: `60081bb2b5b3294165a4d67c5cbeebe74c868014`. This is **not** our selected pin.
- Remote main's `model.rs` and `wrapper_common.h` are byte-identical to registry 0.1.156. Its build.rs has Apple duplicate-C++-link and Linux arm64 dynamic-backend fixes; neither changes the bundled architecture pin. We disable common/mtmd and dynamic backends, rather than preemptively adopting a Git build.

Primary URLs:
`https://crates.io/api/v1/crates/llama-cpp-2`,
`https://github.com/utilityai/llama-cpp-rs/tree/63e549708237b16b39018288582655539e0a9d5b`,
`https://github.com/ggml-org/llama.cpp/tree/e79e4bf660e19f2ad851e06c6913f7a8c5852621`.
Pin exact direct dependencies and commit Cargo.lock in M1/M2. Registry-confirmed selected versions:

| Dependency | Version / policy |
|---|---|
| llama-cpp-2, llama-cpp-sys-2 | `=0.1.156`, optional native feature, `default-features=false`; lock matching sys version; no direct sys usage initially |
| hf-hub | `=1.0.0`, `default-features=false`, features `blocking`, `rustls-tls` |
| clap | `=4.6.7`, derive |
| serde / serde_json | `=1.0.229` / `=1.0.151`, derive / `preserve_order` |
| thiserror / anyhow | `=2.0.20` / `=1.0.104`; typed library errors / CLI context only |
| sha2 | `=0.11.0` |
| tracing / tracing-subscriber | `=0.1.44` / `=0.3.23`, stderr writer |

Planning host reports rustc 1.95.0 and cargo 1.95.0. Pin this tested toolchain initially, make no lower MSRV claim. Registry existence is verified, dependency resolution/build compatibility remains M1/M2 evidence. Standard-library channels, locks where available, and deterministic SHA-256-based permutations avoid unnecessary runtime dependencies. Choose and verify any test-only crate before adding it.

hf-hub 1.0.0 is **not** the older `api::sync::Api` API. Its real source:
`hf-hub-1.0.0/src/client.rs:203,302,327`: `HFClient::builder().cache_dir(path).build_sync()`;
`src/blocking.rs:197`: `HFClientSync::model(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync<RepoTypeModel>`;
`src/repository/download.rs:1332–1350`: builder-backed blocking `download_file`, fields `filename: String`, `local_dir: Option<PathBuf>`, `revision: Option<String>`, `force_download: bool`, `local_files_only: bool`, `progress: Option<Progress>`, return `HFResult<PathBuf>`.
Use `repo.download_file().filename(file).revision(commit).local_files_only(offline).send()` with no `local_dir`, so the central content cache is used. The blocking implementation owns an internal runtime thread; our application need not introduce Tokio.

### 3.2 Exact llama API contract

Paths below are relative to the downloaded `llama-cpp-2-0.1.156/` (`R`) or `llama-cpp-sys-2-0.1.156/` (`S`) registry directories. Line numbers identify inspected source, not documentation guesses.

| Source | Exact signature / consequence |
|---|---|
| R `src/model.rs:728` | `pub fn chat_template(&self, name: Option<&str>) -> Result<LlamaChatTemplate, ChatTemplateError>`; use `None` only for metadata inspection. |
| R `src/model.rs:936` | `pub fn apply_chat_template(&self, tmpl: &LlamaChatTemplate, chat: &[LlamaChatMessage], add_ass: bool) -> Result<String, ApplyChatTemplateError>`; calls C `llama_chat_apply_template`, **no kwargs/enable_thinking**, not common's full Jinja renderer. |
| S `llama.cpp/common/chat.h:250–268,325–326` | C++ `common_chat_templates_inputs` has `use_jinja`, `enable_thinking`, `std::map<std::string,std::string> chat_template_kwargs`; `common_chat_templates_apply(const common_chat_templates *, const common_chat_templates_inputs &) -> common_chat_params`. Not callable through existing Rust C bindings. S `wrapper_common.h` has grammar/fit/speculative helpers, no chat wrapper; build.rs:463–478 allowlists `llama_*`/`llama_rs_*`, not C++ common symbols. |
| R `src/model.rs:302` | `pub fn str_to_token(&self, str: &str, add_bos: AddBos) -> Result<Vec<LlamaToken>, StringToTokenError>`; `AddBos::Never`. Internal `llama_tokenize` parses special tokens. Template explicitly includes every required special token, including MiniCPM's `<s>`. |
| R `src/model.rs:213,421` | Deprecated `token_to_str(&self, token: LlamaToken, special: Special) -> Result<String, TokenToStringError>` is forbidden under warnings-as-errors. Use `token_to_piece_bytes(&self, token: LlamaToken, buffer_size: usize, special: bool, lstrip: Option<NonZeroU16>) -> Result<Vec<u8>, TokenToStringError>`. For A–P use sufficient fixed buffer (32), `false`, `None`, compare exact ASCII bytes; propagate error, no lossy UTF-8. |
| R `src/llama_batch.rs:50–99` | `pub fn add(&mut self, LlamaToken(id): LlamaToken, pos: llama_pos, seq_ids: &[i32], logits: bool) -> Result<(), BatchAddError>`; initialized logits stores the **batch token offset**, not sequence ID or output ordinal. Allocate sufficient batch capacity and seq-ID slots. |
| R `src/context.rs:101` | `pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), DecodeError>`; on success replaces initialized-logit indices with those of this batch. A failed decode invalidates our context for reuse; recreate it. |
| R `src/context.rs:313–330` | `pub fn get_logits_ith(&self, i: i32) -> &[f32]`; requires nonnegative initialized batch offset `< n_ctx`. Do not pass `-1`, absolute prompt position after chunking, sequence ID, or dense-output rank. For a final token at local offset 17 in the most recent decode, call 17. Copy needed readouts before next decode. |
| S `llama.cpp/src/llama-context.cpp:850–889,3740–3748` | Native positive index translates through `output_ids[i]` to output row. Native supports negative indexing but Rust assertions do not. Native getter synchronizes the context; include retrieval in forward timing. |
| R `src/context/kv_cache.rs:51` | `pub fn copy_kv_cache_seq(&mut self, src: i32, dest: i32, p0: Option<u32>, p1: Option<u32>) -> Result<(), KvCacheConversionError>`; `None,None` passes -1,-1. Return only reports integer conversion errors: native copy is void, so success is **not** a parity proof. |
| R `src/context/kv_cache.rs:83` | `pub fn clear_kv_cache_seq(&mut self, src: Option<u32>, p0: Option<u32>, p1: Option<u32>) -> Result<bool, KvCacheConversionError>`; half-open range; full removal `Some(seq),None,None`. Check bool. `clear_kv_cache(&mut self)` clears all data/metadata. |
| R `src/context/params/get_set.rs:21,53,83,113,713` | `with_n_ctx(self, Option<NonZeroU32>)`, `with_n_batch(self,u32)`, `with_n_ubatch(self,u32)`, `with_n_seq_max(self,u32)`, `with_kv_unified(self,bool)` all return Self. Configure explicitly; n_seq_max is concurrent sequence capacity, not token count. |

S `llama.cpp/src/llama-context.cpp:287–303`: n_ctx is padded to 256; with `kv_unified=false` the per-sequence capacity is padded from `n_ctx/n_seq_max` and total may change. With `kv_unified=true`, sequence limit is the whole n_ctx, sharing a total token-cell pool. We select **unified KV** for shared/batch mode; explicitly budget total occupied cells plus every branch's absolute length. Do not divide 4096 by 22 accidentally or assume 4096 per sequence without allocating it.

S `llama.cpp/src/llama-memory-hybrid.cpp:143–154`: removal tries recurrent first and may return false; copy dispatches to attention and recurrent memory. `llama-memory-recurrent.cpp:170–197,235–270`: bounded per-token snapshots may permit some rollback, but arbitrary rollback is not supported; copying aliases/copies the current recurrent tail, not a historic prefix selected by p0/p1. Copy only the entire, never-advanced prefix sequence. `llama-kv-cache.cpp:502` also asserts full KV buffers for the relevant cross-stream copy case. Keep full attention/SWA storage where needed, probe in a subprocess because native assertions are not catchable Rust errors.

S `llama.cpp/src/llama-model.cpp:309,2729`, model graph `src/models/qwen35.cpp`, and `src/llama-vocab.cpp:530,2132` are Qwen3.5 and MiniCPM5 support evidence. MiniCPM5's pinned HF config is `model_type=llama`, `architectures=[LlamaForCausalLM]`. No runtime or GGUF metadata verification has occurred in Phase A.

### 3.3 Build flags, without imaginary overrides

S build.rs was read completely before proposing any workaround:

- :398–412 selects `CARGO_MANIFEST_DIR/llama.cpp`; `dynamic-link`/`LLAMA_BUILD_SHARED_LIBS` changes shared/static, not source location.
- :666–688 forwards `CMAKE_*` and `GGML_*`, with later explicit settings taking precedence.
- :701 onward controls `GGML_NATIVE`; :987–993 enables CUDA (`GGML_CUDA=ON`, NCCL off); :1047–1050 controls OpenMP by feature.
- Metal feature exists but this build.rs has no corresponding `cfg!(feature="metal")` switch. S `llama.cpp/ggml/CMakeLists.txt:95–100,238` defaults Metal ON on Apple. Do not equate `--no-default-features` with CPU-only on macOS.
- CPU-only Mac build: `GGML_METAL=OFF`, separate target directory, `--no-default-features --features native`; runtime zero GPU layers **and** op/KV offload disabled. Metal build: `GGML_METAL=ON`, `--features metal`; inspect CMake cache and actual runtime device logs. CUDA: `--features cuda`, toolkit/driver and pinned CUDA arch configuration recorded. No all-features CI combining incompatible accelerators.
- Use available-parallelism threads (bounded positive i32) by default; CLI overrides both generation/prefill thread settings. `n_batch=512`, `n_ubatch=512` initial defaults, benchmark before tuning. Full-layer offload requests `with_n_gpu_layers(u32::MAX)`, CPU requests 0; R `src/model/params.rs:517–522` converts overflow to i32::MAX. Retain actual offloaded layers in metadata.
- Require Xcode command-line tools, CMake, C/C++ compiler and libclang for bindgen. Record build environment. Do not force deployment targets, SDK paths, compiler or linkage environment variables without a reproduced need.

## 4. Workspace and public API

Final layout:

```
Cargo.toml, Cargo.lock, rust-toolchain.toml
crates/openjev-core/src/{lib,types,validate,prompt,numerics,primitives,eval,postprocess}.rs
crates/openjev-llama/src/{lib,registry,cache,engine,direct,shared,batch,worker}.rs
crates/openjev-cli/src/{main,args,input,output,commands}.rs
crates/*/tests/...
fixtures/                 # small credited prompt/oracle fixtures, never weights
manifests/models.json     # our checked artifact hashes and profile/probe status
schemas/                 # machine-readable export of section 7 in M1/M4
scripts/                 # parity/report helpers, no reference modifications
reference/               # unchanged
THIRD_PARTY.md, LICENSE, README.md, PROMPT.md
docs/{PLAN,PROGRESS,RESULTS}.md
```

`openjev-core` has no llama, hf-hub, CMake or async dependency. `openjev-llama` has optional `native` dependencies; `metal`/`cuda` imply native. M1 creates only a feature-gated backend skeleton and backend-disabled CLI parser, so `cargo test --workspace` does not compile C++. `default-members` core/CLI is a convenience, not a way to hide warnings. M2 explicitly activates native; M4 documents native CLI installation/build. A backend-disabled scoring command returns a structured backend-unavailable error, never mock probabilities.

Public names (Rust signatures are a design contract, not source files yet):

- `Decision { id: String, state: StateValue, question: String, options: Vec<DecisionOption> }`; `DecisionOption { id: String, description: String }`.
- `StateValue` is a validated nonempty string/object/array wrapper over insertion-ordered `serde_json::Value`; `TryFrom<Value>` and string constructor. No publicly unchecked state constructors. `Question { id, question, options }` is state-free; a shared request owns one `StateValue` and `Vec<Question>`.
- `Primitive::{Choice,Noul,Score}`; Noul creates IDs `yes`,`no` descriptions `Yes`,`No` in that order, exposes `p_yes`; Score creates 2–16 ordered levels with finite numeric values (default 0..K-1), exposes `expected_value`, `argmax_level`, and the unchanged distribution. Numeric level values do not enter state serialization; only level descriptions enter the prompt.
- `PromptProfile::{Qwen3,Qwen35,MiniCpm5}`, `PreparedPrompt { text, prompt_sha256, prompt_version, profile }`; core renders bytes, backend produces `EncodedPrompt { token_ids, answer_token_ids, prompt metadata }` privately.
- `Readout`, `ModelMetadata`, `ExecutionMetadata`, `SharedTiming`, `Postprocess`, `EvalReport`, `BenchReport`, `OpenJevError` (validation/serialization/template/slot/context/cache/model/decode/parity/io categories).
- Backend `ModelSpec::{RegistryId,Local{path,expected_sha256,profile},Hub{repo,revision,file,expected_sha256,profile}}`; `ModelRegistry::list/resolve/pull/path`, `EngineOptions` (device, layers, threads, token caps, batches, sequence cap).
- Synchronous engine methods: `score_direct(&mut self, &Decision) -> Result<Readout>`, `score_shared(&mut self, &StateValue, &[Question]) -> Result<Vec<Readout>>`, `score_batch(&mut self, &[Decision]) -> Result<Vec<Readout>>`. Batch preserves input order and never shares unrelated states.
- `EngineHandle::spawn(ModelSpec, EngineOptions) -> Result<EngineHandle>` starts **one owner thread** and exposes owned request/response messages over bounded std channels. Initialize backend and load model inside it, create contexts borrowing the model inside that thread's scope, not a self-referential model/context struct. Do not move borrowed contexts across threads or add unsafe Send/Sync. Drop contexts, then model, then backend there; support explicit shutdown/join, propagate worker panic as error. One process backend lifetime is coordinated to avoid duplicate init/free. Public Readout is owned and thread-safe; no borrowed logits escape.

## 5. Input, prompt and arithmetic

### 5.1 Validation and serializer

Actual `core.validate_row` permits **empty option id and empty description**; option IDs must merely be strings and unique. Only decision id/question and state must be nonempty. Whitespace-only strings are valid (do not trim). Top-level state must be string, object or array; empty string/object/array and scalar top-level states fail. Nested empty values, null and booleans are valid. Unknown row metadata is retained for evaluation but never enters prompt. Require unique decision IDs within a run/shared batch. JSON duplicate keys are rejected at ingestion (intentional stricter rule than Python's last-key-wins); report the compatibility boundary. Reject lone-surrogate strings rather than silently replace them.

Use `serde_json` preserve_order, validate numbers recursively **before serialization**, and forbid sorting object keys. Library callers must supply a preserved-order Value; an already sorted map cannot recover source order. Integer overflow or a float yields a path-specific error, not conversion to string. Maintain fixed outer key order `evidence, criterion, options`, option key order `letter, description`. This differs from the fixture's own option key order and is intentional.

Bound every complete raw public JSON document to 128 nested array/object containers. `serde_json::from_str`, `from_slice`, and `from_reader` deserialization first captures a boxed `RawValue`, then passes its preserved text to the same strict parser used by explicit string constructors. This bypasses serde_json's ordinary recursive visitor limit without disabling safety: a standalone state with 128 nested arrays is accepted, 129 is rejected, and an enclosing Decision/Question object counts as one container. The raw routes therefore preserve lexical integer `-0` for normalization to `0` while still rejecting `-0.0`, `-0e0`, other float spellings, overflow, and duplicates. After strict parsing, extract typed Decision/Question fields directly from the insertion-ordered object and move state and retained metadata subtrees unchanged; do not feed arbitrary `Value` subtrees through another Serde pass or reserve/blacklist legal JSON keys. `StateValue::try_from(Value)` is the required bounded validation path for untrusted already-built trees and cannot recover duplicate keys or a lost numeric lexeme (serde_json represents parsed `-0` there as floating `-0.0`, which remains rejected). Generic upstream operations that recursively serialize an arbitrary-depth tree before this API—including `serde_json::from_value::<StateValue>` through `RawValue`, and `Value::to_string()`—are explicitly outside the non-overflow depth guarantee.

Python serializer: separators `, ` and `: `; UTF-8 unescaped non-ASCII; escape quote/backslash and controls U+0000–001F exactly as Python (`\b`, `\f`, `\n`, `\r`, `\t`, lower-case `\u00xx` otherwise). No HTML/slash escaping, Unicode normalization or U+2028/U+2029 escaping. Decimal integers, lowercase true/false/null, no terminal newline. Use differential tests against stdlib `json.dumps(..., ensure_ascii=False, allow_nan=False)`, with nested order, all control chars, Unicode, quotes, punctuation within strings and signed/unsigned limits. Inputs containing NUL serialize to escaped text, so CString tokenization sees no raw NUL.

Exact system string (one line, one space between sentences):

```
Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.
```

### 5.2 Restricted chat rendering: decided and source-checked

Let S be that system string and U the Python-spaced payload. Qwen3 and text-only Qwen3.5 render these literal concatenated bytes:

```
<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

The display uses `\n` for actual newline bytes; braces are substitution notation, not literal prompt characters. MiniCPM5 prepends exactly `<s>` and otherwise uses these same bytes. Qwen3.5 upstream trims message content, but S has no outer whitespace and U starts/ends with JSON braces, so this changes nothing. Preserve whitespace *inside* U. No implicit template fallback or assistant continuation beyond this suffix.

Pinned tokenizer artifacts fetched without GGUFs:

| Native model @ revision | Artifact SHA-256 | Extracted template SHA-256 |
|---|---|---|
| Qwen/Qwen3-0.6B @ c1899de289a04d12100db370d81485cdf75e47ca | tokenizer_config.json `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| openbmb/MiniCPM5-2B @ 12a3808a956f869c767195e9266b59c4d21d92e2 | tokenizer_config.json `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b` | separate chat_template.jinja `cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c` |
| Qwen/Qwen3.5-4B @ 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a | tokenizer_config.json `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8` | chat_template.jinja and embedded template `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |

URL form: `https://huggingface.co/{repo}/resolve/{40-char revision}/{artifact}`. A Python/Jinja 3.1.4 planning probe rendered all 252 supplied rows for each fetched template with disabled thinking; each matched the restricted renderer **252/252**. Separately, the restricted Qwen3 renderer matched all **144 authored + 108 perturbation** committed prompt hashes. No GGUF token counts were measured. Only Qwen3's row-level predictions are present in this reference snapshot; the ladder mentions other prediction files that are absent. Do not invent MiniCPM/Qwen3.5 row-level reference logits.

M1 stores small test-oracle renders with source attribution, hashes and regeneration instructions outside reference. M2 compares registered model template metadata against expected profile semantics and records its hash; absence/mismatch is an explicit template error until reviewed against the pinned native oracle. An arbitrary local/Hub model's explicit profile is a deliberate override recorded as such, with no parity claim.

### 5.3 Slots and readout

For each A–P in use: tokenize the letter with AddBos::Never, require exactly one token, exact ASCII round-trip, IDs in vocabulary bounds and unique. Tokenize full prompt with no added BOS. Require nonempty IDs and `tokenize(prompt + letter) == prompt_ids + [letter_id]` for **every** option, not only winning one. Hash UTF-8 prompt text, not token IDs. Do not use browser labelBase.

Convert selected f32 logits to f64; compute stable subtract-max softmax and full-vocabulary logsumexp in f64. `allowed_token_mass = exp(logsumexp(raw_slot_logits) - full_vocab_log_normalizer)`, clamp only tiny rounding excursions into [0,1] and reject larger contradictions. `full_vocab_argmax_id` is from raw logits. Raw option logits and mass are never temperature-scaled. Optional normalized-margin `confidence = (max(p)-1/K)/(1-1/K)` is uncalibrated, not a correctness probability; distinguish evaluator's maximum-probability ranking statistic from this field.

## 6. Backend execution and cache integrity

### 6.1 Artifact registry and download-once rule

Transfer only in M2 or explicit `models pull`/scoring cache miss, never Phase A. Default ID is minicpm5-2b. Exact artifact pins and independently queried HF LFS hashes:

| ID | Repo @ revision / filename | Bytes / SHA-256 |
|---|---|---|
| qwen3-0.6b | Qwen/Qwen3-0.6B-GGUF @ 23749fefcc72300e3a2ad315e1317431b06b590a / Qwen3-0.6B-Q8_0.gguf | 639446688 / `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| minicpm5-2b | openbmb/MiniCPM5-2B-GGUF @ 2079a22f3beaa4e306449978533478fe0522f4b3 / MiniCPM5-2B-Q4_K_M.gguf | 1561318368 / `ec2d5801640099e97d8d7e8003ad4d81f336e757811f03a26173dddf386602fd` |
| qwen3.5-4b | bartowski/Qwen_Qwen3.5-4B-GGUF @ 4168f45a16a1290d65a4ec0fa312ae917a4c15d6 / Qwen_Qwen3.5-4B-Q4_K_M.gguf | 3013027808 / `13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983` |

Hashes come from `https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true`, matching selected filename, LFS sha256 and size; these are expectations, not hashes of locally downloaded weights. Add them to our manifest without editing reference.

Root precedence: `--cache-dir`, `$OPENJEV_HOME`, `~/.cache/openjev`; HF cache under `root/hub`, receipts/probes under `root/openjev`. Reuse the same canonical blob path for models pull, smoke, all tests and benchmarks; do not make model copies in repo/test temp directories. On hit, verify complete byte length and SHA-256 before loading (once per worker load, not per decision). Offline reads make no HTTP request. Never trust an ETag or a size-only check as a cryptographic validation.

Serialize pull for the same repo/revision/file with a process-safe lock; hf-hub handles its transfer temp/cache mechanics, our verified receipt is written atomically only after hash/size verification. Interrupted transfer is not a verified model. Concurrent-process and tamper tests must establish download-once behavior. Mismatched cached data is quarantined/untrusted and returns an integrity error; `models pull --repair` is explicit permission to redownload exactly that pinned artifact once. Local files are never modified/quarantined automatically. Remote custom specs require a 40-hex commit and expected SHA-256; local custom files may omit expected hash, but compute actual hash and label integrity `local-unverified`. A model identity fingerprint includes bytes, template profile, tokenizer metadata and native build/config, not only a friendly ID. Local model metadata uses `source=local`, canonical path as file, and `revision=sha256:<actual hash>`; remote revision always denotes the GGUF repo commit, never the native BF16 commit (stored separately).

### 6.2 Context policy

Separate **per-prompt cap** from **allocated total cache**:

- `--max-tokens` defaults 4096 and is a strict per-prompt limit like Python. Raising it is explicit, checked against model's native trained limit; no RoPE extrapolation in v1.
- Unspecified `--n-ctx` is auto allocation starting at 4096 cells and growing before decode to the required 256-rounded total. `--max-context-tokens` defaults 32768 total cells and bounds growth. Report requested and actual n_ctx. No mid-decode growth/truncation; destroy/recreate context if a subsequent request needs more.
- Explicit `--n-ctx N` is a hard **total** cache cap (subject to documented native allocation padding, never used to accept more user input). Over-cap direct input errors. Shared/batch planner partitions into smaller waves; if even one shared branch cannot fit, use visible serial fallback or error with require-shared. Never silently raise an explicit cap.
- Shared unified budget: one prefix plus sum of active suffix lengths, with every full sequence length within native context limit and enough recurrent state slots. Independent batch budget: sum of full active lengths. Include reserve prefix seq 0 in `n_seq_max=active_branches+1`; default maximum concurrent branches 32, configurable down/up within native limits. Query/record actual context settings and reject arithmetic/position overflow.
- Sequence capacity is independent of LlamaBatch's per-token seq-ID allocation. We add one seq ID per token and use cache copy; allocating `LlamaBatch::new(n_batch,1)` does not limit the context to one sequence.

### 6.3 Direct, shared, serial fallback and batch

Direct: tokenize/validate, start clean context, add prompt tokens at absolute positions 0..L-1 in chunks no larger than n_batch (and compatible with physical ubatch). Keep n_batch no larger than effective n_ctx to satisfy the wrapper's logits assertion. Request logits only at the final token. After its decode, call get_logits_ith with its **local chunk index**, copy the vocabulary-derived statistics immediately, then clear/drop context. Include all prefill decode chunks and synchronized retrieval in forward_seconds. Total excludes initial download/model load but includes validation, prompt/tokenization, context preparation, forward and readout. Report load/download separately when requested; never call a multi-chunk prefill literally one kernel invocation.

Shared:

1. Validate nonempty questions with unique IDs; compare state by serialized **bytes/order**, not map-insensitive JSON equality. Prepare and slot-check every full prompt.
2. Mirror `_state_prefix`: locate the one unmodified user payload in rendered prompt; create Python-spaced `{"evidence": state}` and remove its closing brace; prepend template text before payload; tokenize no-BOS; drop **one final token**, not one character. Reject empty prefix, missing/multiple payload occurrences, or any full prompt not starting with prefix and a nonempty suffix. These cases can use visible full-prompt serial fallback, never an approximate common prefix.
3. Prefill seq 0 once at positions 0..P-1, no logits needed. Keep it immutable. Full-copy `copy_kv_cache_seq(0,i,None,None)` into empty branches 1..B before suffix decode.
4. Add suffixes with absolute positions P..L_i-1, their own sequence IDs and logits=true only for each final suffix token. Store `(decision_index, batch_local_offset)` each time a suffix ends; read/copy those logits immediately after that decode and before clearing/reusing batch. No padding tokens are necessary. Scheduler keeps deterministic sequence-major order; parity probes must include ragged suffixes and ends in multiple chunks.
5. Full-clear every branch and check returned bool. Retain seq 0 for later waves of this call only. On error discard context before fallback; do not return a mixture of tentative shared and retried results. No cross-request persistent warm cache in v1.
6. Record common shared_timing on each result, including wall time and actual suffix counts. `padded_suffix_tokens=true_suffix_tokens` for unpadded Rust batching. Do not sum repeated group timings as independent work. Shared rows omit per-row forward/total timings when only group measurements exist.

Shared eligibility is opt-in via a successful M5 probe record keyed by artifact hash, llama pin, device, layer/offload settings, context/batch/sequence configuration family and profile. Unseen custom models default to serial fallback. Native assertions/crashes during probes are recorded by the subprocess controller, not allowed to crash the CLI parent. `models probe ID --mode shared|batch` explicitly executes the documented parity suite for the selected device/config and writes a versioned local probe receipt; it never enables a failed profile. Custom specs may be supplied as ID using the same model-spec grammar. A successful receipt is required on the deployment machine, not merely a developer's unrelated GPU. Enabling experimental reprobe is explicit. Runtime copy/decode/clear failure falls back visibly only after a fresh context is made.

`serial` mode in v1 is serial **full-prompt** scoring (also the shared fallback); do not use Python's `native-state-prefix-cache-v1` or cache readout string for it. Output `requested_mode`, `effective_mode`, `fallback_reason`, probe identity; include stderr warning even if quiet suppresses progress (quiet does not suppress a semantic-path change). `--require-shared` prohibits fallback. Batch mode uses separate seq IDs for independent prompts, no prefix copy, ordered readouts and per-mode timing; if memory requires waves, report them. Unprobed hybrid batch configuration also uses visible serial fallback until its isolation/ordering parity probe passes.

## 7. Output schema (complete v1 readout superset)

This is a normative schema definition for M1's machine-readable JSON Schema export. Notation: `?` means optional/omitted, `T|null` permits explicit null, `[]` means array. No unexplained fields or NaN/Infinity; output struct serialization uses these exact snake_case names. Importers allow extra upstream metadata for compatibility. All vector lengths equal K and all indices/choices agree with option_ids, checked in code in addition to schema. This superset preserves every direct, serial, shared and browser-ladder readout field; unavailable measurements are omitted, never fabricated as zero.

```
Readout {
  schema: "openjev-readout-v1",
  id: nonempty string,
  primitive: "choice"|"noul"|"score",
  choice: string, choice_index: integer[0,K),
  option_ids: unique string[K],                 # empty string ID is legal
  probabilities: number[0,1][K],               # sum within 1e-12 after our computation
  option_logits: finite number[K],             # raw unscaled base-run logits
  answer_token_ids: unique nonnegative integer[K],
  allowed_token_mass: number[0,1],
  full_vocab_argmax_id: nonnegative integer,
  full_vocab_log_normalizer: finite number,
  input_tokens: positive integer,
  forward_seconds?: nonnegative number,
  total_seconds?: nonnegative number,
  prompt_sha256: lowercase hex[64],
  prompt_version: "direct-options-v1",
  model: ModelMetadata,
  readout: string enum described below,
  probability_status: "conditional option score; uncalibrated as decision confidence",
  limitations: string[],
  execution: ExecutionMetadata,
  confidence?: number[0,1],
  confidence_status?: "normalized margin; uncalibrated",
  p_yes?: number[0,1],                          # noul only
  level_values?: finite number[K],             # score only
  expected_value?: finite number,              # score only
  argmax_level?: string,                       # score only, option ID
  cache_hit?: boolean,
  prefix_tokens?: nonnegative integer,
  prefix_sha256?: lowercase hex[64],
  prefill_seconds?: nonnegative number,
  copy_seconds?: nonnegative number,
  suffix_forward_seconds?: nonnegative number,
  shared_timing?: SharedTiming,
  postprocess?: Postprocess
}
ModelMetadata {
  id: string, source: string, revision: string, file: string,
  quant: string, backend: string, artifact_sha256: hex[64],
  integrity: "manifest-sha256"|"caller-sha256"|"local-unverified",
  dtype: string,                               # actual quantized/mixed, never fake BF16
  native_reference?: {source:string, revision:hex[40], dtype:"bfloat16"},
  template_profile: "qwen3"|"qwen3.5"|"minicpm5",
  template_sha256: hex[64]|null,
  template_override: boolean,
  serving_config?: string,
  adapter?: string|null, adapter_sha256?: hex[64]|null,
  adapter_revision?: string|null,
  torch_version?: string, transformers_version?: string
}
ExecutionMetadata {
  requested_mode: "direct"|"serial"|"shared"|"batch",
  effective_mode: "direct"|"serial"|"shared"|"batch",
  fallback_reason: string|null,
  device: "cpu"|"metal"|"cuda", device_name:string,
  gpu_layers_requested: string|nonnegative integer,
  gpu_layers_actual: nonnegative integer,
  threads: positive integer, n_ctx_requested: positive integer|null,
  n_ctx_actual: positive integer, max_tokens:positive integer,
  n_batch:positive integer, n_ubatch:positive integer, n_seq_max:positive integer,
  kv_unified:boolean, waves:positive integer,
  probe_id: string|null, run_id:string, group_id?:string
}
SharedTiming {
  total_seconds:nonnegative number, encode_seconds:nonnegative number,
  prefix_tokens:nonnegative integer, prefill_seconds:nonnegative number,
  replicate_seconds:nonnegative number, suffix_forward_seconds:nonnegative number,
  batch_size:positive integer, true_suffix_tokens:nonnegative integer,
  padded_suffix_tokens:nonnegative integer
}
Postprocess {
  version:"openjev-postprocess-v1",
  temperature:positive finite number,
  calibration_id:string|null,
  permutation_count:positive integer, seed:nonnegative integer,
  permutation_algorithm:"sha256-factoradic-v1",
  aggregation:"mean-id-aligned-probabilities",
  raw_fields_reference:0,
  base_probabilities:number[0,1][K],
  samples:RawSample[N]
}
RawSample {
  permutation:integer[K],                     # displayed index -> original index
  option_ids:unique string[K], answer_token_ids:unique nonnegative integer[K],
  option_logits:finite number[K], probabilities:number[0,1][K],
  prompt_sha256:hex[64], prompt_version:"direct-options-v1", input_tokens:positive integer,
  allowed_token_mass:number[0,1], full_vocab_argmax_id:nonnegative integer,
  full_vocab_log_normalizer:finite number,
  forward_seconds?:nonnegative number, total_seconds?:nonnegative number,
  execution:ExecutionMetadata, shared_timing?:SharedTiming
}
ErrorRecord {
  schema:"openjev-error-v1", id?:string,
  error:{code:string,message:string,details:object},
  parse_status?:"unparsed"
}
```

`readout` for actual direct/serial-full-prompt/batch rows is exactly `native full-vocabulary last-position logits restricted to declared answer slots`; actual shared rows use exactly Python shared's `native selected suffix-position logits`. Import schema also accepts historical `native-state-prefix-cache-last-position` (Python serial) and `native-full-vocabulary-last-position` (ladder), but we never claim those execution paths if not used. `serving_config` uses Rust-specific `llama-direct-v1`, `llama-serial-full-prompt-v1`, `llama-state-prefix-parallel-v1`, `llama-independent-batch-v1`, not misleading native PyTorch identifiers. Imported Torch/Transformers/adapter metadata is retained; our own outputs omit inapplicable versions. `backend` includes both `llama-cpp-2/0.1.156` and full llama.cpp commit.

Prefix hash, if emitted, is SHA-256 of Python `json.dumps(prefix_token_ids)` bytes (comma-space list), matching serial.py, not a text-prefix hash. `cache_hit=false` describes fresh prefill; do not invent a cross-request hit. Result time fields are mode-appropriate as described in section 6.

Postprocessing changes `probabilities`, `choice`, primitive expectations and confidence, not base-run raw prompt/logit/vocabulary fields. Presence of postprocess makes this distinction explicit: `raw_fields_reference=0` points to the identity permutation sample and every other sample carries its own prompt hash/tokens/logits/timings. For transformed single-row operations, top-level total_seconds measures the entire operation and forward_seconds sums its model work; per-sample timings remain available. Shared transformed groups retain group-level timing instead of fabricated row timings. A transformation is **not** a pure direct baseline. Keep `prompt_version=direct-options-v1` because the per-run prompt algorithm is unchanged; version the postprocess separately. Never present averaged logits or a synthetic hash as if they came from a single forward pass.

Command result schemas use stable separate tags: `openjev-models-v1` (models array with registry metadata, cached/verified and support status), `openjev-model-path-v1` (id/path/integrity), `openjev-help-v1` (command/usage/text), `openjev-version-v1` (version/build), `openjev-write-summary-v1` (path/written/failed), `openjev-eval-v1` (section 9 fields), `openjev-bench-v1` (section 9 timings/config), and `openjev-calibration-v1` (section 10 fields). Shared execution emits one Readout per question, not an incompatible envelope.

## 8. CLI grammar and I/O contract

```
openjev [GLOBAL] decide --question TEXT... --option TEXT... [--option-id ID...] [STATE] [--id ID]
openjev [GLOBAL] noul --question TEXT [STATE] [--id ID]
openjev [GLOBAL] score --question TEXT --level TEXT... [--level-id ID...] [--level-value NUMBER...] [STATE] [--id ID]
openjev [GLOBAL] ask (--json ROW | --input FILE | stdin-one-JSON-row)
openjev [GLOBAL] run [--mode direct|serial|shared|batch] [--input FILE | stdin-JSONL] [--output NEWFILE]
openjev models [list | pull ID [--repair] | path ID]
openjev [GLOBAL] models probe ID --mode shared|batch
openjev [GLOBAL] eval --fixture authored144|perturbations108 [--predictions FILE] [--compare-to browser-ladder]
openjev [GLOBAL] bench --state-file FILE --questions FILE [--repeats N] [--output NEWFILE]
openjev calibrate --input LABELLED-LOGITS-JSONL [--output NEWFILE]
```

STATE is exactly one of `--state TEXT`, `--state-file FILE`, `--state-json JSON`, `--state-json-file FILE`, or piped UTF-8 stdin (text). Files/text stdin are not trimmed or guessed as JSON; structured state requires explicit flags. A TTY with no state is a validation error, not an interactive prompt. Explicit state ignores stdin rather than blocking to read it. `ask` accepts a full validated Decision; no implicit state option.

Global flags: `--model` (registry ID, local GGUF path, or `hf:OWNER/REPO@40HEX:FILENAME`), `--model-sha256`, `--template-profile`, `--cache-dir`, `--offline`, `--device cpu|metal|cuda`, `--gpu-layers all|N`, `--threads N`, `--n-ctx N`, `--max-tokens N`, `--max-context-tokens N`, `--n-batch N`, `--n-ubatch N`, `--max-sequences N`, `--require-shared`, `--permute N`, `--seed U64`, `--temperature T` or `--calibration FILE` (mutually exclusive), `--confidence`, `--pretty`, `--quiet`. Reject flags irrelevant to a command instead of ignoring them. Default run mode direct; model minicpm5-2b; temperature 1; permutation count 1.

Unspecified decision ID is `decision-1`; multiple decide questions become `decision-1`..`decision-N` or `{id}/1`..`{id}/N`. Repeated questions use identical shared state/options and request shared mode automatically. Generated option IDs are `option-1`..K (not slugs); explicit option-id list must have exactly K entries and permits empty IDs subject to uniqueness. Score uses `level-1`..K. Flags support quoted empty descriptions. Label/level counts and values must align; no duplicate options are silently deduplicated.

Run ignores blank JSONL lines, rejects empty input and duplicate IDs. Shared run requires every state's serialized bytes identical; heterogeneous state input is an error, not silently grouped. All input validation occurs before model loading. File output is create-only like Python (`create_new`, never overwrite), including eval/bench/calibration reports. Protect against same input/output file and symlink surprises. If row scoring fails after some successes, run emits an ErrorRecord for that ID, continues safe independent rows in input order and exits 1; shared failure falls back for the whole group or emits errors for the group. Fatal parse/validation errors stop before inference. Eval consumes ErrorRecords as invalid, not absent success.

**stdout is always machine-readable JSON or JSONL, including help/version.** Intercept clap's `try_parse` errors; never call its printing/exiting path. `--help` emits `{schema:"openjev-help-v1",...,text:"..."}` to stdout and exits 0; version similarly. Usage/input errors produce one ErrorRecord on stderr, stdout empty, exit 2. Runtime/fatal model errors produce one ErrorRecord on stderr, stdout empty unless earlier JSONL results exist, exit 1. Per-row run errors are JSONL ErrorRecords in the result stream and a concise stderr diagnostic. Progress, native logs, warnings and download bars only go to stderr; install a native logging callback and test captured streams. Broken pipe exits nonzero cleanly without panic text. `--output` directs JSONL/results to new file and leaves one JSON write-summary on stdout. `--pretty` is legal only for a single object/report, rejected for multirow JSONL. No human banners, bare model paths or timing tables on stdout.

## 9. Parity, evaluation and performance

### 9.1 Strict reference gate

`browser-ladder-qwen3-0.6b.predictions.jsonl` has **252 unique predictions**, not 144. Both fixtures together have 252 unique IDs, three families with 84 combined rows each. Authored subset has 144, perturbation subset 108. M3 indexes predictions by ID, rejects duplicates, explicitly selects the 144 authored IDs, checks every one exists, and verifies ordered option_ids, prompt_sha256, input_tokens and answer_token_ids. It does **not** zip first 144 records or reject the known extra 108 as unknown for this documented subset selection. General eval still rejects unknown IDs unless using an explicit selected reference subset.

Gate: 144/144 identical hashes **and** 144/144 identical token counts, all slots/boundaries valid, all finite readouts. Any mismatch fails M3 before interpreting quality. Already-proven text hashes do not excuse running this against GGUF tokenization. Report all 108 perturbation hashes/token counts separately as extended coverage.

Numerical BF16-vs-Q8_0 reference comparison: measure slot-logit MAE, RMSE, max absolute difference, distribution differences and first-argmax agreement with mismatch IDs and margins, along with native/hardware/config. Do not assert guessed 98% agreement or small logit MAE as an established fact. M3 review must explicitly accept measured quantization differences or stop for diagnosis; exact prompt/token parity alone is not numerical equivalence. Freeze any regression tolerance only after reviewing baseline evidence, and never silently loosen it on a later run. MiniCPM/Qwen3.5 missing row-level BF16 predictions are an explicit limitation; published aggregates are not row-logit goldens.

M5 same-artifact shared/direct tolerance is a different question: predeclare initial max absolute slot-logit tolerance **1e-3**, max probability delta **1e-4**, and identical first argmax. Probe all three models on CPU/Metal configurations, binary and 3/16-way choices, short/long repeated states, ragged suffixes, chunk boundaries, 1/2/21 branches, repeated copy-clear cycles, changed-state isolation and independent-batch isolation. Record per-model maxima, failures and repeatability. Eligibility requires passing these initial bounds; if not, default serial fallback. A justified per-model alternative tolerance requires Astra review, held-out probe validation and an explicit versioned manifest entry before enablement—never an automatic multiplier. Review artifacts include both passing and failing probes and time spent falling back. New native/device/model configuration invalidates old eligibility.

### 9.2 Eval semantics

Port and test `indexed`, `vector`, `align`, `basic`, and the needed probability/coverage subset of `summarize`/`evaluate`; do not claim all upstream bootstrap/risk-screen machinery is implemented. `openjev-eval-v1` includes `available_gold`, `scored`, `evaluated`, `coverage`, `missing`, `invalid`, `family_results`, `mean_family_balanced_accuracy`, `mean_family_macro_f1`, `overall` (plain accuracy), `errors`, fixture/model/config hashes and limitations. Each family/overall summary includes `n`, `accuracy`, `balanced_accuracy`, `macro_f1`, `source_groups`, `invalid_or_missing`, `probability_rows`, `probability_coverage`, `nll`, `brier`, `nll_valid_distributions_only`, `brier_valid_distributions_only`, `nll_probability_floor`. Unimplemented bootstrap/reliability fields are omitted with a documented subset, not approximated under upstream names.

- Keep **all declared gold rows** in accuracy and class recall denominators; missing/invalid/unparsed are failures. Unknown prediction IDs and duplicates are errors.
- Align distributions by semantic option ID. For arrays with option_ids, require unique exact ID set; for dict distributions require exact keys. Reject bools, out-of-range/nonfinite values, wrong lengths, mass error >1e-4. A native `prediction_id` without probabilities is allowed by import, never converted to fake one-hot probability.
- Gold class is `options[label].id`, not positional letter. First argmax in gold option order breaks ties after alignment. Per-family balanced accuracy is mean recall over represented **gold semantic classes**; headline is arithmetic mean of family balanced accuracies, **not pooled** balanced accuracy. Macro-F1 matches upstream represented-class rule. Report plain accuracy separately.
- NLL uses floor 1e-12, Brier is sum over classes without dividing by K. Top-level family NLL/Brier are null if any row lacks a valid distribution; expose valid-distributions-only means and coverage. Never silently drop failures and present the remaining score as complete.
- Perturbation stability joins `provenance.base_id` to the authored original, verifies `source_group_id`/`group_id` relation and aligns option IDs. Report modal agreement, mean total-variation distance, eligible/missing/invalid pair counts, by variant and equal source-group macro. Criterion reversals/evidence changes in authored144 are not equivalent semantic variants.
- `--compare-to browser-ladder` selects known IDs from Qwen row-level reference where available; elsewhere report only the exact published aggregate, labeled native BF16 versus quantized local GGUF. Authored ladder 0.4403525/0.6862540/0.8132382 is not an acceptance promise. No TypeSafe 711-row or live-Jev claims; TypeSafe remains out of v1 fixture eval.

### 9.3 This-machine benchmark

M6 creates new raw JSONL and report paths, hashes them, records OS/CPU/GPU/RAM, Rust/compiler/native SHA/features, GGUF/hash, layers, threads, context, batches and probe policy. Measure CPU-only and Metal on **this machine**, not upstream browser/RTX3090 numbers. Load each model once per device, verify cached artifact once, warm up separately, then at least 5 measured repetitions (configurable); report individual samples, median/p95, decisions/sec, input/prefix/suffix tokens, encoding/context/prefill/copy/suffix/readout/wall timing and direct/shared ratio. Alternate direct/shared order to reduce warm-cache bias. Never double-count shared_timing repeated on 21 rows.

Use one owned state ×21 questions in the documented shape777 geometry; construct/export an owned fixture outside reference if no full fixture is present. Include at least one small prompt and a longer state within token caps. Document no equivalence to hidden Jev documents/token lengths/hardware. Distinguish genuine shared acceleration from `effective_mode=serial`; unavailable shared speedup is null with reason, not labeled speedup=1. Exclude downloading/loading/file writing from scoring latency, but publish those separately. Any model/device that cannot run gets an explicit failed/unavailable result, not a skipped cell disguised as complete coverage.

## 10. Permutation and calibration (M7)

Temperature T must be finite and >0. Apply softmax(raw slot logits/T) per run; mass/lognormalizer and stored raw logits remain unscaled. With one identity run and T=1, omit postprocess. With explicit transformation emit full provenance even if T numerically equals 1. Calibration does not remove the uncalibrated-confidence honesty string.

`--permute N` means **N total unique orders including identity**, default 1, require `1 <= N <= min(K!,64)`. Deterministic `sha256-factoradic-v1`: identity first; hash the ASCII domain `openjev-permutation-v1` followed by K as u64 big-endian, UTF-8 decision-ID byte length as u64 big-endian, ID bytes, seed u64 big-endian and counter u64 big-endian (starting 0). Interpret the first eight digest bytes as u64 big-endian. Accept only values below `floor(2^64/K!)*K!` (compute bound in u128), reduce modulo K!, unrank lexicographically, skip repeats. Stop after 100000 candidate draws with an explicit error rather than loop indefinitely. Freeze test vectors. Each run re-renders/revalidates boundary, temperature-scales and softmaxes; align to original option IDs, arithmetic mean probabilities, derive first original-index argmax. Store every permutation index map, prompt hash/token count, slot IDs/logits, raw distribution and timing in samples; total cost is visible. RawSample probabilities are **raw T=1 probabilities** so recalibration is reproducible; final aggregation applies recorded T from raw logits. Do not pretend permutation averaging is generation-free single-pass speed.

`calibrate` input rows contain unique id, group_id, option_ids, raw option_logits, gold label/label_id, model artifact fingerprint/profile/prompt_version, and split metadata. Validate K and label alignment; reject missing/invalid rows rather than fitting a biased subset. Require a caller-designated calibration split, disjoint by source group from evaluation IDs; never train T on authored144 test and present it as test quality. Fit a single T by minimizing mean NLL: log(T) in [log(0.05),log(20)], 81-point uniform log grid followed by 80 deterministic golden-section steps around the best interior grid point (include endpoints, choose smallest T on exact ties). Record boundary optimum, pre/post calibration NLL, count and rejected-count (must be 0), data SHA, IDs/groups hash, search bounds/iterations, model+prompt fingerprint, temperature, `schema=openjev-calibration-v1`, calibration ID hash. Compare to T=1 and retain T=1 if no improvement. Application rejects mismatched artifact/profile/prompt version. Temperature-after-permutation fitting is not supported; calibration is per-run raw logits, then permutation averaging as specified. Reliability diagrams/conformal methods remain future work.

## 11. Milestones and review/commit discipline

**Every M1–M7 completion gate** runs `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace`, plus feature/device-specific equivalents and the milestone evidence below. Record exact commands, feature set, exit status and tests run; default backend-disabled tests cannot stand in for native-feature tests. Integration tests are opt-in (`integration` feature plus explicit environment/cache availability), never download on ordinary cargo test. An unexecuted gated test is “not run”, not “passed”. Update `docs/PROGRESS.md`, synchronize todo status without deleting the brief, request Astra design/code review, fix findings and rerun affected checks, then commit that milestone. Do not combine unfinished later gates into the same success claim. Phase A has no Cargo workspace to test and records these checks as not applicable, not green.

| Milestone | Deliverables and gate |
|---|---|
| **M1 core/skeleton** | Workspace with no native build on default checks; exact types/validation/serializer/profile rendering, schema export, slots abstraction and numeric/eval hand tests, CLI parse skeleton/fake backend test double only in tests, MIT/THIRD_PARTY. All 144 text hashes and serializer differential tests pass offline; no claimed token parity. Include permitted empty option ID/description, recursive float rejection, insertion-order, exact strings and deterministic ties. Commit after review. |
| **M2 three GGUF smoke** | Registry/cache with pinned size/hash, one-time downloads, native adapter, CPU+Metal build instructions. Load/warm/score all three exact files, finite full-vocab/slot readout, disabled-thinking prompt/slot checks; archive per-model stdout/stderr/config metadata. Run malformed/corrupt/offline/concurrent-cache tests. No model substitution. Record a failed load in docs/RESULTS.md and continue as explicitly requested; mark that model unsupported/unverified and report M2 as attempted with failures, not three-model success. A Qwen load failure still prevents passing the mandatory M3 parity gate. This is the first runtime architecture evidence. |
| **M3 strict parity** | Qwen Q8_0 all144 exact hashes, token counts, slot IDs and boundaries; detailed BF16-vs-GGUF raw-logit/probability/argmax report, explicit Astra numerical review. Include negative tests demonstrating altered spacing/BOS/indexing fails. Separate optional 108 coverage. No approximate hash/token acceptance. |
| **M4 CLI** | All core scoring/input/model commands and stdout-only-JSON policy, streaming/create-only output, Noul/Score, batch/direct primitives. Native worker lifecycle/error handling. Process tests capture help/version/errors/progress, stdin precedence, all subcommands, multi-question ordering, broken pipe, exit codes, no native stdout leakage. Eval/bench/calibrate grammar may be present but must explicitly report not-yet-implemented until their milestones. |
| **M5 shared sequence copy** | Correct immutable-prefix/copy/suffix/clear path and batch isolation; per-model/device probes and frozen tolerance decisions, subprocess crash handling, required-shared error and visible serial fallback. Exercise chunk-local logits indices and repeated state changes. A hybrid's serial fallback is accepted safe behavior with recorded unsupported shared status, not proof of shared performance. |
| **M6 eval/bench CPU+Metal** | Eval subset parity against Python on hand cases and both full fixtures; missing/invalid denominator and family/class weighting tests. This-machine per-model CPU/Metal measurements and 1×21 shared results, row evidence and docs/RESULTS.md. Report quantization gap, runtime limitations and absent BF16 row-level sources. Publish no guessed quality/speedup. |
| **M7 postprocessing/docs/CI** | Permutation provenance/order/alignment tests, deterministic temperature fitting and split-isolation tests, README/PROMPT/schema/limitations/reference credits finalized. Linux x86 CPU + macOS arm64 CI matrix: offline core/CLI checks mandatory, native builds and opt-in cached integration jobs explicit; CUDA instructions without pretending CUDA CI ran. Verify packaged fixtures/manifest/license and no GGUFs/caches enter git. Full review then commit. |

### Phase A evidence and remaining blockers

Confirmed: real registry/source versions; current upstream/submodule pins; exact API and build behavior; profile rendering against fetched Jinja; 144+108 Qwen prompt hashes; HF artifact sizes/LFS SHA-256; upstream validation and eval semantics. No reference file was changed, no Rust file generated, no model downloaded.

Not yet established (must not be asserted): native compilation, dependency-lock compatibility, GGUF template metadata/token counts, all three load/forward paths on this machine, same-artifact direct/shared numeric parity, BF16 quantization agreement, CPU/Metal performance, or lower MSRV. These are M1–M6 execution gates with the explicit failure policies above. The reference snapshot lacks MiniCPM/Qwen3.5 row-level BF16 predictions; this blocks their row-logit comparison, not their smoke/direct functionality or aggregate-ladder comparison. No design ambiguity requires silently guessing a fallback.

## Appendix A. Machine-readable emitted-readout JSON Schema

The following Draft 2020-12 schema is the M1 export source (documentation, not implemented validation). Cross-field vector lengths, argmax consistency, normalization, integer-state restrictions and mode/postprocess semantics additionally require the checks specified above. It validates our emitted superset, not legacy Python imports that legitimately lack newly required fields; import compatibility uses the field definitions with only upstream-required fields required. JSON numeric syntax excludes nonfinite values. Additional command report schemas are exported with their own tags in M4/M6/M7.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "urn:openjev:readout:v1",
  "oneOf": [{"$ref":"#/$defs/Readout"},{"$ref":"#/$defs/ErrorRecord"}],
  "$defs": {
    "Hash": {"type":"string","pattern":"^[0-9a-f]{64}$"},
    "Nonnegative": {"type":"number","minimum":0},
    "PositiveInt": {"type":"integer","minimum":1},
    "Token": {"type":"integer","minimum":0},
    "Probability": {"type":"number","minimum":0,"maximum":1},
    "Mode": {"enum":["direct","serial","shared","batch"]},
    "Profile": {"enum":["qwen3","qwen3.5","minicpm5"]},
    "Ids": {"type":"array","minItems":2,"maxItems":16,"uniqueItems":true,"items":{"type":"string"}},
    "Tokens": {"type":"array","minItems":2,"maxItems":16,"uniqueItems":true,"items":{"$ref":"#/$defs/Token"}},
    "Numbers": {"type":"array","minItems":2,"maxItems":16,"items":{"type":"number"}},
    "Probabilities": {"type":"array","minItems":2,"maxItems":16,"items":{"$ref":"#/$defs/Probability"}},
    "ModelMetadata": {
      "type":"object","additionalProperties":false,
      "required":["id","source","revision","file","quant","backend","artifact_sha256","integrity","dtype","template_profile","template_sha256","template_override"],
      "properties": {
        "id":{"type":"string"},"source":{"type":"string"},"revision":{"type":"string"},"file":{"type":"string"},
        "quant":{"type":"string"},"backend":{"type":"string"},"artifact_sha256":{"$ref":"#/$defs/Hash"},
        "integrity":{"enum":["manifest-sha256","caller-sha256","local-unverified"]},"dtype":{"type":"string"},
        "native_reference":{"type":"object","additionalProperties":false,"required":["source","revision","dtype"],"properties":{"source":{"type":"string"},"revision":{"type":"string","pattern":"^[0-9a-f]{40}$"},"dtype":{"const":"bfloat16"}}},
        "template_profile":{"$ref":"#/$defs/Profile"},
        "template_sha256":{"anyOf":[{"$ref":"#/$defs/Hash"},{"type":"null"}]},
        "template_override":{"type":"boolean"},"serving_config":{"type":"string"},
        "adapter":{"type":["string","null"]},
        "adapter_sha256":{"anyOf":[{"$ref":"#/$defs/Hash"},{"type":"null"}]},
        "adapter_revision":{"type":["string","null"]},"torch_version":{"type":"string"},"transformers_version":{"type":"string"}
      }
    },
    "ExecutionMetadata": {
      "type":"object","additionalProperties":false,
      "required":["requested_mode","effective_mode","fallback_reason","device","device_name","gpu_layers_requested","gpu_layers_actual","threads","n_ctx_requested","n_ctx_actual","max_tokens","n_batch","n_ubatch","n_seq_max","kv_unified","waves","probe_id","run_id"],
      "properties": {
        "requested_mode":{"$ref":"#/$defs/Mode"},"effective_mode":{"$ref":"#/$defs/Mode"},
        "fallback_reason":{"type":["string","null"]},"device":{"enum":["cpu","metal","cuda"]},"device_name":{"type":"string"},
        "gpu_layers_requested":{"anyOf":[{"const":"all"},{"type":"integer","minimum":0}]},
        "gpu_layers_actual":{"type":"integer","minimum":0},"threads":{"$ref":"#/$defs/PositiveInt"},
        "n_ctx_requested":{"anyOf":[{"$ref":"#/$defs/PositiveInt"},{"type":"null"}]},
        "n_ctx_actual":{"$ref":"#/$defs/PositiveInt"},"max_tokens":{"$ref":"#/$defs/PositiveInt"},
        "n_batch":{"$ref":"#/$defs/PositiveInt"},"n_ubatch":{"$ref":"#/$defs/PositiveInt"},"n_seq_max":{"$ref":"#/$defs/PositiveInt"},
        "kv_unified":{"type":"boolean"},"waves":{"$ref":"#/$defs/PositiveInt"},
        "probe_id":{"type":["string","null"]},"run_id":{"type":"string"},"group_id":{"type":"string"}
      }
    },
    "SharedTiming": {
      "type":"object","additionalProperties":false,
      "required":["total_seconds","encode_seconds","prefix_tokens","prefill_seconds","replicate_seconds","suffix_forward_seconds","batch_size","true_suffix_tokens","padded_suffix_tokens"],
      "properties": {
        "total_seconds":{"$ref":"#/$defs/Nonnegative"},"encode_seconds":{"$ref":"#/$defs/Nonnegative"},
        "prefix_tokens":{"type":"integer","minimum":0},"prefill_seconds":{"$ref":"#/$defs/Nonnegative"},
        "replicate_seconds":{"$ref":"#/$defs/Nonnegative"},"suffix_forward_seconds":{"$ref":"#/$defs/Nonnegative"},
        "batch_size":{"$ref":"#/$defs/PositiveInt"},"true_suffix_tokens":{"type":"integer","minimum":0},"padded_suffix_tokens":{"type":"integer","minimum":0}
      }
    },
    "RawSample": {
      "type":"object","additionalProperties":false,
      "required":["permutation","option_ids","answer_token_ids","option_logits","probabilities","prompt_sha256","prompt_version","input_tokens","allowed_token_mass","full_vocab_argmax_id","full_vocab_log_normalizer","execution"],
      "properties": {
        "permutation":{"type":"array","minItems":2,"maxItems":16,"uniqueItems":true,"items":{"type":"integer","minimum":0,"maximum":15}},
        "option_ids":{"$ref":"#/$defs/Ids"},"answer_token_ids":{"$ref":"#/$defs/Tokens"},
        "option_logits":{"$ref":"#/$defs/Numbers"},"probabilities":{"$ref":"#/$defs/Probabilities"},
        "prompt_sha256":{"$ref":"#/$defs/Hash"},"prompt_version":{"const":"direct-options-v1"},"input_tokens":{"$ref":"#/$defs/PositiveInt"},
        "allowed_token_mass":{"$ref":"#/$defs/Probability"},"full_vocab_argmax_id":{"$ref":"#/$defs/Token"},"full_vocab_log_normalizer":{"type":"number"},
        "forward_seconds":{"$ref":"#/$defs/Nonnegative"},"total_seconds":{"$ref":"#/$defs/Nonnegative"},
        "execution":{"$ref":"#/$defs/ExecutionMetadata"},"shared_timing":{"$ref":"#/$defs/SharedTiming"}
      }
    },
    "Postprocess": {
      "type":"object","additionalProperties":false,
      "required":["version","temperature","calibration_id","permutation_count","seed","permutation_algorithm","aggregation","raw_fields_reference","base_probabilities","samples"],
      "properties": {
        "version":{"const":"openjev-postprocess-v1"},"temperature":{"type":"number","exclusiveMinimum":0},
        "calibration_id":{"type":["string","null"]},"permutation_count":{"type":"integer","minimum":1,"maximum":64},
        "seed":{"type":"integer","minimum":0,"maximum":18446744073709551615},
        "permutation_algorithm":{"const":"sha256-factoradic-v1"},"aggregation":{"const":"mean-id-aligned-probabilities"},
        "raw_fields_reference":{"const":0},"base_probabilities":{"$ref":"#/$defs/Probabilities"},
        "samples":{"type":"array","minItems":1,"maxItems":64,"items":{"$ref":"#/$defs/RawSample"}}
      }
    },
    "Readout": {
      "type":"object","additionalProperties":false,
      "required":["schema","id","primitive","choice","choice_index","option_ids","probabilities","option_logits","answer_token_ids","allowed_token_mass","full_vocab_argmax_id","full_vocab_log_normalizer","input_tokens","prompt_sha256","prompt_version","model","readout","probability_status","limitations","execution"],
      "properties": {
        "schema":{"const":"openjev-readout-v1"},"id":{"type":"string","minLength":1},"primitive":{"enum":["choice","noul","score"]},
        "choice":{"type":"string"},"choice_index":{"type":"integer","minimum":0,"maximum":15},
        "option_ids":{"$ref":"#/$defs/Ids"},"probabilities":{"$ref":"#/$defs/Probabilities"},"option_logits":{"$ref":"#/$defs/Numbers"},"answer_token_ids":{"$ref":"#/$defs/Tokens"},
        "allowed_token_mass":{"$ref":"#/$defs/Probability"},"full_vocab_argmax_id":{"$ref":"#/$defs/Token"},"full_vocab_log_normalizer":{"type":"number"},"input_tokens":{"$ref":"#/$defs/PositiveInt"},
        "forward_seconds":{"$ref":"#/$defs/Nonnegative"},"total_seconds":{"$ref":"#/$defs/Nonnegative"},
        "prompt_sha256":{"$ref":"#/$defs/Hash"},"prompt_version":{"const":"direct-options-v1"},"model":{"$ref":"#/$defs/ModelMetadata"},
        "readout":{"enum":["native full-vocabulary last-position logits restricted to declared answer slots","native selected suffix-position logits","native-state-prefix-cache-last-position","native-full-vocabulary-last-position"]},
        "probability_status":{"const":"conditional option score; uncalibrated as decision confidence"},
        "limitations":{"type":"array","items":{"type":"string"}},"execution":{"$ref":"#/$defs/ExecutionMetadata"},
        "confidence":{"$ref":"#/$defs/Probability"},"confidence_status":{"const":"normalized margin; uncalibrated"},"p_yes":{"$ref":"#/$defs/Probability"},
        "level_values":{"$ref":"#/$defs/Numbers"},"expected_value":{"type":"number"},"argmax_level":{"type":"string"},
        "cache_hit":{"type":"boolean"},"prefix_tokens":{"type":"integer","minimum":0},"prefix_sha256":{"$ref":"#/$defs/Hash"},
        "prefill_seconds":{"$ref":"#/$defs/Nonnegative"},"copy_seconds":{"$ref":"#/$defs/Nonnegative"},"suffix_forward_seconds":{"$ref":"#/$defs/Nonnegative"},
        "shared_timing":{"$ref":"#/$defs/SharedTiming"},"postprocess":{"$ref":"#/$defs/Postprocess"}
      },
      "dependentRequired":{"confidence":["confidence_status"],"confidence_status":["confidence"]},
      "allOf":[
        {"if":{"properties":{"primitive":{"const":"noul"}}},"then":{"required":["p_yes"]},"else":{"not":{"required":["p_yes"]}}},
        {"if":{"properties":{"primitive":{"const":"score"}}},"then":{"required":["level_values","expected_value","argmax_level"]},"else":{"not":{"anyOf":[{"required":["level_values"]},{"required":["expected_value"]},{"required":["argmax_level"]}]}}}
      ]
    },
    "ErrorRecord": {
      "type":"object","additionalProperties":false,"required":["schema","error"],
      "properties": {
        "schema":{"const":"openjev-error-v1"},"id":{"type":"string"},"parse_status":{"const":"unparsed"},
        "error":{"type":"object","additionalProperties":false,"required":["code","message","details"],"properties":{"code":{"type":"string"},"message":{"type":"string"},"details":{"type":"object"}}}
      }
    }
  }
}
```

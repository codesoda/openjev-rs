# Progress

## Phase A — design/source adjudication (reviewed)

- Created `docs/PLAN.md` before any Rust source or Cargo workspace.
- Read the Python scoring/validation/prefix-cache/CLI/evaluation sources, manifests, fixtures/golden shape, browser worker and GLiNER integration note. Preserved reference unchanged.
- Registry correction: requested llama-cpp-2 0.1.157 is not published. Downloaded real 0.1.156 wrapper/sys crates via `cargo info`; inspected sys build.rs completely. Selected exact 0.1.156 / llama.cpp `e79e4bf660e19f2ad851e06c6913f7a8c5852621`. Git main labels itself 0.1.157 but does not change that submodule pin.
- Confirmed current dependency versions, including hf-hub 1.0.0's blocking builder API. Documented exact llama signatures, batch-local logits indices, full sequence copy/removal behavior, context allocation, Metal defaults and unavailable LLAMA_CPP_PATH override.
- Fetched only small native tokenizer/template/config artifacts and HF API metadata. No GGUFs downloaded. Confirmed artifact LFS hashes/sizes for all three pins.
- Python/Jinja probe: restricted disabled-thinking rendering matches fetched templates on 252/252 rows for each model. Restricted Qwen3 rendering matches 144/144 authored and 108/108 perturbation committed prompt hashes. This does not verify GGUF token counts or logits.
- Closed all six section-8 design decisions, with explicit model failure, template, numeric-state, hybrid serial-fallback, accelerator and licensing policies. Recorded missing MiniCPM/Qwen3.5 row-level BF16 predictions as a reference limitation.
- Defined workspace isolation, public API/ownership, CLI JSON-only policy, readout superset, evaluation denominators, cache integrity, postprocess provenance and M1–M7 gates.

Validation: `git diff --check` passed; repository Rust-file search found none; `git diff -- reference` is empty. The embedded JSON Schema parses as JSON and all 59 local references resolve. Full Draft 2020-12 validation was not run because Python's `jsonschema` module is unavailable (the attempted check reported ModuleNotFoundError). Cargo fmt/clippy/test are **not applicable yet**, not claimed passed. Native build, model smoke, token/logit parity, shared correctness and CPU/Metal performance are explicitly unrun.

Parent/Astra review: read the full plan, spot-checked exact wrapper signatures and sys build-source selection, and independently reproduced published Qwen fixture evaluation aggregates. Clarified that todo.md/user requirements retain precedence and that M2 records failed model loads and continues (rather than stalling); M3 exact parity remains non-negotiable. Approved Phase A for commit and M1 implementation by GPT 5.6 Sol. No runtime evidence is inferred from design approval.

## M1 — core/skeleton (approved after review remediation)

Implemented the three-crate Rust 1.95 workspace and lockfile. Direct dependencies are pinned to the reviewed versions, including optional `llama-cpp-2`/`llama-cpp-sys-2` 0.1.156 and hf-hub 1.0.0. Default workspace members are core/CLI; `openjev-llama` is backend-disabled by default, and the ordinary dependency tree contains no llama/hf-hub native build. Cargo resolved/downloaded registry crates and rustup installed missing 1.95 components; no GGUF, tokenizer, model, backend build, CMake invocation, or inference occurred.

Core now has validated `Decision`, `Question`, and insertion-ordered `StateValue`; unknown input metadata retention; strict raw JSON duplicate/lone-surrogate/float/overflow rejection with paths; Python-spaced integer-only JSON serialization; exact Qwen3/Qwen3.5/MiniCPM5 restricted rendering; typed readout/model/execution/shared/postprocess schema containers with finite-output serializers; generic slot verification; f64 numerics; Noul/Score adapters; and the documented evaluation probability/coverage subset. `schemas/readout-v1.schema.json` is parsed and compared structurally with PLAN Appendix A in a test. M7 postprocessing algorithms are not implemented; only the M1-required schema containers exist.

The CLI parses the documented command surface. Help and version are intercepted as JSON stdout; usage/backend errors are JSON stderr; scoring emits no fake readout and returns `backend_unavailable`. `LICENSE`, `THIRD_PARTY.md`, and the credited synthetic prompt oracle/regeneration instructions are present. `reference/` remains unchanged. Pragmatic M1 layout differences are non-semantic: slot logic has its own `slots.rs`, postprocess has schema-only data types rather than an algorithm module, and the CLI remains consolidated in `args.rs`/`lib.rs`/`main.rs` until M4.

Offline evidence:

- 144/144 authored Qwen3 prompt SHA-256 values match the 252-row reference prediction file by semantic ID.
- 108/108 perturbation prompt SHA-256 values also match by ID; this is text-hash coverage only, not token or logit parity.
- Python 3.9.6 stdlib differential tests cover recursive insertion order, all U+0000–U+001F controls, quote/backslash/slash punctuation, Unicode including U+2028, signed i64 minimum and unsigned u64 maximum. A separate Python upstream-evaluator differential covers semantic-ID reordering, first ties, invalid distributions, and missing-row denominators.
- Hand tests cover empty option IDs/descriptions, whitespace preservation, nested float forms (`1.0`, `1e0`, `-0.0`), integer overflow, duplicate keys, slot multitoken/roundtrip/collision/vocabulary/boundary failures, masked vocabulary entries, first ties, nonfinite output rejection, Noul/Score adaptation, missing/invalid rows, and unknown/duplicate evaluation IDs.
- Actual CLI probes: `cargo run -q -- --help` exited 0 with `openjev-help-v1` on stdout and 0 stderr bytes; a backend-disabled `decide` exited 1 with empty stdout and `backend_unavailable` JSON on stderr.

Final local gate commands, all exit 0:

- `cargo fmt --all`
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace` — superseded by the post-gate-remediation run below.
- `cargo clippy --all-targets -- -D warnings` — default-member scope passed.
- `cargo test` — superseded by the post-gate-remediation run below.

Native compilation, all GGUF loading/scoring, tokenizer counts/slot IDs against a real model, model/logit parity, shared KV behavior, downloads/cache integrity, and CPU/Metal performance remain explicitly unrun for M2+; M1 must not be read as evidence for them. No commit was created.

### Astra M1 gate remediation

The first Astra code gate was **blocked** on five reproduced defects. All five are now fixed locally and await independent parent/Astra re-review:

1. Public JSON ingestion now has one documented 128-container nesting limit. The hand parser returns a path-bearing typed serialization error before deeper recursion; iterative already-built-`Value` validation and disposal avoid recursively walking/dropping rejected 5,000-level values. The first fix used a bounded Serde visitor; the lexical-number follow-up below replaced that visitor with bounded strict parsing of captured raw JSON.
2. `StateValue`, `Decision`, and `Question` public `Deserialize` implementations initially gained one insertion-order, duplicate-aware, integer-only bounded visitor instead of first materializing an unchecked `Value`. Nested duplicates were rejected with paths through the requested public types. The follow-up below retains those properties while preserving source numeric spelling. The documented boundary remains explicit: `TryFrom<Value>` cannot recover duplicates or lexemes already lost by another parser.
3. `Readout::validate` now checks exact schema/prompt/honesty constants and limitations; hashes; positive token/execution configuration; mode/fallback/readout consistency; finite normalized values; normalized-margin consistency; Noul `yes,no` identity; Score field exclusivity; model metadata; shared timing; and schema-only raw-sample/postprocess provenance. `jsonschema = 0.56.0` was inspected in the local registry and added as a pinned, default-feature-disabled dev dependency. Real serialized Choice/Noul/Score readouts pass the exported Draft 2020-12 schema; mutated contract, primitive, timing, and provenance cases fail code validation. No M7 algorithm was added.
4. Allowed-token mass now computes numerator and denominator from one common shifted origin, avoiding cancellation between huge rounded log-normalizers. Base, `+1e20`, and `-1e20` f32 shifts all reproduce mass 0.5 while contradiction/nonfinite handling remains explicit.
5. `NumericReadout.choice_index` now uses first argmax of the emitted f64 probabilities. The `[0.0, 1e-20]` rounded-probability tie selects index 0, and a readout built from that numeric result passes code and JSON Schema validation.

Post-remediation commands, all exit 0:

- `cargo fmt --all`
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace` — 43 tests passed (4 CLI, 27 core unit, 11 core integration, 1 backend skeleton), 0 failed; doc-test harnesses contained 0 tests.
- `cargo clippy --all-targets -- -D warnings` — default-member scope passed.
- `cargo test` — 42 tests passed, 0 failed; the non-default backend crate's one unit test is intentionally absent from this default scope.

### Raw numeric-lexeme follow-up

The independent follow-up review found one remaining M1 blocker: serde_json presents lexical integer `-0` to a generic numeric visitor as `f64`, so `StateValue::parse_json("[-0]")` normalized successfully while `serde_json::from_str::<StateValue>("[-0]")` rejected it. Accepting floating negative zero would also have incorrectly admitted forbidden `-0.0`/`-0e0`.

Inspected serde_json 1.0.151's real `raw.rs`, `de.rs`, `read.rs`, and `value/de.rs` before changing the route. `Box<RawValue>` preserves complete source text for `from_str`/`from_slice`, buffers it for `from_reader`, and uses serde_json's iterative `ignore_value` scanner rather than its ordinary recursive visitor. Owned `Value` deserialization reserializes the available tree with `Value::to_string`; it preserves available insertion order and values but cannot recreate duplicates or the lexical distinction between integer `-0` and floating `-0.0`. The generic numeric visitor was removed. Public JSON `Deserialize` entry points now capture a boxed raw value and feed its text into the existing depth-limited `parse_json_strict`; `TryFrom<Value>` remains the validated already-parsed-value path.

Cross-route tests cover `StateValue` and `Decision` through explicit strict parsing plus serde_json string, slice, and reader entry points, and `Question` where it has a raw JSON route. They accept top-level and nested integer `-0` as normalized `0`, i64/u64 boundaries, and reject `-0.0`, `-0e0`, `1.0`, `1e0`, signed/unsigned overflow, duplicate keys, and 5,000 nested containers. Explicit boundary tests now establish that standalone `StateValue` accepts 127 and 128 nested arrays and rejects 129 on every raw route. A Decision/Question envelope itself consumes one container, so 127 nested child arrays reaches the same whole-document limit and 128 is rejected. Tests also pin owned-`Value` behavior: ordinary integer/order conversion remains accepted, while a `Value` that already represents parsed `-0` as `-0.0` is rejected rather than guessed back into an integer.

Final follow-up checks all exited 0:

- `cargo fmt --all`
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace` — 47 tests passed (4 CLI, 31 core unit, 11 core integration, 1 backend skeleton), 0 failed; doc-test harnesses contained 0 tests.
- `cargo clippy --all-targets -- -D warnings` — default-member scope passed.
- `cargo test` — 46 tests passed, 0 failed; the non-default backend crate's one unit test is intentionally absent from this default scope.
- `git diff --check`; `git diff --exit-code -- reference` confirmed no reference changes.

The M1 gate remains pending independent parent/Astra re-review. No commit was created.

### Raw-value collision remediation

The final targeted review reproduced a `serde_json` `raw_value` feature collision in the derived `RawDecision`/`RawQuestion` conversion path: arbitrary state or metadata objects containing serde_json's legal private-protocol-looking keys could be reinterpreted, and JSON-looking strings could be rejected. Decision and Question now destructure the already strict-parsed insertion-ordered object directly. Small typed extractors remove required fields with `shift_remove`, report field/index paths, move the untouched state into `StateValue::try_from`, parse option objects and their two required strings while ignoring extra fields like Python, and retain all remaining Decision metadata in original relative order. No arbitrary state, option-extra, or metadata subtree goes through another Serde conversion, and no legal JSON key is blacklisted.

Regression coverage exercises `StateValue` and `Decision` through explicit parsing plus serde_json string, slice, and reader routes with nested objects/arrays, states equal to the strings `"[1,2]"` and `"hello"`, option extras, and retained metadata. Both `$serde_json::private::RawValue` and `$serde_json::private::Number` appear as legal state and metadata keys. The resulting prompt payload is byte-compared with Python 3 stdlib `json.dumps(..., ensure_ascii=False, allow_nan=False)`. All prior negative-zero, float, duplicate-key, overflow, and depth tests remain present.

The depth contract is now explicit in README, PLAN, and validation-helper documentation: bounded non-overflow handling applies to raw JSON routes and `StateValue::try_from(Value)`. For an untrusted arbitrary-depth already-built tree, callers must use `TryFrom<Value>`. Generic upstream operations such as `serde_json::from_value::<StateValue>` with `RawValue` and `Value::to_string()` may recursively serialize before this crate can enforce its bound and are not covered.

Final gate commands all exited 0:

- `cargo fmt --all`
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace` — 54 tests passed (4 CLI, 37 core unit, 12 core integration, 1 backend skeleton), 0 failed; doc-test harnesses contained 0 tests.
- `cargo clippy --all-targets -- -D warnings` — default-member scope passed.
- `cargo test` — 53 tests passed, 0 failed; the non-default backend crate's one unit test is intentionally absent from this default scope.
- `git diff --check`; `git diff --exit-code -- reference` confirmed no reference changes.

Final parent/Astra adjudication: PASS. Inspected direct field extraction, preserved-order `shift_remove`, untouched state/metadata moves, restricted prompt serializer and Python differential reserved-key regression. This resolves the last separate-Astra blocker without blacklisting valid keys. Approved the documented boundary: raw JSON and `StateValue::try_from` are depth-bounded; arbitrary upstream `serde_json::from_value` reserialization is not. Independently reran `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` (54 passed), `git diff --check`, and unchanged-reference check; all passed. M1 is approved for its milestone commit; proceed to M2 with runtime claims still unverified.

## M2 — pinned GGUF cache and direct smoke (initial implementation evidence)

Implemented the bundled three-model manifest, registry lookup/path/pull surfaces, canonical cache-root precedence (`--cache-dir` API argument, `OPENJEV_HOME`, then `~/.cache/openjev`), hf-hub 1.0 blocking pinned-revision downloads, process-safe per-artifact locks, atomic verified receipts, complete size/SHA-256 verification on both cache resolution and every owner-worker load, explicit offline miss and corruption errors, and explicit-only repair quarantine/re-download. Before transfer, the exact pinned snapshot paths under `~/.cache/huggingface/hub` were inspected; none existed. Observed execution history showed one transfer of each artifact into `~/.cache/openjev/hub` (5,213,792,864 manifest bytes combined), followed by offline CPU reuse; retained offline captures prove reuse rather than independent transport counts. No GGUF entered the repository or test temporary directories.

The native direct-smoke path uses one owner thread. Backend/model ownership and every borrowing context remain scoped inside that thread; no self-reference, unsafe workspace code, or Send workaround was added. The adapter uses `AddBos::Never`, `token_to_piece_bytes` with exact ASCII checks, the core's all-slot roundtrip/collision/vocabulary/append-boundary validation, chunk-local final `get_logits_ith`, immediate owned full-logit copying, and the core f64 readout. Each smoke performs one warmup context and one measured context. llama.cpp logs use the safe `send_logs_to_tracing` callback and remain on stderr; the example emits JSONL only on stdout and continues after a model failure/diagnostic.

Runtime evidence on Mac15,6 / Apple M3 Pro / 18 GB / macOS 26.2:

- Metal build: separate `target-m2-metal`, `GGML_METAL=ON`, Cargo `metal`, OpenMP off, all layers requested. CMake cache confirmed Metal ON; native logs showed Apple M3 Pro MTL and actual offload qwen3 29/29, MiniCPM5 43/43, Qwen3.5 34/34.
- True CPU build: separate `target-m2-cpu`, `GGML_METAL=OFF`, Cargo `native`, OpenMP off, zero GPU layers, KQV and op offload disabled. CMake cache confirmed Metal OFF; safe runtime device enumeration contained CPU only.
- All three exact artifacts loaded one at a time and produced finite Metal and CPU full-vocabulary/slot readouts at actual context 4096. Architectures/trained contexts were qwen3/40,960, llama/131,072, qwen35/262,144. Slot IDs for A/B/C were 32/33/34, 54/55/56, 32/33/34.
- MiniCPM5 and Qwen3.5 GGUF template metadata hashes exactly matched the pinned native expectations. Qwen3-0.6B GGUF hash `57f1fd00...d0361` differs from pinned native `a55ee1b...74d8`; its structured outcome is `needs-template-adjudication`, not passed. The exact Qwen model still loaded/scored on both devices so remaining model attempts did not stall. No semantic-equivalence claim or silent template relaxation was made.
- The safe wrapper does not expose actual offloaded-layer count, so structured output records `gpu_layers_actual: null`; selected native stderr is the actual Metal layer evidence.

Detailed rows, reduced stderr, CMake/runtime configuration and timings are in `docs/RESULTS.md` and `docs/results/`. Final reproducibility captures used `OPENJEV_INTEGRATION=1 --offline`; all six artifact rows were cache hits. Invoking the harness without `OPENJEV_INTEGRATION=1` exited 2 immediately, emitted one JSON error on stdout and zero stderr bytes, and did not resolve or load a model. M3's 144-row exact Qwen hash/token gate, exhaustive parity, shared/batch algorithms, and public production CLI execution remain unimplemented and unclaimed.

Final M2 local checks all exited 0 after the evidence refresh:

- `cargo fmt --all` and `cargo fmt --all -- --check`.
- `cargo check --workspace`.
- `cargo clippy --workspace --all-targets -- -D warnings` and default-member `cargo clippy --all-targets -- -D warnings`.
- `cargo test --workspace` — 59 tests passed (4 CLI, 49 core unit/integration, 6 backend registry/cache), 0 failed; doc-test harnesses contained 0 tests.
- Default-member `cargo test` — 53 tests passed, 0 failed.
- Metal `GGML_METAL=ON CARGO_TARGET_DIR=target-m2-metal cargo clippy -p openjev-llama --features metal --all-targets -- -D warnings` and matching feature test — 5 tests passed.
- True CPU `GGML_METAL=OFF CARGO_TARGET_DIR=target-m2-cpu cargo clippy -p openjev-llama --features native --all-targets -- -D warnings` and matching feature test — 5 tests passed.

The post-check CMake caches still reported Release, BLAS/OpenMP/native-tuning OFF in both targets, Metal ON only in `target-m2-metal`, and Metal OFF in `target-m2-cpu`. No workspace Rust source contains `unsafe`.

### Targeted cache remediation and Qwen adjudication

Parent/Astra blocked the initial M2 review on concrete cache safety/repair defects and separately approved a narrow Qwen template equivalence. The cache now validates portable `OWNER/NAME` and safe nested relative filename components before locks or other filesystem work. Absolute, traversal/dot, empty, backslash, Windows drive, and network-like forms fail as manifest errors; the reproduced absolute outside-file case leaves both that file and the cache root untouched. Controlled directories and resolved snapshot targets are checked against canonical ownership boundaries.

HF imports now canonicalize the source to a regular file before hard-linking, so a normal `snapshots/.../file -> ../../blobs/hash` source imports bytes rather than the symlink inode. `symlink_metadata` detects dangling destinations. Without `repair`, corruption still returns an integrity/safety error without mutation or fetch. With `repair`, corrupt owned blob bytes are retained in quarantine and owned snapshots are rebuilt; a corrupt external default HF source is bypassed but never modified. Offline repair reuses a valid alternate or returns `OfflineMiss`. A stale receipt is removed before repair and is not republished until the final owned snapshot passes size/SHA verification. Unix publication renames the synced temporary receipt directly over the destination without an unlink gap.

Tiny-file regressions cover malformed model fields and unchanged outside files, relative HF snapshot symlinks, dangling blobs, corrupt external + successful fetch, corrupt owned blob/snapshot + valid external reuse, failed repair with no receipt, healthy repair-enabled no-fetch reuse, and true two-process contention. The process test launches two copies of the test binary against one lock/cache/marker and observes one fetch followed by a verified reusable path. Ordinary tests contain no weights.

Qwen's 4,100-byte GGUF template (`57f1fd00…d0361`) and 4,168-byte native template (`a55ee1b1…74d8`) remain explicitly nonidentical. The manifest now has an equivalence record keyed to those hashes and artifact `9465e63a…031`. Runtime status is `reviewed-equivalent` only for exactly two string system/user messages, no tools, `add_generation_prompt=true`, and `enable_thinking=false`; identical hashes report `exact`, and unseen triples report `mismatch`. Credited fixtures live outside `reference/`. The reproducible Jinja 3.1.4 oracle passed 144 authored + 108 perturbation rows, all 252 reference prompt hashes, and four recorded edge states (`<tool_response>…`, `<think>…`, Unicode/quotes, insertion-ordered structured state). No broader tools/multiturn/multimodal/reasoning claim is made.

New create-only `--offline` captures in `docs/results/*-final.*` passed all three models on Metal and true CPU with `cache_hit=true`; the original mismatch evidence remains unchanged. Qwen reports `reviewed-equivalent`; MiniCPM5 and Qwen3.5 report `exact`. All six readouts are finite. `gpu_layers_actual:null` remains honest because the safe wrapper has no count API; selected Metal stderr still records 29/29, 43/43, and 34/34 layer offload.

Final remediation gate commands, all exit 0:

- `python3 scripts/verify_qwen_template_equivalence.py` — 144 authored, 108 perturbation, 252 reference hashes, and 4 edge states.
- `cargo fmt --all` and `cargo fmt --all -- --check`.
- `cargo clippy --workspace --all-targets -- -D warnings`.
- `cargo check --workspace`.
- `cargo test --workspace` — 72 tests passed (4 CLI, 49 core unit/integration, 19 backend registry/cache), 0 failed; doc-test harnesses contained 0 tests.
- Default-member `cargo clippy --all-targets -- -D warnings` and `cargo test` — 53 tests passed (4 CLI, 49 core), 0 failed.
- Metal `GGML_METAL=ON CARGO_TARGET_DIR=target-m2-metal cargo clippy -p openjev-llama --features metal --all-targets -- -D warnings` and matching feature test — 19 tests passed.
- True CPU `GGML_METAL=OFF CARGO_TARGET_DIR=target-m2-cpu cargo clippy -p openjev-llama --features native --all-targets -- -D warnings` and matching feature test — 19 tests passed.
- Offline Metal and CPU `m2_smoke --all` create-only captures — 3/3 passed on each device.

M2 remediation is ready for independent parent/Astra confirmation. No commit was created, no reference file changed, and no M3/shared/batch implementation was added.

### Final mutation-containment remediation

Astra's final targeted re-review reproduced two mutation escapes that verification containment alone did not prevent: repair could rename a snapshot through `cache/hub -> outside`, and offline receipt invalidation could unlink through `cache/openjev/receipts -> outside`. Cache mutations now share a parent-containment check that requires a lexically normal owned path, walks existing parent components from the selected cache root, anchors each resolved directory under the canonical cache root, and additionally scopes hub mutations to the canonical hub boundary. A missing parent is a safe no-op only for removal; creation/link/rename paths require an existing validated parent. The helper checks parents rather than leaf targets, so an owned leaf symlink can still be unlinked safely without following its target.

The check now precedes lock-file creation, import blob/snapshot hard links and replacement, quarantine source and destination renames/removals, receipt invalidation, temporary creation/replacement, and receipt publication. Escaping parents return `CacheSafety`; repair does not reinterpret such paths as corrupt owned artifacts. Directory creation retains its existing component-by-component containment checks. Same-user filesystem races remain explicitly out of scope.

Four Unix regressions provide the focused proof:

- `repair_rejects_hub_parent_symlink_without_moving_outside_snapshot`: offline repair returns `CacheSafety`, performs zero fetches, and leaves the outside snapshot sentinel byte-for-byte unchanged.
- `offline_miss_rejects_receipts_parent_symlink_without_deleting_outside_receipt`: offline resolution returns `CacheSafety`, performs zero fetches, and leaves the outside `<sha256>.json` sentinel byte-for-byte unchanged.
- `repair_rejects_nested_blob_parent_symlink_without_moving_outside_blob`: the same guarantee holds for a nested owned `blobs` parent symlink escape.
- `repair_of_owned_leaf_snapshot_symlink_retains_normal_behavior`: a normal owned relative snapshot leaf symlink over corrupt owned bytes is removed without following it, the corrupt blob is quarantined, and a valid offline external artifact repairs both blob and snapshot.

Final mutation-remediation commands all exited 0:

- `cargo fmt --all` and `cargo fmt --all -- --check`.
- `cargo check --workspace`.
- `cargo clippy --workspace --all-targets -- -D warnings`.
- `cargo test --workspace` — 76 tests passed (4 CLI, 49 core unit/integration, 23 backend registry/cache), 0 failed; doc-test harnesses contained 0 tests.
- Default-member `cargo clippy --all-targets -- -D warnings` and `cargo test` — 53 tests passed (4 CLI, 49 core), 0 failed.
- Metal `GGML_METAL=ON CARGO_TARGET_DIR=target-m2-metal cargo clippy -p openjev-llama --features metal --all-targets -- -D warnings` and matching feature test — 23 tests passed.
- True CPU `GGML_METAL=OFF CARGO_TARGET_DIR=target-m2-cpu cargo clippy -p openjev-llama --features native --all-targets -- -D warnings` and matching feature test — 23 tests passed.
- `git diff --check`; `git diff --exit-code -- reference` confirmed no reference changes.

Per the focused review scope, the lengthy real-model CPU smokes were not rerun because engine/template code did not change; the existing create-only evidence above remains the M2 runtime record. No commit was created, no new target directory or download was used, and no M3/reference/workflow change was made. Final parent/Astra inspection: PASS. Verified canonical-parent traversal is checked before quarantine rename/unlink and receipt invalidation, separately from safe leaf-symlink removal, and that ownership boundaries are anchored to the canonical trusted root. Separate-Astra findings are resolved by targeted regressions, including unchanged outside sentinels and zero fetches. Independently reran fmt, workspace clippy with warnings denied, all 76 workspace tests, diff check and unchanged-reference check. M2 approved for commit; all six retained native smoke captures pass. The statement that M3 was unrun is superseded by the section below; it remains accurate for the M2 commit itself.

## M3 — production direct scoring and strict Qwen parity (reviewed and approved)

Implemented the production owner-thread `EngineHandle::score_direct` request returning the complete validated core `Readout`. Every decision renders the restricted approved profile, tokenizes with no BOS, verifies all 2–16 slots and append boundaries, creates a clean context, performs exactly one prefill (chunked at `n_batch` when needed), retrieves the last chunk's local final index, copies native f32 logits before context destruction, and computes the f64 readout. There is no generation, truncation, implicit warmup, fallback, or borrowed logits escape. Because each call uses a fresh context/full prefill, direct inference `cache_hit` is now always `false`; the previous production code incorrectly copied artifact download-cache status into that field. Artifact cache status remains separate cache/runner metadata. M2 `smoke_direct` still performs its deliberate warmup plus measured pass.

Added `EncodedPrompt`/`EngineHandle::encode_direct` and strict reference validation for integration gates, plus an explicit last-chunk-local index validator used by production. Negative tests use an actually spacing-mutated prompt, an inserted BOS token, and an absolute chunk index; all fail without altering the renderer or decode path. Per-decision validation/token checks occur before decode. Model cache bytes are verified at resolution and once again at worker load, never per row. Worker shutdown still joins after contexts/model/backend drop in owner-thread scope.

Production metadata now reports exact artifact/native revisions, quantized/mixed dtype, backend/native commit, input/prompt/forward/total values, actual context/batch/thread/device configuration, and template status/evidence. The reviewed Qwen equivalence remains keyed to the exact artifact/GGUF/native template triple; missing or unseen template identities can still load for M2 diagnostics but production encoding/scoring refuses them before decode. The safe wrapper's actual layer-count gap is resolved honestly in the normative schema: `ExecutionMetadata.gpu_layers_actual` is `Option<u32>` and serializes explicit null, with required `gpu_layers_status`. CPU with zero/offload disabled reports `Some(0)/known-disabled`; Metal/CUDA safe-wrapper unknown reports `None/unavailable`, never the requested value. Core validation and JSON Schema tests cover legitimate unknown and reject contradictory status/count metadata.

Added opt-in `m3_parity` and `m3_exact_gate` surfaces under `integration`; ordinary tests neither load nor download models. The create-only offline runner prevalidates all inputs/output paths, indexes 252 unique reference rows by ID, hard-gates authored144 and separately records perturbations108, writes full production Readout JSONL, and emits one JSON stdout summary while native/build logs go only to stderr.

Actual Metal evidence using only `~/.cache/openjev` and `target-m2-metal`:

- Integration encoded gate: 144/144 authored exact prompt hashes, input token counts, ordered option IDs, answer token IDs, and generic slot/boundary verification.
- Production scoring gate: authored 144/144 exact hashes/tokens/IDs/slots and finite readouts; extended perturbations 108/108 exact on the same fields.
- Authored native-BF16 versus local Q8_0: first argmax 140/144 (0.972222); logit MAE/RMSE/max 0.534313/0.664228/2.543209; probability MAE/RMSE/max 0.023046/0.068618/0.513266; four mismatch IDs and both margins retained.
- Extended perturbations: 107/108 argmax agreement (0.990741); logit MAE/RMSE/max 0.565334/0.721156/2.591896; probability MAE/RMSE/max 0.014599/0.056621/0.542852; one mismatch retained.
- Astra accepted 140/144 with logit MAE 0.534313, and the extended 107/108 result, as the measured pinned-backend/Q8_0 versus native-BF16 baseline. This is not a numerical-equivalence claim and does not attribute the delta solely to quantization. No 98% acceptance gate or other guessed tolerance applies.

Retained create-only files and hashes are documented in `docs/RESULTS.md`. The full raw native stderr was reduced after recording its SHA-256/line/byte count, matching the M2 evidence policy. No reference file changed, model downloaded, new native target directory created, or M4/M5/M6/M7 behavior implemented.

Original full M3 pre-review checks, all exit 0:

- `cargo fmt --all` and `cargo fmt --all -- --check`.
- `cargo check --workspace`.
- `cargo clippy --workspace --all-targets -- -D warnings`.
- `cargo test --workspace` — 77 tests passed (4 CLI, 50 core unit/integration, 23 backend registry/cache), 0 failed; the feature-gated M3 integration test compiled as zero default tests.
- Default-member `cargo clippy --all-targets -- -D warnings` and `cargo test` — 54 tests passed (4 CLI, 50 core), 0 failed.
- Metal `GGML_METAL=ON CARGO_TARGET_DIR=target-m2-metal cargo clippy -p openjev-llama --features metal,integration --all-targets -- -D warnings` and matching feature test — 25 backend tests passed; integration test was environment-gated in this ordinary feature run.
- True CPU `GGML_METAL=OFF CARGO_TARGET_DIR=target-m2-cpu cargo clippy -p openjev-llama --features native,integration --all-targets -- -D warnings` and matching feature test — 25 backend tests passed; integration test was environment-gated.
- Explicit Metal encoded integration gate with `OPENJEV_INTEGRATION=1` — 144 authored rows passed in 42.98 seconds.
- Full create-only Metal production report runner with `OPENJEV_INTEGRATION=1` — authored144 and perturbations108 both passed exact gates; quantitative results retained.
- Selected offline Qwen M2 smoke after M3 changes — passed and retained distinct nonnegative warmup/measured times (0.049901/0.045627 seconds), confirming the two-pass diagnostic behavior remains; temporary smoke output was not added as new milestone evidence.

Targeted `cache_hit` remediation checks also all exited 0 with `CARGO_NET_OFFLINE=true`:

- `cargo fmt --all`, `cargo fmt --all -- --check`, and `cargo check --workspace`.
- `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace` — 78 tests passed (4 CLI, 50 core, 24 default backend/cache/registry), including the model-free cached-artifact/repeated-direct metadata regression.
- Default-member `cargo clippy --all-targets -- -D warnings` and `cargo test` — 54 tests passed (4 CLI, 50 core).
- Existing Metal target: `GGML_METAL=ON CARGO_TARGET_DIR=target-m2-metal cargo clippy -p openjev-llama --features metal,integration --all-targets -- -D warnings` and matching feature test — 26 unit tests passed; the integration surface returned immediately because `OPENJEV_INTEGRATION` was unset.
- Existing true-CPU target: `GGML_METAL=OFF CARGO_TARGET_DIR=target-m2-cpu cargo clippy -p openjev-llama --features native,integration --all-targets -- -D warnings` and matching feature test — 26 unit tests passed; the integration surface returned immediately because `OPENJEV_INTEGRATION` was unset.
- `git diff --check` and `git diff --exit-code -- reference`.

No model was resolved, loaded, or downloaded during these checks. The lengthy numerical benchmark was deliberately not rerun; retained artifact hashes/sizes still match the documented originals.

Final parent/Astra adjudication: PASS. Confirmed production construction uses only `direct_inference_cache_hit()` returning `Some(false)` and no artifact-cache state; the report runner rejects contrary direct metadata. The model-free regression exercises the production helper, not native inference; no extra native rerun is claimed. Independently reran fmt, workspace clippy with warnings denied, all 78 workspace tests, diff/reference checks. Separate-Astra numerical acceptance and the independently recomputed 144/144 + 108/108 exact fields remain unchanged. M3 approved for milestone commit. The 252 create-only rows remain unchanged with the historical `cache_hit=true` defect annotated in `docs/RESULTS.md`; their raw logits and accepted numerical baseline are unaffected.

## M4 — production CLI, model surfaces, and explicit serial fallback

Implemented the production CLI over the M3 owner-thread direct scorer. `decide`, `noul`, `score`, `ask`, and `run` now construct complete validated core decisions before resolving any artifact. Text state from flags/files/stdin is preserved byte-for-byte without trimming or JSON guessing; structured state uses only explicit JSON flags. Explicit state returns without reading stdin, non-TTY stdin is detected in `main`, and a TTY without state fails validation. Ask consumes one strict Decision JSON document. Run consumes nonblank JSONL rows, rejects duplicate IDs and all parse/validation errors before load, preserves row order, emits per-row runtime ErrorRecords and continues safe independent rows, and exits 1 if any row failed.

Repeated `decide --question` automatically requests shared execution. Explicit shared/batch run modes and repeated questions use serial full-prompt direct scoring in M4, with exact requested/effective mode metadata, nonempty M5 reason, `cache_hit=false`, serial serving config, and an stderr warning even under `--quiet`. `--require-shared` returns a pre-load structured error. No cache copy, independent sequence packing, probe receipt or fake batch timing was added. Explicit serial mode uses full prompts and no prefix metadata.

Noul and Score remain thin adapters over the returned production Choice readout. Noul emits ordered `yes/no` and `p_yes`; Score requires finite aligned values (or defaults to 0..K-1) and emits expectation/argmax/distribution. Confidence is opt-in normalized margin with its exact uncalibrated status. Nonidentity permutation, nonzero seed, nondefault temperature and calibration are rejected as M7-unsupported rather than ignored. Eval/bench/calibrate and models probe return explicit milestone-specific `not_implemented` records.

Added custom model resolution without weakening the registered identity path. Local files are canonicalized and hashed in place, never moved; optional caller hash yields caller integrity and omission yields `local-unverified`. Custom Hub specs require safe `OWNER/REPO`, a 40-lowercase-hex commit, safe relative filename, caller SHA-256 and explicit profile. Custom runtime specs have private construction, cannot claim manifest/native equivalence metadata, and score with `template_override=true`, `override-unverified`, and no native reference. Registered artifacts retain exact manifest fingerprints and template adjudication.

Model list hashes present canonical entries and reports verified/missing/failure truthfully, plus explicit unprobed M5 statuses. Pull/path use verified canonical cache behavior and return JSON envelopes. Output preflight rejects existing/same-file/unsafe-parent targets; final creation uses `create_new`. `--output` leaves JSONL in the file and one `openjev-write-summary-v1` object on stdout. `--pretty` is restricted to single objects. Help/version are intercepted as JSON and command help includes piped examples. Broken stdout returns nonzero without panic text.

A native startup deadlock was found by the first real CLI probe: `main` held a `StderrLock` while the owner thread's llama tracing callback attempted to write native logs. Main now keeps unlocked stderr/stdout handles (which lock per write), so owner startup, scoring and shutdown complete while logs remain exclusively on stderr. Owner-thread join has clean and panic regressions; no unsafe or Send/Sync workaround was introduced.

Testing/evidence:

- Ordinary deterministic tests inject a scorer only through the library test seam; production never emits mock readouts. They cover Noul/Score/confidence/mode adaptation without models.
- Process tests spawn the compiled `openjev`, capture streams/exits, exercise all help surfaces, stdin precedence/no-blocking, parse-before-backend, backend-disabled behavior, JSON model list, and broken pipe.
- Opt-in release process tests (`OPENJEV_INTEGRATION=1`) reused cached Qwen on Metal for decide/noul/score/ask/run, runtime-row continuation, output no-overwrite, repeated-question fallback, and custom local override. Three tests passed in the retained final capture.
- Create-only reports under `docs/results/m4-*` retain per-case stdout/stderr hashes, JSON checks, model-cache envelopes and concise native evidence. No transfer or new target directory occurred.
- Metal and true-CPU native all-target clippy/tests reused `target-m2-metal` and `target-m2-cpu`. Environment-gated M3/native tests returned immediately in ordinary feature runs; only the separately recorded release Qwen/Metal run performed inference. No six-model CPU/Metal smoke was repeated.

Current local gate commands (all exit 0; final rerun recorded at handoff):

- `cargo fmt --all -- --check`.
- `CARGO_NET_OFFLINE=true cargo clippy --workspace --all-targets -- -D warnings`.
- `CARGO_NET_OFFLINE=true cargo test --workspace` — 87 tests passed, 0 failed; integration/native surfaces were zero tests without features.
- Default-member `CARGO_NET_OFFLINE=true cargo clippy --all-targets -- -D warnings` and `cargo test` — 63 tests passed.
- Metal: `GGML_METAL=ON CARGO_TARGET_DIR=target-m2-metal cargo clippy/test -p openjev-llama -p openjev-cli --features openjev-cli/metal,openjev-cli/integration --all-targets` — 39 test functions passed; environment-gated integration functions did no inference with `OPENJEV_INTEGRATION` unset.
- True CPU: the equivalent commands with `GGML_METAL=OFF`, `target-m2-cpu`, and `openjev-cli/native,openjev-cli/integration` — 39 test functions passed; no model load.
- Actual opt-in Metal CLI: `OPENJEV_INTEGRATION=1 ... cargo test --release -p openjev-cli --features metal,integration --test native_cli -- --nocapture` — 3/3 process tests passed using the cached Qwen artifact; the post-help-validation rerun is retained as `m4-native-cli-tests-final2.txt`.
- `git diff --check` and `git diff --exit-code -- reference`.

M4 is ready for parent/Astra review. No commit was created.

### Astra M4 targeted gate remediation

Astra's targeted review blocked M4 on two concrete implementation defects and one nonblocking help-metadata defect. All three are now fixed locally; parent/Astra targeted confirmation remains pending.

1. Custom Hub resolution no longer gives hf-hub an unchecked `cache/hub`. Registered and caller-hashed Hub paths now share a pre-download containment preflight that creates and validates the canonical Hub root plus hf-hub's `.locks/models--…`, repository, `blobs`, pinned `snapshots/<commit>/<nested file parent>`, and `.no_exist` parents before downloader invocation. Existing or nested parent symlink escapes fail with `CacheSafety` before the injected downloader runs. Postfetch resolution requires the exact pinned snapshot path, a canonical regular file inside the owned Hub, and the caller SHA-256; custom artifacts safely derive their byte count from the hashed file. Offline custom resolution rejects a hash-correct snapshot symlink to an external target without fetching or modifying the target. Tiny zero-network regressions cover online root/lock/repository/blob/snapshot-parent escapes with zero downloader calls, offline external-target refusal, and normal mock download plus offline reuse. No local/external file is quarantined or repaired, and same-user concurrent malicious filesystem races remain out of scope.
2. `run` no longer buffers every result in a `Vec`. Complete JSONL parsing/validation and output alias preflight remain before model startup; a selected `--output` file is then atomically reserved with `create_new` before scorer loading. Each success or ErrorRecord is serialized and flushed before the next score starts. stdout follows the same per-row flush policy. A sink error stops later scoring immediately and still calls owner shutdown; a file summary is emitted only after all rows, clean shutdown, and file sync complete. The documented failure policy is intentional: startup failure leaves the newly reserved file empty, while a later sink failure leaves only its flushed prefix and no summary. Model-free injected-scorer/sink tests prove row-one visibility before call two and prove call three is skipped after a call-two broken pipe. The cached-Qwen release process test additionally observed the first of 20 file rows while the process was still scoring.
3. Help metadata now matches clap's rendered help against the built command tree, instead of scanning raw tokens. The `command` and `usage` fields therefore identify `openjev models pull` for nested help, even when global `--model` values are `run`/`models` before or after the subcommand. Regressions cover long/short help and clap's `help models pull` alias; every help response remains one stdout JSON object with empty stderr.

A new create-only native capture, `docs/results/m4-native-cli-tests-final3.txt` (760 bytes, SHA-256 `1e55eb8c870cc792ca2add8e9b81b9a096726040f2e556518bfd7209b83b4f73`), supersedes but does not overwrite the earlier M4 native-test captures. It records 4/4 release Metal process tests passing in 9.39 seconds against the cached Qwen artifact, including incremental file visibility. No network transfer, real-weight copy, new target directory, M5 behavior, or reference change occurred.

Final targeted-remediation commands all exited 0:

- `cargo fmt --all` and `cargo fmt --all -- --check`.
- `CARGO_NET_OFFLINE=true cargo clippy --workspace --all-targets -- -D warnings`.
- `CARGO_NET_OFFLINE=true cargo test --workspace` — 97 tests passed (17 CLI, 50 core, 30 backend/cache/registry), 0 failed; doc-test harnesses contained 0 tests.
- Default-member `CARGO_NET_OFFLINE=true cargo clippy --all-targets -- -D warnings` and `cargo test` — 67 tests passed (17 CLI, 50 core), 0 failed.
- Metal in existing `target-m2-metal`: clippy/test for `openjev-llama` + `openjev-cli` with `openjev-cli/metal,openjev-cli/integration` — 48 test functions passed; environment-gated integration functions did no inference with `OPENJEV_INTEGRATION` unset.
- True CPU in existing `target-m2-cpu`: equivalent clippy/test with `GGML_METAL=OFF` and `openjev-cli/native,openjev-cli/integration` — 48 test functions passed; environment-gated integration functions did no inference.
- Actual cached Metal release process gate with `OPENJEV_INTEGRATION=1` — 4/4 native CLI tests passed; the new test observed incremental file output before process completion.
- `git diff --check`; `git diff --exit-code -- reference` confirmed no reference changes.

Final parent/Astra targeted adjudication: PASS. Inspected shared custom/registered pre-download containment, exact postfetch snapshot ownership, immediate write/flush per row before next scoring, create-new sink reservation and shutdown-on-output-error. The targeted fixes resolve separate-Astra M4 blockers; help metadata regression coverage is included. Independently ran fmt, workspace clippy with warnings denied, all 97 workspace tests, diff/reference checks. Earlier parent release probes verified JSON-only help/decide/noul and exit-2 validation. Native 4/4 streaming capture is retained with hash. M4 approved for milestone commit; genuine shared/batch work remains M5.

## M5 — receipt-gated shared KV and independent packed batching

Implemented exact Python-compatible shared-prefix construction: placeholder choice, the state payload exactly once, ordered evidence JSON with only its closing brace removed, no BOS, then exactly one dropped token. Every full rendered prompt must start with the nonempty token prefix and retain a nonempty suffix. Shared execution prefills immutable sequence 0 once, performs only full `copy_kv_cache_seq(0, branch, None, None)` into clean branch IDs, advances suffixes in ragged chunks, reads only chunk-local final-token logits, checks full sequence-clear success, and plans waves against `prefix + sum(active suffix lengths)` under one unified-KV context. Explicit `n_ctx` is never enlarged silently; auto context is bounded by `max_context_tokens`. Independent batch mode packs complete unrelated prompts into separate sequences, preserves input order, and never reports prefix reuse.

Both modes create a fresh context for each request. Shared readouts use selected suffix-position slot logits and common group timing; serial/batch readouts use full-vocabulary logits. Metadata distinguishes requested/effective mode, exact probe ID, cache truth, common shared timing and serial fallback reason. No per-row fake totals or artifact-cache-as-inference-cache claims were added.

Native eligibility is fail closed. Receipt identity covers artifact SHA-256, exact wrapper/native pin, probe-suite version, requested device/layers and KQV/op-offload choices, threads, explicit/automatic context limits, batch/ubatch/sequence capacity, unified KV and prompt profile. Receipt files are consistent-hash validated, cache-contained and atomically replaced; malformed, forged, failed or nonmatching receipts are rejected. The hardcoded gates are slot-logit max absolute delta `<= 1e-3`, probability max delta `<= 1e-4`, and identical first argmax; receipt JSON cannot relax them. Standard scoring neither probes nor overwrites receipts. A matching passing receipt allows the real mode; absence/failure/runtime error discards the entire native group and reruns fresh serial full prompts. `--require-shared` checks eligibility before inference and errors instead of falling back.

`models probe ID --mode shared|batch` launches a subprocess before model/backend/context construction. Parent success requires a normal child exit plus a complete passing receipt. Crash/assertion/nonzero/malformed output cannot be reinterpreted as success; a process regression forces an abort and verifies the parent returns a disabled JSON report. The bounded deterministic suite contains binary 1-branch, ragged 3-way 2-branch multichunk, 16-way long-state 21-branch, changed-state isolation and a `n_seq_max + 1` repeated copy/clear case that forces branch-ID reuse across waves. Eligibility validation may stop after a decisive failure but records every remaining case as `unrun-after-decisive-failure`, never passed.

Actual create-only offline probes reused `~/.cache/openjev`, `target-m2-metal` and `target-m2-cpu` for all three pinned artifacts, both devices and both modes. All twelve configurations failed the frozen gates and therefore remain serial-only. Metal usually passed the one-branch case and failed the ragged case; true-CPU shared failed the first case for all profiles, while true-CPU batch passed the first and failed the ragged case. Every observed first argmax remained equal, but slot-logit and/or probability limits failed. No tolerance was changed and no profile was enabled. The reports and exact commands are in `docs/RESULTS.md` and `docs/results/m5/`.

Final handoff review found and fixed one fail-closed receipt defect before the gates: a self-consistent receipt with only one passing case could previously satisfy the summary because validation did not pin the complete suite shape. Receipt validation now requires all five versioned case IDs in order, exact passing-row cardinalities (including `n_seq_max + 1` cycles), a decisive failure before any `unrun-after-decisive-failure` entry, complete-or-absent measurements, and a nonempty failed summary reason. Reordered/incomplete receipt regressions pass. The twelve finalized failed receipts already have this complete shape; the earlier same-outcome pre-final captures omitted `probe_suite_version` from configuration identity and remain preserved but ineligible.

Scheduler coverage was also made explicit without changing native behavior: unit tests now exercise empty/completed, uneven ragged progress, cursor/capacity bounds, chunk-local indices, prefix-plus-suffix wave limits, empty wave rejection, and mandatory successful full sequence clear before branch-ID reuse. The native probes remain the evidence for actual copy/decode/clear on reached binary/ragged cases; the unrun 21-way/isolation/cycle cases are not relabelled passed.

Astra accepted the numerical outcome as intended fail-closed behavior: all twelve exact configurations remain disabled and serial full-prompt fallback is the supported user path. The measured deltas are not an M5 blocker, and no tolerance relaxation or additional full probe matrix is required.

A subsequent focused lifecycle review reproduced one concrete authorization defect: starting a reprobe did not invalidate a preexisting passing exact-key receipt, so a child abort could report `enabled=false` while ordinary scoring still loaded the old pass. The parent now resolves and verifies the exact artifact/configuration/mode key without backend/model/context initialization, acquires the exact-key process lock, and atomically publishes a suspension marker before launching the child. The lock is held through child completion and publication, serializing ordinary same-key reprobes. Receipt loading checks suspension both before and after validation, so a concurrent reprobe cannot complete a stale load after revocation. Crash, launch failure, malformed JSON, nonzero-after-passing, identity mismatch, or publication failure leaves suspension in place; an old pass is never restored. A normal, fully validated, exact parent-matching passing result is atomically written before suspension is cleared. Failed numerical receipts may remain as diagnostics but remain suspended/ineligible. Unrelated configuration receipts and historical reports are untouched.

Model-free lifecycle regressions seed passing records only in temporary caches and cover immediate suspension, dropped/crashed/malformed transitions, nonzero passing candidates, failed receipt publication, parent-owned identity rejection, successful replacement, and publication failure. The opt-in native process regression is stronger than the original report-only check: it labels a synthetic complete pass as state-machine test data in an isolated temporary cache, uses the existing cached Qwen file only for parent artifact identity, forces the child abort before model load, observes `enabled=false`/`receipt=null`/exit 1, then proves `load_passing_receipt` rejects the old pass. No user-cache pass is forged and no parity claim is derived from synthetic data.

Final commands, all exit 0:

- `cargo fmt --all`; `cargo fmt --all -- --check`; `CARGO_NET_OFFLINE=true cargo check --workspace`.
- `CARGO_NET_OFFLINE=true cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace` — 107 test functions passed (18 CLI, 51 core, 38 backend/cache/registry/probe), 0 failed; feature-gated integration surfaces were zero tests.
- Default-member `CARGO_NET_OFFLINE=true cargo clippy --all-targets -- -D warnings` and `cargo test` — 69 test functions passed (18 CLI, 51 core), 0 failed.
- Existing Metal target: `GGML_METAL=ON CARGO_TARGET_DIR=target-m2-metal cargo clippy/test -p openjev-llama -p openjev-cli --features openjev-cli/metal,openjev-cli/integration --all-targets` — 62 test functions passed (15 CLI, 46 backend, 1 M3 gate function), 0 failed. Environment-gated scoring/process bodies returned immediately because `OPENJEV_INTEGRATION` was unset.
- Existing true-CPU target: equivalent commands with `GGML_METAL=OFF`, `target-m2-cpu`, and `openjev-cli/native,openjev-cli/integration` — the same 62 test functions passed, 0 failed, with the same environment-gating caveat.
- Actual cached-Qwen release Metal CLI gate: `OPENJEV_INTEGRATION=1 ... cargo test --release -p openjev-cli --features metal,integration --test native_cli -- --nocapture` — 5/5 process tests passed. It covers JSON scoring/streaming, failed-receipt ordered serial metadata with `probe_id=null`, no shared timing and `cache_hit=false`, semantic fallback warning despite `--quiet`, `--require-shared` refusal, and durable revocation of an isolated synthetic preexisting pass after a forced probe-child abort.
- Mandatory M3 regression after engine changes: `OPENJEV_INTEGRATION=1 ... cargo test --release -p openjev-llama --features metal,integration --test m3_exact_gate -- --nocapture` — the cached Qwen release Metal gate passed all 144 authored rows with exact prompt hash, token count, option IDs and answer token IDs.
- `git diff --check`; `git diff --exit-code -- reference` passed.

The finalized matrix was not reprobed or overwritten after these review-only lifecycle/test refactors; no native scoring algorithm, tolerance or receipt configuration changed. No commit was created, no new native target directory or model download was used, and `reference/` remains unchanged. M6/M7 behavior was not added. Final parent/Astra targeted review: PASS. Confirmed exact-key lock and durable suspension precede child launch, loader checks suspension before and after receipt validation, and only a parent-matching successful passing result clears suspension. Failure/drop leaves old authorization disabled; isolated synthetic-pass tests cover this without manufacturing parity evidence. Independently reran fmt, workspace clippy with warnings denied, all 107 workspace tests, diff and unchanged-reference checks. M5 approved for commit as implemented but safely disabled on all twelve measured configurations—not as successful accelerated parity or a speedup. M6 must report shared speedup unavailable with these fallback reasons.

## M6 — evaluation and benchmark checkpoint before native measurements

Implementation checkpoint written before any long inference command. The working tree adds real `openjev eval` and `openjev bench` surfaces but **M6 is not yet marked passed**: all-model quality and CPU/Metal timing measurements remain pending at this checkpoint.

Completed before measurement:

- Embedded byte-identical, credited authored144, perturbations108 and Qwen3-0.6B 252-row browser-ladder assets in `openjev-core`; installed binaries no longer depend on the caller's working directory. Frozen SHA-256/count tests cover every asset.
- Ported the required evaluation subset with strict unique/unknown-ID handling, every-gold-row denominators, semantic-ID probability alignment and first-tie behavior, per-family represented-class balanced accuracy/macro-F1, mean-family headlines, `1e-12` NLL floor, Brier sum, probability coverage, and null complete-distribution metrics. Imported evaluation JSON uses ordinary floating-point serde parsing rather than the integer-only state parser; malformed rows with a usable ID become explicit invalid rows.
- Added perturbation stability joined from each `provenance.base_id` to the authored original, semantic-ID-aligned modal agreement and total variation, by-variant summaries, equal source-group macro summaries, explicit missing/invalid coverage, and a mandatory complete 36-original baseline for `perturbations108` imports. No perturbation is silently used as its own baseline.
- Added exact published BF16 aggregate values for all three models and explicit row-level comparison only for the Qwen 252-row evidence. Full authored144 and perturbations108 Qwen rows match the preserved Python metric subset to `1e-12`; the hand differential still covers missing, invalid, semantic alignment and ties. Bootstrap/calibration fields are explicitly absent rather than fabricated.
- Added create-only eval report/raw-prediction outputs with retained hashes, model/config metadata where present, fixture identity, limitations, and JSON-only stdout/stderr behavior. Inference-backed eval requires raw `--predictions-output` evidence and uses one direct owner-worker model load.
- Added project-authored 703-byte and approximately 8,000-byte state fixtures plus one 21-question JSONL set. Bench validates all input/output paths before model load, requires at least five repeats, warms separately, alternates direct/requested-shared order, uses one loaded owner-worker model, excludes loading/warmup/validation/writes from timed group wall, records actual rows/tokens/timing breakdowns, and never duplicates shared group timing across rows. A failed shared receipt measures the requested/emitted serial fallback but reports `shared_speedup=null` and the exact gate failure.
- Added M6 eval/bench JSON schemas and deterministic injected-sample median/p95/throughput tests. Backend-disabled process tests cover float prediction import, explicit invalid/missing denominator behavior, prevalidation, and create-only empty files on startup failure.

Pre-measurement ordinary gate: `cargo fmt --all`, workspace clippy with `-D warnings`, and workspace tests pass. Counts at this checkpoint are 113 test functions (21 CLI, 54 core, 38 backend), with no inference in ordinary tests. No M5 probe was rerun or changed; no receipt/tolerance/backend algorithm/reference file was edited; no model was downloaded and no new native target directory was created. Next action is to build only in existing `target-m2-metal`/`target-m2-cpu`, run feature gates and CLI smokes, then create one-at-a-time quality and timing evidence under `docs/results/m6/`.

## User-requested compact decision output (separate from M6)

Added opt-in `--compact` projection for `decide`, `noul`, `score`, `ask`, and `run` while leaving default full readouts unchanged. Compact rows retain ID/choice, original option arrays, the exact probability honesty string, primitive-specific values, requested confidence, and only the requested/effective/fallback execution subset when fallback occurred. Errors, JSONL row flushing/continuation, create-only summaries, and `--pretty` constraints remain unchanged. Non-decision commands reject the flag with a structured validation record; help documents its scope.

`--quiet` subscriber setup now uses a clap-parsed `Cli` rather than scanning argv, suppressing routine tracing/native INFO and DEBUG while retaining WARN/ERROR and explicit semantic fallback warnings. Added `docs/COMPACT.md`, `schemas/compact-v1.schema.json`, README usage, typed projection/schema tests, and backend-disabled process tests. This work does not change M6 eval/bench outputs, model listing, core Readout/storage, native scoring, probes, or benchmark files. No commit was created.

Validation completed before the final test-only warning assertion: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` all exited 0; the workspace run executed 117 tests (25 CLI, 54 core, 38 backend), 0 failed. The existing `target-m2-metal/release/openjev` was rebuilt in place with `GGML_METAL=ON ... --features metal`; no CPU binary or new target directory was built. The opt-in cached-Qwen release Metal compact/quiet process test then passed 1/1 and confirmed compact fields plus no native INFO/DEBUG stderr; one model WARN remained visible as required. A direct command also exited 0 with one `openjev-compact-v1` row and no INFO/DEBUG stderr.

## M6 — bounded measurement checkpoint and handoff status

This checkpoint is the durable answer to the user's status request. **M6 is not yet complete or approved. No native benchmark process is currently running.** The implementation and ordinary/feature gates are complete, all requested Metal quality runs are retained, and five of the six minimum short-state device/model benchmark cells are complete. The remaining required cell is Qwen3.5-4B true CPU. Final documentation, final post-compact gates, schema validation against a native bench report, and Astra review also remain.

Completed measured quality, all on Metal with direct inference, one loaded model per command, existing verified cache, existing `target-m2-metal`, and create-only raw/report/stdout/stderr files under `docs/results/m6/`:

| Model | authored144 mean-family balanced accuracy | BF16 published | perturbations108 mean-family balanced accuracy | BF16 published | perturbation modal agreement | mean TV |
|---|---:|---:|---:|---:|---:|---:|
| `qwen3-0.6b` | 0.4475764575 | 0.4403525153 | 0.5388007055 | 0.5276895944 | 0.8888888889 | 0.1156075405 |
| `minicpm5-2b` | 0.6223418022 | 0.6862540338 | 0.7246913580 | 0.6925925926 | 0.8055555556 | 0.1949458361 |
| `qwen3.5-4b` | 0.8030296329 | 0.8132381608 | 0.7754850088 | 0.7657848325 | 0.8240740741 | 0.1752005732 |

These are measured pinned-GGUF/backend gaps, not quantization-only attribution. Every authored report scored 144/144 with zero invalid rows. Every perturbation report scored 108/108 plus retained predictions for the 36 authored-original baselines; stability coverage is 108/108. Qwen's approximately 0.81 expectation applies to Qwen3.5 and is met at 0.8030 as a measured non-gating result.

Completed 1-state × 21-question benchmarks use five direct and five requested-shared samples in alternating order after a separate warmup. Every requested-shared sample actually emitted serial full-prompt fallback because no passing exact receipt exists. Consequently every report correctly has `shared_speedup=null`; none claims a shared speedup.

| Model | Device/state | direct median group wall (s) | requested-shared fallback median (s) | status |
|---|---|---:|---:|---|
| `qwen3-0.6b` | Metal, 703 bytes | 3.323810 | 3.339685 | complete |
| `minicpm5-2b` | Metal, 703 bytes | 7.042852 | 7.057393 | complete |
| `qwen3.5-4b` | Metal, 703 bytes | 13.058682 | 13.030589 | complete |
| `qwen3-0.6b` | Metal, 8,068 bytes | 13.196053 | 13.219938 | complete long-state coverage |
| `qwen3-0.6b` | true CPU, 703 bytes, 11 threads, n_ctx=512 | 255.110770 | 253.567627 | complete |
| `minicpm5-2b` | true CPU, 703 bytes, 11 threads, n_ctx=512 | 425.680628 | 444.267292 | complete but potentially host-contended; exploratory, not an isolated baseline |
| `qwen3.5-4b` | true CPU, 703 bytes | — | — | **not run; required remaining cell** |

The MiniCPM CPU capture overlapped user-requested compact-scope workspace clippy/tests and a Metal release rebuild on the same host. Its report and raw samples are preserved, but the latency is explicitly potentially contended and must not be presented as an isolated baseline. Per user instruction it was not automatically rerun for hours. A targeted rerun is the remediation if an isolated MiniCPM CPU baseline is required.

An earlier Qwen CPU attempt used implicit host-default threads and automatic 4,096 context. The harness terminated it before completion, leaving create-only zero-byte report/sample/stdout files and retained native stderr. `qwen3-0.6b-cpu-short-bench.failed.json` records that failure without deriving partial timing. The successful `*-final` capture uses explicit 11 host threads and n_ctx=512; all observed fixture prompts fit and the exact configuration remained receipt-ineligible, so requested shared still safely fell back without any probe or gate change.

Feature gates before measurements passed in both existing native targets: Metal and true-CPU clippy with warnings denied plus all native/CLI all-target tests. No M5 receipt was enabled, regenerated, loosened or probed; no backend algorithm, tolerance or `reference/` file changed; no model was downloaded; no new native target directory or commit was created. Because compact edits landed after those gates and rebuilt only Metal release, the CPU release binary and final feature/workspace gates must be refreshed before final review. The M3 exact gate need not be rerun solely for M6 because no engine/backend source changed, but its existing 144/144 invariant must remain noted.

Bounded resume contract: first rebuild the CPU release binary in existing `target-m2-cpu` to include compact changes; run only the missing Qwen3.5 CPU short benchmark if the user/parent elects to spend the expected long runtime; do not rerun the contended MiniCPM cell unless an isolated baseline is explicitly required. Then validate one native eval and bench report against schemas, finish README/todo/RESULTS plus evidence hashes, rerun final fmt/workspace/default/Metal/CPU gates, diff/reference checks, and hand the uncommitted tree to Astra. If the Qwen3.5 CPU command exceeds the environment's long-command limit, retain zero/partial evidence and a create-only failed-attempt reason rather than fabricating or truncating a result.

## Resident HTTP server extension — starting checkpoint

The user requested `--serve` with Jev web API compatibility. Before that extension, the existing compact-output and partial M6 work is checkpointed separately to avoid mixing provenance. This checkpoint is **not M6 completion or a performance-gate approval**: Qwen3.5 CPU timing, final M6 review/writeup, and M7 remain outstanding as described above. No shared/batch profile has been enabled. Parent reran `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace`: all passed (117 tests, native integrations not enabled). No models were downloaded or long benchmarks started.

## Resident HTTP server extension — uncommitted implementation evidence

Implemented the separately approved `openjev --serve` extension without changing core prompt bytes, native receipt gates, the default command response schemas, or the checkpointed M6 algorithms. The root parser now accepts either one existing subcommand or `--serve`; server-only and decision-projection-only combinations fail before model loading. The pinned HTTP stack is axum 0.8.9 and Tokio 1.53.1. One dedicated synchronous owner thread constructs and retains one `NativeScorer`; the async frontend sends owned jobs through bounded channels and never adds `Send` to the scorer trait. Startup binds before expensive loading but does not begin serving until one model load and one validated disclosed warmup succeed. Startup failures join the owner thread and never advertise readiness.

The service exposes `POST /v1/systemone`, `GET /v1/models`, `GET /healthz`, and `GET /readyz`. It defaults to loopback, rejects non-loopback binds without a nonempty indirect bearer secret, enables no CORS, logs no request-derived content/secrets, caps bodies (including streaming bodies) at 1 MiB, and admits at most 16 body/queue/running jobs. A positive whole-request deadline covers body read, queue, and inference. Timed-out or disconnected queued work is skipped; native decode remains noninterruptible and keeps its permit until it actually completes. SIGINT/SIGTERM closes readiness/admission and joins shutdown after in-flight native work. Worker panic clears readiness and drops reply senders rather than hanging clients.

The separate Jev adapter uses strict insertion-ordered JSON and supports the pinned SDK's Choice/Noul/Score subset. Arbitrary external IDs map to private stable IDs; singleton Choice does no inference; custom Noul criteria remain explicit decisions; Score reports expected index and preserves its typed legend. Original unrounded core probabilities determine first argmax, expectation, and normalized-margin confidence. Only the HTTP fields are independently rounded to two decimals and rounded distributions are not renormalized. The lean response reports actual loaded model identity and logical full-prompt input-token sum. Multi-question shared attempts retain exact receipt gating; any unavailable/failing chunk discards all tentative rows and fresh-serial-scores the entire request, while `--require-shared` fails closed. Fallback and conditional-probability status are disclosed only through bounded static/ASCII metadata and diagnostics.

`docs/SERVE.md`, `schemas/jev-http-v1.schema.json`, pinned `scripts/sdk-compat/` source/lockfile, README/todo/source credits, adapter tests, router/worker injection tests, and an opt-in native server test are present. The SDK compatibility source pins `@typesafe-ai/sdk` 0.6.0 / commit `66880ccded6cb642dc1809620c2b108c33730214`; a clean `npm ci` and module import passed, then `node_modules` was removed. The source uses a root `baseURL`, mixed Choice/Noul/Score, models, and a 404 path. Parent may run the final SDK-against-server invocation without changing the test source.

Current local gates (all exit 0) include `cargo fmt --all`; offline workspace clippy with warnings denied; offline workspace tests (131 tests, 0 failed, including 24 CLI unit/router tests, 10 process tests, schema/compact tests, 54 core tests, and 38 backend tests); Metal and true-CPU all-target feature clippy; and Metal/CPU all-target feature tests (82 each, 0 failed; the guarded real server body is skipped unless `OPENJEV_INTEGRATION=1`). A Metal release build in existing `target-m2-metal` passed. Prompt goldens remain 144 authored + 108 perturbations, and `reference/` is unchanged. No GGUF was downloaded, no new target was created, and no long benchmark was started by this extension.

The required real offline cached Qwen3-0.6B Metal smoke was deliberately postponed after the pre-GPU process check found an unrelated active `laya-goldens --profile all --device cpu` benchmark (PID 34426). No concurrent GPU/model work was started. The opt-in test source will load the selected model once, force Metal, issue repeated mixed requests, check actual identity/typed answers/token usage/fallback disclosure/process health/stdout, and terminate via SIGTERM. This runtime gate remains explicitly pending until the competing benchmark exits; no production-readiness inference is made from compile-only feature tests. M6 remains partial at checkpoint `1dcfad3`, Qwen3.5 CPU timing and final M6 review remain outstanding, M7 remains outstanding, and no shared/batch receipt has been enabled.

## Resident HTTP server extension — final review and acceptance

Parent/Astra reviewed the complete HTTP/worker and Jev adapter implementation, CLI changes, schema, tests, and actual pinned official SDK sources (`src/types.ts`, `src/client.ts`, `src/resources/models.ts`). Review identified and Sol fixed: (1) `--require-shared` incorrectly permitting direct single-question requests; (2) swallowed worker panic/shutdown errors; (3) underlying native-worker failure leaving readiness true; and (4) excessive per-question cloning of large state before native token validation. Regressions now prove zero client inference for unsupported shared requests, typed 422 errors, nonzero propagation of shutdown/panic errors, 503 readiness after terminal worker failures, and a conservative 4 MiB expanded-state budget checked before cloning. The native HTTP test now selects Metal only for macOS Metal builds and CPU otherwise. Synthetic shared test results remain test-only; no production receipt was created or enabled.

The initial real cached-model HTTP/SDK smoke passed and is preserved at `docs/results/serve/20260920T030040Z/`. After all source fixes, the complete real smoke was rebuilt and rerun at `docs/results/serve/20260920T031535Z-final/`: one native Metal integration test passed, and the strengthened official `@typesafe-ai/sdk@0.6.0` client passed against a real authenticated Qwen3-0.6B Metal server. It checked all Choice/Noul/Score fields, distributions/rounding, Score legend, usage (338 logical input tokens, zero output tokens), model listing, unknown-model 404, unsupported-float 422, and wrong-key 401. The native test checked repeat-request equality, fallback headers, live process residency, empty stdout, and SIGTERM shutdown. Logs recorded one model load, Metal MTL0/Apple M3 Pro, and disclosed serial fallback. Both test servers exited successfully; no model download or latency benchmark occurred.

The final release binary SHA-256 is `476a5b9414a702e0e2c7cfa792dce24bdd724ed1a825cdb470071f2b3582dcbb`. Parent independently verified every retained smoke checksum and the source/binary hashes against the final files. `node_modules` and smoke servers were removed. SDK smoke assertions were strengthened by parent to validate actual fields, not merely the SDK call returning without an exception.

Final parent gates, all exit 0, are retained at `docs/results/serve/20260920T031934Z-review-gates/`: `cargo fmt --all -- --check`, workspace clippy with `-D warnings`, workspace tests (135 tests, excluding two nested cache-test subprocess reports), and CPU/Metal all-target workspace clippy/tests using the existing native target directories with locked offline dependencies. Those feature test sweeps left `OPENJEV_INTEGRATION` unset; they are not a claim that guarded native test bodies ran. The real runtime evidence is the explicitly enabled Metal HTTP integration and SDK smoke above. Existing 144 authored and 108 perturbation prompt goldens pass. No core/backend implementation or `reference/` file changed.

The resident HTTP extension is approved for commit/push as a bounded Jev wire-compatible service, not hosted Jev prediction parity or a parallel-inference speedup. Documentation records float/option/context limits, empty-state adapter handling, singleton behavior, confidence and wire rounding, startup-only model/probe configuration, and cancellation limits. Original M6 completion, M7 work, and investigation of shared-path numerical differences remain outstanding.

## Tagged release CI and packaging — implementation handoff

Implemented the bounded release-tooling portion of the active downloaded-release goal from `docs/plans/tagged-releases.md`. `.github/workflows/ci-release.yml` runs on main pushes, pull requests, `v*` tags, and manual dispatch. Its pinned `macos-14` arm64 and `ubuntu-22.04` x86-64 jobs verify runner architecture and the event commit, install Rust 1.95.0 rustfmt/clippy, compile native plus integration surfaces with `OPENJEV_INTEGRATION` unset, run workspace all-target clippy/tests, and build the release CLI with Metal on macOS or CPU on Linux. Tag publication needs both jobs, grants `contents:write` only in that job, validates exact `v<Cargo version>` binding and tag commit identity, downloads both workflow artifacts, verifies their manifests, writes one `SHA256SUMS`, and creates a new release without clobbering an existing one. Checkout/upload/download actions use the reviewed pinned SHAs and build checkouts do not persist credentials.

Native release configuration explicitly avoids host tuning. macOS sets deployment target 14.0 and `GGML_METAL_EMBED_LIBRARY=ON`. Linux sets Rust `target-cpu=x86-64`, `GGML_NATIVE=OFF`, and disables optional SSE4.2/AVX/AVX2/AVX-VNNI/AVX512/FMA/F16C/BMI2 CMake switches. The Linux archive is documented as glibc 2.35+ with system libstdc++/libgcc, not static-musl. `scripts/check_native_build.py` checks every retained release-relevant CMake cache value and reports complete expected/actual diagnostics while deliberately ignoring only the non-semantic BOOL/STRING/UNINITIALIZED type marker. Each build checks `file` plus `otool -L` or `ldd`; the macOS check permits only `/usr/lib` and `/System/Library` dependencies, and Linux requires every dependency to resolve under standard system library roots.

`scripts/release.py` validates ref/version binding, binary JSON `--version`/`--help`, creates a normalized versioned archive, and writes `BUILD-INFO.json` with source SHA/ref, workflow URL, target, exact Rust/native/backend identity, system requirements, distribution disclosures, and payload SHA-256 values. Archives contain only `openjev`, `LICENSE`, `THIRD_PARTY.md`, `THIRD_PARTY_LICENSES.html`, the official unmodified Rust 1.95.0 compiler-payload `RUST-COPYRIGHT-library.html`, the complete unmodified crates.io archives `colored-3.1.1.crate` and `option-ext-0.2.0.crate`, release `README.md`, `SERVE.md`, and the manifest. The generated notice bundle covers both release-target Cargo closures and hash-pinned native llama.cpp/ggml notices; CI checks its recorded `Cargo.lock` and bundle hashes, verifies the Rust notice against its recorded compiler version/commit and SHA-256, and verifies each MPL covered-source archive against its exact lockfile checksum without installing cargo-about or fetching license sources. Verification rejects absolute/traversal, duplicate, linked, device, extra, missing, or reordered members; safely extracts into a fresh system temporary directory outside the checkout; verifies every payload hash; then executes help/version there. `docs/RELEASE.md` documents checksum verification, runtime requirements, MPL source availability, local install, absence of model weights, and that the macOS binary is neither Developer ID signed nor notarized.

Local automated packaging tests use fake executables and perform no native rebuild or model access. `scripts/tests/test_release.py` covers deterministic bytes, exact archive structure and manifest, byte-identical covered-source mapping and manifest hashes, outside-checkout help/version smoke, exact tag/Cargo/binary matching, source-SHA mismatch, traversal rejection, and two-target checksum generation. `scripts/tests/test_check_native_build.py` covers BOOL/STRING/UNINITIALIZED equivalence, wrong and missing values, complete diagnostics, and the exact macOS/Linux gate sets. A separate local check packaged the already-existing `target-m2-metal/release/openjev` without rebuilding it, safely extracted and executed its JSON help/version outside the checkout, identified it as Mach-O arm64, and confirmed `otool -L` reported no non-system dependency; the temporary archive was removed. This is a packaging/linkage check of a prior local binary, not the future GitHub artifact or downloaded-release smoke.

Final local tooling gates all exited 0: Python 3.9 release tests (4 test methods), Python bytecode compilation, `actionlint` including its shell checks, `cargo fmt --all -- --check`, locked offline workspace all-target clippy with warnings denied, locked offline workspace tests (135 test functions: 43 CLI, 54 core, 38 backend; native integration bodies disabled), `git diff --check`, and unchanged `reference/`. Publication, push/tag creation, release download/install, actual downloaded Metal server smoke, and official SDK smoke remain parent-owned and explicitly unrun. CI runner behavior and Linux linkage remain pending until the workflow runs; no release success is inferred from local packaging tests.

Parent/Astra release-tooling review: approved for first hosted CI run, not release acceptance. Fixed explicit toolchain selection (`RUSTUP_TOOLCHAIN=1.95.0`) so installing Rust cannot accidentally leave jobs using the runner default. Constrained CMake-cache verification to the llama-cpp-sys build rather than unrelated dependencies, and bounded Rust/CMake parallelism to two for standard private runners. Independently reran actionlint, 4 packaging tests, fmt, workspace clippy with warnings denied, and all 135 workspace tests; all passed.

First hosted run 35489637122 completed the Linux job successfully. The macOS Rust installation, clippy, tests, and release build succeeded, but the native CMake verification failed because its shell greps required exact cache type markers such as `:BOOL=` and `:STRING=`. The retained log did not print the actual cache entries, so it does not establish a wrong build value. The bounded follow-up replaces only those greps with `scripts/check_native_build.py`: it keeps the sys-only cache lookup and every value gate, ignores only the type marker (including CLI-originated `UNINITIALIZED`), and prints all expected and actual values if a fresh hosted run finds a real mismatch.

The same follow-up adds the exact crates.io `colored-3.1.1.crate` and `option-ext-0.2.0.crate` source archives for the specifically approved MPL-2.0 dependencies. Their checked-in bytes match the corresponding `Cargo.lock` SHA-256 checksums, CI verifies that identity network-free, both release archives include them as root files, and `BUILD-INFO.json` hashes them. The final packaging-only license addition also preserves the official Rust 1.95.0 `COPYRIGHT-library.html` compiler payload for standard-library and compiled-runtime attribution. Metadata records its SHA-256 and exact `rustc -vV` release/commit, regeneration copies it only from the matching installed compiler after checking the `rustc` component manifest, and the network-free checker binds the copy to the workflow's `RUSTUP_TOOLCHAIN` pin. Existing MPL source archives remain unchanged. A fresh hosted run remains required before tag creation or downloaded-artifact acceptance.

## Hosted release CI — first iteration and remediation

Hosted main run [35489637122](https://github.com/codesoda/openjev-rs/actions/runs/35489637122) on `9c72968` passed the entire Linux job. macOS passed native clippy, workspace tests, and release compilation, then failed the strict CMake-cache grep gate. No tag/release was created. Retained failure evidence is under `docs/results/releases/ci-35489637122/`. The replacement parser preserves every required configuration value but ignores irrelevant BOOL/STRING/UNINITIALIZED cache-type spelling and prints actual values on failure; the prior log did not identify which grep failed, so the exact cause is not asserted until the next hosted run.

Independent Astra review also identified missing full dependency notices in binary archives. Release packaging now includes attributed cargo-about notices for 276 pinned packages, exact native vendor notices, the official matching Rust 1.95.0 library copyright/permission bundle, and complete unmodified `.crate` source archives for the two MPL-2.0-only covered dependencies. Source archives are verified against Cargo.lock checksums. Parent reviewed and accepted this explicit source-availability/notice remediation for the present graph, not a blanket licensing exception. Network-free CI checks fail if lock, notices, source archives, metadata, native pin or toolchain pin drift. Added SHA-pinned standard Actions caching of Cargo sources and target outputs, keyed by target/toolchain/lock/workflow, to avoid repeating cold compilation for the release tag. No model cache or credentials are cached.

Local gates after remediation: license/source checker, seven Python tests, actionlint, fmt, workspace clippy with warnings denied and 135 workspace tests all pass. Rust core/backend and reference files remain unchanged. Next required gate is a successful hosted main build, then tag publication and installed downloaded-artifact smoke.

## Hosted macOS server-test flake remediation

Hosted run 35492156731 on macOS 14 arm64 exposed a test-only startup race: two deadline tests used 30 ms wall-clock deadlines and 5–10 ms sleeps, so a loaded runner could expire the HTTP request before the fake scorer entered its first noninterruptible call. The resulting warmup-only count was 1 rather than the asserted warmup-plus-inference count of 2. Production timeout, cancellation, admission, scoring, and parity behavior is unchanged.

The three timing-sensitive server tests now inject a blocking scorer gate backed by a condition variable. Tests wait for an explicit worker-entry notification before driving cancellation, keep native work blocked until an explicit release, and use a release-on-drop guard plus bounded 30-second synchronization waits so a failed assertion cannot strand `WorkerHandle::drop` joining a blocked owner thread. The disconnected-request test likewise waits until the first call entered and both admission permits are held before aborting the queued request; it no longer relies on 5/10/50 ms sleeps.

The two HTTP deadline tests run on Tokio's current-thread runtime. They begin with a generous 60-second real request deadline, pause Tokio time only after the worker has entered the blocking scorer, then manually advance Tokio time to produce the real handler 504 and cancellation flag. This does not advance the worker's `std::time::Instant`; after explicit release, the first native call completes while canceled queued work is skipped. The admission-one test asserts the permit remains charged after the 504 and returns 429 with `Retry-After` until the gate is released. Tokio's `test-util` feature is enabled only as an `openjev-cli` dev dependency. `Cargo.lock` remains byte-for-byte unchanged at SHA-256 `fe4334a69de7f19123c9821226bb879112e490b724db2a5973f318edca4e71cf`.

Validation after the test fix, all exit 0:

- All ten `server::tests` passed 25/25 repeated serial iterations (`--test-threads=1`) and one additional high-concurrency run (`--test-threads=16`).
- `cargo fmt --all -- --check`.
- Workspace and default-member all-target Clippy with warnings denied.
- Workspace tests passed twice, with `--test-threads=16` and `--test-threads=1`: 135 test functions each time, plus zero-test guarded/doc harnesses.
- Default-member tests passed with `--test-threads=16`: 97 test functions, plus zero-test guarded/doc harnesses.
- The third-party license/source checker passed for 276 packages and two source archives; `git diff --check`, unchanged `reference/`, and unchanged `Cargo.lock` checks passed.

No native build, model access, commit, push, tag, release, or CI monitoring was performed.

Parent/Astra inspected the complete deterministic-test diff: all Rust edits are inside `#[cfg(test)]`; Tokio test-util is dev-only and Cargo.lock is unchanged. Gate entry is acknowledged before advancing Tokio timeout time, queued admission is observed explicitly, and a drop guard releases the mock worker during unwinding. The tests retain 504/429, permit-retention, cancellation, and exact scorer-count assertions; no production deadline or native numerical gate changed. Parent independently reran fmt, warnings-denied workspace clippy, 135 workspace tests and the license/source checker successfully. Hosted run 35492156731 passed Linux; its failed macOS scheduling assertions are preserved under `docs/results/releases/ci-35492156731/`. Approved for another hosted run, not yet a release acceptance.

## v0.1.0 hosted CI, publication, and downloaded release acceptance

The final hosted main run [35493609481](https://github.com/codesoda/openjev-rs/actions/runs/35493609481) passed on exact source `bb23406606e423fb35f5e62fdf5f170a6b14ad3f`. Parent then created tag `v0.1.0` at that exact commit. Tag run [35494477837](https://github.com/codesoda/openjev-rs/actions/runs/35494477837) completed successfully: both native macOS/Linux jobs passed formatting, warnings-denied clippy, workspace tests, release build, CMake configuration validation, linkage/package verification and outside-checkout help/version smoke, then the publication job verified both archives and created the immutable GitHub Release. The repository was independently confirmed private throughout.

Parent downloaded `openjev-v0.1.0-aarch64-apple-darwin.tar.gz`, `openjev-v0.1.0-x86_64-unknown-linux-gnu.tar.gz`, and `SHA256SUMS` using `gh release download`. The checksum file and GitHub asset digests agreed (`b2b26ee4…96251` macOS, `4430be0d…9b8e6` Linux, and `3cf1f0d6…34967` for `SHA256SUMS`). Safe member lists, packaged manifests/payload hashes, and macOS linkage passed. Linux was archive-verified locally but deliberately not executed on macOS; hosted Linux CI had already run package help/version/linkage. No Linux model inference occurred.

No prior binary was replaced. The downloaded macOS payload was installed unchanged at `~/.local/share/openjev/releases/v0.1.0/openjev-v0.1.0-aarch64-apple-darwin/`, with `~/.local/bin/openjev` pointing to it and already on `PATH`. Parent independently matched the resolved installed binary to release source/tag/run metadata, version `0.1.0`, and SHA-256 `b9999f65f936fdd17e193af90e57bd568c2888f98889b15c3ce98d1a319fc66a` from packaged `BUILD-INFO.json`.

The real installed-release gate started at `2026-09-20T06:37:12Z` and passed using the existing verified Qwen3-0.6B cache, `--offline`, and explicit Metal from a temporary directory outside the checkout. Eight raw HTTP checks covered health/readiness, three equal decoded JSON mixed Choice/Noul/Score responses, 401/404/422 paths and 313 input / 0 output tokens. The same resident PID stayed live through all requests; logs show one pinned model load, the embedded Metal library and `MTL0 (Apple M3 Pro)`; stdout stayed empty; SIGTERM returned 0. The pinned official `@typesafe-ai/sdk` 0.6.0 smoke passed with 338 input / 0 output tokens. Every shared request explicitly reported serial full-prompt fallback. The smoke process exited and temporary `node_modules` was removed.

Final parent local documentation/evidence gates all passed without another native rebuild or model inference: `cargo fmt --all`; locked offline workspace all-target clippy with warnings denied; locked offline workspace tests (135 unique tests plus two child-process repeated tests); seven Python packaging/native-check tests; the license/source checker; and `actionlint`. The retained `local-clippy.log` and `local-tests.log` are in `docs/results/releases/v0.1.0-downloaded-metal/`. `EVIDENCE.sha256` binds all 28 raw evidence files while deliberately excluding the explanatory `README.md` and the checksum manifest itself; all 28 entries were independently verified, and a credential-pattern scan was clean. Parent also independently confirmed `crates/`, core/backend implementation, `Cargo.lock`, and `reference/` are unchanged from the tagged source.

This closes the bounded tagged-release/download/install/Metal HTTP/SDK acceptance only. It makes no acceleration, performance, parity, Linux inference, complete platform/model, signing/notarization, full-brief, M6-completion, or M7-completion claim. Raw evidence and its harness are under [`results/releases/v0.1.0-downloaded-metal/`](results/releases/v0.1.0-downloaded-metal/).

Independent Astra final acceptance review: **PASS, no release blockers.** The reviewer independently checked local/remote tag identity, live successful CI/release metadata and GitHub asset digests, both archive/payload hashes, installed binary identity/version/system linkage, retained HTTP/SDK/Metal evidence, process cleanup, repository privacy and unchanged implementation/lock/scripts/workflow/reference paths. Parent reviewed Sol's documentation diff and corrected the distinctions between decoded-response equality and byte equality, and between smoke start time and completion.

Provenance remains deliberately split: the immutable release source is tag `v0.1.0` / commit `bb234066…ad3f`; this documentation-only follow-up records the later downloaded-artifact evidence and final review. It does not move the release tag, replace published assets, or change implementation. The commit containing this section is the evidence follow-up, not the release source.

## HTTP demo command

Added `openjev demo`, an HTTP-only client that posts eight short examples to a
running server's `/v1/systemone`. Covers Choice, Noul, Score, and a mixed-question
request. The command does not initialize a backend or download weights. Default
URL is `http://127.0.0.1:8080`; supports `--base-url`, `--api-key-env`,
`--timeout-secs`, and an optional `--model` selector for the already-resident
model. Progress goes to stderr; stdout is JSONL, or one array with `--pretty`.
Rows retain the Jev response, round-trip milliseconds, and execution/fallback/
probability headers. Errors stop the run without following redirects; responses
are capped at 1 MiB. README documents two-terminal usage and the release boundary.

Validation passed:

- `cargo fmt --all -- --check`.
- `cargo clippy --locked --offline --workspace --all-targets -- -D warnings`.
- `cargo test --locked --offline --workspace`: 142 test functions, including seven
  new demo tests covering real mock HTTP transport, all example payloads, auth,
  default/custom model selection, output modes, HTTP errors, redirects, malformed
  and oversized responses, timeout, invalid URL, and unavailable-server guidance.
- Regenerated and checked third-party license notices. Reqwest 0.13.5 was already
  locked; the only lockfile change adds it to the CLI's dependencies. The release
  license closure remains 276 packages.
- Actual HTTP smoke: the new non-native debug client sent all eight examples to
  the installed v0.1.0 server using cached Qwen3-0.6B Q8_0, offline, explicitly
  Metal (Apple M3 Pro). Seven single-question round trips were 129–236 ms; the
  mixed three-question request was 397 ms. These are one-run smoke timings, not
  benchmarks (other validation ran concurrently). All responses reported the
  correct model and zero output tokens; the mixed request disclosed serial
  fallback. The owned server stopped cleanly with SIGTERM and empty stdout.
  Raw demo output: [`results/demo/qwen3-0.6b-metal.json`](results/demo/qwen3-0.6b-metal.json).

No backend numerical behavior changed. No models were downloaded, no native
rebuild was needed for this client smoke, and no installed binary or immutable
release was replaced. The demo is not included in published v0.1.0.

### Local installation and `serve` subcommand

Following the explicit install request and decision against backward
compatibility, the server is now `openjev serve`; the former `--serve` flag is
rejected. Server-only arguments belong after the subcommand. Updated help,
README, current HTTP documentation, demo diagnostics, and native/server CLI
tests; historical release evidence and original design notes are unchanged.

Fmt, warnings-denied workspace/all-target Clippy, and all 142 workspace tests
passed again. Built the optimized Metal executable using the existing
`target-m2-metal` cache and installed it via the existing
`~/.local/bin/openjev` → `~/.openjev/bin/openjev` chain. The active payload is
`~/.openjev/bin/openjev-local-d894391c21a0/openjev`, SHA-256
`d894391c21a05e79267ab6467a951a8f9abed89b132847490d0ce77fc33e99de`.
It is labelled as a local development build, not a GitHub release; earlier
payloads, including v0.1.0, remain available unchanged.

The installed PATH binary passed `serve --help`, `demo --help`, and legacy
`--serve` rejection. From outside the checkout it then served cached Qwen3-0.6B
with explicit Metal, while another invocation of the installed binary completed
all eight demo requests. Verified resident model identity, mixed three-answer
response and disclosed serial fallback, empty server stdout, Apple M3 Pro Metal
backend, and clean SIGTERM shutdown. No weights were downloaded. Raw results:
[`results/demo/qwen3-0.6b-metal-serve-subcommand.json`](results/demo/qwen3-0.6b-metal-serve-subcommand.json).

### README VHS recording

Added a top-of-README GIF and linked MP4, generated with `demo/readme.tape` via
`bash demo/record.sh`. The wrapper owns an offline cached-Qwen Metal server;
the tape runs the real demo HTTP client. `demo/present.py` shows an explicitly
labelled compact projection of all eight live responses, with reading pauses
and visible serial fallback. No predictions are substituted. Model startup is
outside capture and the paced playback is not a benchmark. Reproduction and
release-boundary notes are in `demo/README.md`.

VHS 0.11.0 generated a 32.52-second 1280×860 H.264 MP4 (~310 KiB) and GIF
(~315 KiB). Verified eight raw responses, model identity, zero generated tokens,
three mixed answers, and empty server stdout; reviewed sampled Choice, Score,
and final mixed frames for readability/clipping. ShellCheck, Python compilation,
VHS validation, FFprobe, and whitespace checks pass. Raw recording logs and rows
are under ignored `out/demo-recording/`. No Rust/backend changes or new model
downloads were needed for this media work.

### Slower input/result walkthrough and recorded startup

Replaced the initial fast clip with a 138.44-second recording (~1.2 MiB GIF,
~1.3 MiB MP4). It now begins with actual server launch, cache verification,
warmup and `/readyz`; startup is no longer off-camera. Each single-question
example shows its state/question/options for 8 seconds, then its result for
6 seconds. The mixed example gets 14 and 10 seconds. These are reading pauses,
not simulated inference delays. Inputs and answers are taken from real HTTP
traffic; `demo` rows now add the exact `request` object, with a transport test
asserting equality to the body received by the mock server. Other CLI decision
and Jev HTTP schemas are unchanged.

`demo/session.py` owns the real server/client lifecycle and types the commands
it executes. Verified all eight request/response pairs, clean shutdown, empty
server stdout/client stderr, and reviewed startup, single/mixed inputs and
results from the actual MP4. Workspace fmt/clippy/tests (142 tests), three
presenter tests, ShellCheck, Python compilation, VHS validation, and FFprobe
checks pass. Capture uses the fresh optimized Metal build via a temporary PATH
override, not a replacement of the user's installed binary.

README now contains only a GIF linked directly to the MP4; removed the extra
caption and auxiliary links. Reproduction and honesty notes remain in
`demo/README.md`. No predictions were changed or replaced, no weights downloaded,
and no public release created.

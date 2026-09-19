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

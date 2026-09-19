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

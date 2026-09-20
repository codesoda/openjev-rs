# openjev-rs — TODO / implementation brief

Rust port of the **openjev.com / SemIf** idea: a "System One" decision model
(Jev-style `Choice` / `Noul` / `Score` primitives) built by reading **next-token
option logits from a frozen open LLM** in one forward pass — no generation.
Ships as a library + an `openjev` CLI that takes state/question/options from
args or stdin and prints JSON to stdout.

Status: **nothing implemented yet**. This file is the brief. `reference/` holds
the upstream material the design is derived from.

Phase A update: design/source adjudication is in [`docs/PLAN.md`](docs/PLAN.md),
reviewed by parent/Astra and approved for M1. All six section-8 choices and failure policies are
resolved there; runtime gates remain unrun. Published llama crates are 0.1.156
(0.1.157 is currently Git-only), with the expected e79e4bf6 native pin. The plan
also corrects the brief's empty-option validation assumption, prediction-file
cardinality (252; select authored144 by ID), build override assumptions, and
chat-template API limitations. [`docs/PROGRESS.md`](docs/PROGRESS.md) records
actual evidence. M1–M5 are approved and committed; M3's strict authored144 plus perturbations108 prompt/token/slot gates pass. Partial M6 evaluation/benchmark and compact-output work is checkpointed at `1dcfad3` and remains incomplete. The separately requested resident `openjev --serve` Jev-compatible HTTP extension has passed parent/Astra review, real Metal HTTP/official SDK smoke tests, and workspace/CPU/Metal checks; evidence is in `docs/results/serve/`. Tag release CI and deterministic package validation are now implemented locally under `.github/workflows/ci-release.yml` and `scripts/release.py`, but no tag, GitHub Release, installation, or downloaded-artifact smoke is claimed yet. It does not complete M6 or the remaining M7 algorithms and does not enable any failed shared/batch profile. **The full original brief below is preserved**, not rewritten as if later runtime validation had occurred.

---

## 0. What openjev.com actually is (ground truth)

- Renamed **SemIf**; independent project by TheoLeeCJ, MIT, not affiliated with
  TypeSafe. Repo: <https://github.com/TheoLeeCJ/openjev>.
- **No custom model.** Frozen, off-the-shelf checkpoints. Pinned artifacts
  (`reference/semif-py/manifests/models.json`):

  | id | GGUF (browser/llama.cpp) | quant | bytes | native ref |
  | --- | --- | --- | --- | --- |
  | `qwen3-0.6b` | `Qwen/Qwen3-0.6B-GGUF` @ `23749fef…` `Qwen3-0.6B-Q8_0.gguf` | Q8_0 | 639 MB | `Qwen/Qwen3-0.6B` @ `c1899de2…` |
  | `minicpm5-2b` | `openbmb/MiniCPM5-2B-GGUF` @ `2079a22f…` `MiniCPM5-2B-Q4_K_M.gguf` | Q4_K_M | 1.56 GB | `openbmb/MiniCPM5-2B` @ `12a3808a…` |
  | `qwen3.5-4b` | `bartowski/Qwen_Qwen3.5-4B-GGUF` @ `4168f45a…` `Qwen_Qwen3.5-4B-Q4_K_M.gguf` | Q4_K_M | 3.01 GB | `Qwen/Qwen3.5-4B` @ `851bf6e8…` |

  (Python CLI also has a `reranker` mode on `Qwen/Qwen3-Reranker-4B` — out of
  scope for v1.)

- **Mechanism** (`reference/semif-py/src/direct.py`, `core.py`):
  1. Messages: system = `DIRECT_SYSTEM` (exact string in `core.py`); user =
     `json.dumps({"evidence": state, "criterion": question, "options": [{"letter": "A", "description": …}, …]}, ensure_ascii=False)`.
  2. Apply the model's chat template with `add_generation_prompt=True,
     enable_thinking=False`.
  3. Verify each answer slot letter (`A`..`P`, 2–16 options) is exactly one
     tokenizer token, round-trips, no collisions, and that `tokenize(prompt +
     letter) == tokenize(prompt) + [slot]` (boundary check).
  4. One forward pass, take last-position logits, index the slot ids, softmax
     over **only** those. Nothing is sampled.
  5. Output per row: `probabilities`, `option_logits`, `input_tokens`,
     timings, `prompt_sha256`, `prompt_version: "direct-options-v1"`, model
     metadata, and the honesty strings (`readout`, `probability_status`).
- **Shared-state mode** (`shared.py`, `serial.py`): prefill the state prefix once
  (template up to and including `{"evidence": <state>` minus the final token),
  replicate the KV cache per question, run all suffixes as a batch, read one
  logit row per question. This is the Jev "state once, many questions" property.
- **Browser demo** (`webgpu-demo/worker.js`): same idea through wllama's
  chat-completion API (`max_tokens: 1`, grammar over letters, `top_logprobs: 20`,
  `logit_bias` on letter token ids — `labelBase` 32 for Qwen, 54 for MiniCPM).
  Uses a *different, simpler prompt* than the Python CLI; **we follow the Python
  prompt** (it has golden hashes and published numbers).
- **Published quality** (native BF16, `browser-model-ladder.json`): balanced
  accuracy authored144 / perturbations108 — 0.6B 0.440/0.528, 2B 0.686/0.693,
  4B 0.813/0.766. TypeSafe-102 modal agreement: 4B 0.845 vs published Jev 0.883.
- **Explicit caveats to carry over verbatim in our output**: probabilities are
  conditional on the supplied options and *not* calibrated confidence; a forced
  typed output can still be wrong.

## 1. Goals

1. `openjev` CLI: decisions in, JSON out, scriptable/pipeable, local. The later user-requested, separately reviewed `--serve` extension supersedes the original “no server” boundary without changing default CLI output.
2. Library crate usable from other Rust code (later: a `System1` trait shared
   with `gliner2-rs`, see `reference/gliner2-rs-notes/jev-and-gliner.md` §5).
3. Bit-for-bit **prompt parity** with the Python `direct-options-v1` prompt
   (verify via `prompt_sha256` goldens) and numerically close logits.
4. Shared-state multi-question via KV-cache sequence copy (the actual perf win).
5. Reproduce the authored144 / perturbations108 numbers within quantization
   tolerance, and print a timing report.

Non-goals for v1: training/fine-tuning, reranker mode, HTTP server (maybe v2),
browser/WASM, pixels/vision.

## 2. Proposed layout

```
openjev-rs/
  Cargo.toml                 # workspace
  crates/
    openjev-core/            # no llama dep: types, prompt, slots, softmax, calibration, eval math
    openjev-llama/           # backend on llama-cpp-2: model registry, download, scoring, shared-state
    openjev-cli/             # bin `openjev`
  reference/                 # upstream material (read-only)
  docs/                      # design notes, results
  todo.md  README.md  PROMPT.md
```

Rationale: core stays testable without a 3 GB model; backend isolated so a
second backend (mistral.rs / candle / remote) can be added; CLI thin.

## 3. Dependencies (verify versions at start)

- `llama-cpp-2` **0.1.157** (utilityai/llama-cpp-rs, master 2026-09-12;
  llama.cpp submodule `e79e4bf6`). Features: `metal` (macOS), `cuda`, `vulkan`,
  `openmp`, `dynamic-link`. API points confirmed present:
  `LlamaContext::get_logits_ith(i)`, `copy_kv_cache_seq(src, dst, p0, p1)`,
  `clear_kv_cache_seq`, `LlamaBatch` with per-token logits flag and seq ids,
  `LlamaModel::apply_chat_template` / `chat_template()`, `token_to_str`,
  `str_to_token`. Examples in `llama-cpp-2/examples/{simple,reranker,embeddings}`.
- `hf-hub` (blocking) for pinned-revision GGUF download into a cache dir
  (`~/.cache/openjev` or `$OPENJEV_HOME`), sha256 verify against manifest.
- `clap` (derive) for the CLI, `serde`/`serde_json`, `anyhow`/`thiserror`,
  `sha2`, `tracing` (logs → stderr only; stdout is JSON only).
- Fallback if `llama-cpp-2` build is painful on a target: `mistralrs` 0.8.1
  (pure Rust, GGUF, Metal/CUDA) — keep behind the backend boundary.

## 4. Core semantics to implement (`openjev-core`)

### 4.1 Types
- `Decision { id, state: StateValue(String|Json), question, options: Vec<Option{id, description}> }`
  — mirrors `validate_row` in `core.py` (2–16 options, unique ids, nonempty).
- `Primitive`: `Choice` (the base), `Noul` (sugar: options `yes`/`no` →
  `p_yes`), `Score` (sugar: ordered levels → distribution + expected value +
  argmax). Score/Noul are thin layers over Choice for v1; document that.
- `Readout { probabilities, option_logits, allowed_token_mass, full_vocab_argmax_id,
  input_tokens, forward_seconds, total_seconds, prompt_sha256, prompt_version,
  model{id, source, revision, quant, backend}, readout, probability_status }` —
  match the Python field names so the JSONL is interchangeable; include the
  extra fields the browser-ladder predictions have (`allowed_token_mass`,
  `answer_token_ids`, `full_vocab_log_normalizer`) — they need the full-vocab
  logits, which we have.
- `confidence` (opt-in field): Jev's normalised margin
  `(p_max − 1/K) / (1 − 1/K)`, labelled uncalibrated.

### 4.2 Prompt (`direct-options-v1`) — must be byte-identical to Python
- `DIRECT_SYSTEM` string verbatim.
- User payload = `json.dumps(payload, ensure_ascii=False)` → Python default
  separators `", "` and `": "`, key order `evidence, criterion, options`, option
  keys `letter, description`. `serde_json` compact output uses `,`/`:` with no
  spaces — **write a tiny custom serializer or post-process** to match Python
  spacing. State may be a JSON object/array → must re-serialise with the same
  Python rules (float formatting! restrict to what Python `repr` produces or
  reject non-string state with floats until proven).
- Chat template: prefer llama.cpp's built-in Jinja rendering of the GGUF's
  template. **Open question**: passing `enable_thinking=false` — llama.cpp's
  `common_chat_templates_apply` supports `chat_template_kwargs`; check whether
  `llama-cpp-2` exposes it. If not: (a) call the sys crate's Jinja path
  directly, or (b) hand-render the Qwen3/Qwen3.5/MiniCPM templates in Rust
  (Qwen with `enable_thinking=false` appends `<think>\n\n</think>\n\n` after the
  assistant header) and assert equality against the golden hashes.
- Golden test: `reference/semif-py/browser-ladder-qwen3-0.6b.predictions.jsonl`
  has `prompt_sha256` + `input_tokens` + `option_logits` for each authored144
  row (join on `id` with `benchmarks/data/authored144.jsonl`). Rust must
  reproduce the sha256 exactly and the token count exactly; logits within a
  quant tolerance (Q8_0 vs BF16: expect argmax agreement ≥ ~98 %, logit MAE
  small — measure and record, don't guess).

### 4.3 Slots
- Letters `ABCDEFGHIJKLMNOP` (16). Verify single-token + round-trip + no
  collision + boundary stability, exactly as `_slot_ids` / `encode_prompt`.
  Fail loudly, no silent fallback.
- v2 idea: extend to 20–26 or two-char labels only if boundary checks pass.

### 4.4 Numerics
- Softmax over slot logits in f64 (Python does f32 logits → f64 math).
- `allowed_token_mass` = Σ exp(slot − logsumexp(full vocab)).
- Optional post-hoc: option-order permutation averaging (`--permute N`,
  align by option id), temperature scaling (`--temperature T` on slot logits,
  plus a `calibrate` subcommand that fits T on labelled JSONL). Both clearly
  marked as deviations from `direct-options-v1` (bump `prompt_version` /
  add `postprocess` field).

### 4.5 Eval math
- Port the metric subset from `benchmarks/evaluate.py` needed for
  authored144 + perturbations108: per-family balanced accuracy, mean family
  balanced accuracy, plain accuracy, NLL/Brier. Perturbation stability = align
  by option id across variants of a `group_id`.

## 5. Backend (`openjev-llama`)

- `ModelRegistry`: the three pinned GGUFs (+ arbitrary `--model path.gguf` /
  `--model hf:repo@rev:file`). Download via `hf-hub` to cache; verify bytes
  against `models.json` sizes (sha256 if we record them once).
- `Engine::load(model, opts{ n_ctx (default 4096, auto-grow to prompt), n_batch,
  n_gpu_layers=all, threads })`. One `LlamaBackend`, one model, contexts per
  run.
- `score_direct(&Decision) -> Readout`: tokenize prompt via chat template,
  slot checks, batch with `logits=true` only on the last token, `decode`,
  `get_logits_ith(last)`, gather slots, softmax, timings.
- `score_shared(state, &[Question]) -> Vec<Readout>`: build prefix exactly as
  `shared.py::_state_prefix` (drop last token), verify every full prompt starts
  with it, prefill into seq 0, `copy_kv_cache_seq(0, i, ..)` for i in 1..n,
  batch every suffix with its seq id and `logits=true` on each suffix end, one
  `decode` (chunk by `n_batch` if needed), read per-seq logits. Then clear
  seqs 1..n for reuse; keep seq 0 as a warm state cache for a REPL/`--watch`
  mode later.
  - **Risk**: Qwen3.5 is a hybrid (Gated DeltaNet + attention). llama.cpp's
    hybrid memory supports full-sequence `seq_cp` but not partial rollback
    (the crate warns about this on `clear_kv_cache_seq`). Validate shared-mode
    logits == direct-mode logits (tolerance ~1e-3) on all three models; if a
    model fails, fall back to serial per-question full prompts for it.
- `score_batch(&[Decision])`: independent decisions packed into one batch
  with distinct seq ids (throughput for JSONL scoring).
- Thread safety: llama contexts are `!Send` in places; keep a single worker
  thread owning the engine and a channel API so the CLI/library can be async
  later.

## 6. CLI (`openjev`)

All results to **stdout as JSON** (one object, or JSONL for multi), logs to
stderr, non-zero exit on validation/model errors. `--pretty` for indented.

```
openjev decide  --question "Which queue?" --option "Account access" --option "Billing" \
                [--state "..." | --state-file f | (stdin = state)] [--id x]
openjev noul    --question "Is this phishing?" [state via stdin]        # → p_yes
openjev score   --question "How urgent?" --level low --level medium --level high
openjev run     [--mode direct|shared|batch] [--input f.jsonl | stdin JSONL] [--output f]  # semif-score parity
openjev ask     --json '{"state":..,"question":..,"options":[..]}'      # raw row
openjev models  [list | pull <id> | path <id>]
openjev eval    --fixture authored144|perturbations108 [--compare-to browser-ladder]
openjev calibrate --input labelled.jsonl  → temperature
openjev bench   --state-file big.txt --questions q.jsonl  # direct vs shared timing
```
Global: `--model qwen3.5-4b|minicpm5-2b|qwen3-0.6b|<path|hf spec>` (default
`minicpm5-2b`, like the site), `--n-ctx`, `--threads`, `--gpu-layers`,
`--permute N`, `--temperature T`, `--confidence`, `--quiet`.

stdin rules: if stdin is not a TTY and no `--state`, stdin is the state for
`decide/noul/score`; for `run` stdin is JSONL. Multiple `--question` on
`decide` with one state ⇒ shared mode automatically.

Output shape for `decide` (superset of Python):
```json
{"id":"…","choice":"billing","choice_index":1,"probabilities":[…],"option_ids":[…],
 "option_logits":[…],"confidence":0.42,"allowed_token_mass":0.99,"input_tokens":142,
 "forward_seconds":0.08,"total_seconds":0.09,"prompt_sha256":"…","prompt_version":"direct-options-v1",
 "model":{"id":"qwen3.5-4b","source":"bartowski/Qwen_Qwen3.5-4B-GGUF","revision":"…","file":"…","quant":"Q4_K_M","backend":"llama-cpp-2/<llama.cpp sha>"},
 "readout":"native full-vocabulary last-position logits restricted to declared answer slots",
 "probability_status":"conditional option score; uncalibrated as decision confidence"}
```

## 7. Tests & acceptance

- Unit (no model): Python-compatible JSON serialisation, prompt assembly,
  softmax/confidence, eval metrics against small hand cases, CLI arg parsing.
- Golden (downloads Qwen3-0.6B Q8_0, ~640 MB; behind `--features integration`
  or `OPENJEV_INTEGRATION=1`): for all 144 authored rows — sha256 and
  `input_tokens` identical to `browser-ladder-qwen3-0.6b.predictions.jsonl`;
  argmax agreement and logit deltas reported; shared vs direct logit parity.
- Eval: `openjev eval --fixture authored144 --model qwen3.5-4b` prints balanced
  accuracy; expect ≈ 0.81 (BF16 ref) minus a quantization gap — record actual.
- Perf: on the dev Mac (Metal) and on CPU-only, report direct latency per
  model and shared-mode speedup for 1 state × 21 questions (shape of
  `shape777`). Put numbers in `docs/RESULTS.md`.
- `cargo clippy -D warnings`, `cargo fmt --check`, `cargo test` green.

## 8. Open questions / risks (resolve early, in this order)

1. Does `llama-cpp-2` 0.1.157's llama.cpp build support **Qwen3.5** (hybrid
   GDN) and **MiniCPM5** architectures? Load each GGUF in a smoke test on day 1.
   If one fails: try building the sys crate against a newer llama.cpp
   (`LLAMA_CPP_PATH`/`dynamic-link`), or drop that model from the default list.
2. Chat template with `enable_thinking=false` through the crate (see 4.2).
3. Python-compatible `json.dumps` spacing/float formatting for structured state.
4. Shared-mode correctness on hybrid models (see 5).
5. Metal build flags / `n_gpu_layers` defaults on Apple Silicon; CUDA feature
   on Linux — CI matrix minimal (macOS arm64 + Linux x86 CPU).
6. Licensing: MIT for our code; keep `reference/semif-py/LICENSE` and cite in
   `THIRD_PARTY.md` (prompt strings + fixtures are copied from SemIf, MIT).
   Add TypeSafe non-affiliation note like upstream.
7. Later HTTP extension: Jev **wire** compatibility is a bounded subset, not
   hosted prediction/calibration/context parity. Floating-point structured inputs
   remain unsupported; options are limited to 16, questions to 64, bodies to
   1 MiB, and repeated-state expansion to 4 MiB. Residency avoids reloading
   weights but does not itself eliminate serial inference. Existing shared/batch
   profiles remain disabled until their unchanged numerical gates pass.

### M1 implementation status (reviewed and approved)

- [x] Three-crate workspace, exact direct dependency pins and committed-lock
  candidate; default checks do not activate llama.cpp, hf-hub, CMake, or model
  access.
- [x] Validated Decision/StateValue, strict duplicate-key raw JSON ingestion,
  insertion-order integer-only Python serialization, restricted profile
  rendering, typed readout metadata/schema, generic slot verification, f64
  numerics, Choice/Noul/Score adapters, and the documented eval subset.
- [x] Offline tests cover 144/144 authored and 108/108 perturbation text hashes,
  Python stdlib serializer/evaluator differentials, invalid/missing/tie hand
  cases, schema parity, and JSON-only CLI help/version/error behavior.
- [x] MIT `LICENSE`, SemIf/algorithm credit in `THIRD_PARTY.md`, and a credited
  small prompt oracle with regeneration instructions.

M1 issues discovered/resolved: `serde_json::Value` cannot reveal duplicate keys
or original number spelling once parsed, so public source-JSON ingestion captures
`RawValue` text and sends every string/slice/reader route through the strict
order-preserving parser; library-built Values remain recursively revalidated.
Finite-output serializers are explicit so nonfinite readout data errors rather
than becoming JSON `null`.
The optional native dependency graph is present in `Cargo.lock` but absent from
default compilation; native/backend behavior remained deliberately unimplemented
until M2.

The first Astra M1 gate reproduced five blockers: unbounded recursive raw JSON,
duplicate-key loss through public Serde routes, incomplete readout validation,
catastrophic cancellation in allowed-token mass, and raw-logit/probability tie
inconsistency. A follow-up found that the first Serde visitor fix lost lexical
integer `-0`; it is now replaced by one boxed-RawValue-to-strict-parser route
without admitting floating negative zero. All findings have cross-route
regressions described in `docs/PROGRESS.md`. Subsequent ingestion-route and
reserved-key regressions were also fixed; final parent/Astra review approved M1
with 54 workspace tests passing. The fixes do not enter M2/M7 algorithm scope.

### M2 implementation status (reviewed and approved)

- [x] Added `manifests/models.json` with the three full GGUF/native commits, filenames, exact sizes/SHA-256 values, quant/profile, tokenizer hash, and pinned native template hash.
- [x] Added canonical cache root precedence, hf-hub 1.0 blocking downloads at exact commits, process-safe per-artifact locking, download reuse, verified atomic receipts, full size/SHA checks at cache resolution and every worker load, offline cache-only behavior, and explicit corruption/repair policy.
- [x] Added one-owner-thread native loading and a direct smoke helper with scoped borrowing contexts, no workspace unsafe, `AddBos::Never`, nondeprecated byte-piece decoding, complete slot/boundary checks, chunk-local logits indexing, immediate raw-logit copy, and f64 core readout.
- [x] Built separate Metal (`GGML_METAL=ON`, all layers) and true CPU (`GGML_METAL=OFF`, zero layers and offload flags disabled) targets; inspected their real CMake caches and runtime device/native logs.
- [x] Observed each exact artifact transfer into `~/.cache/openjev`, then loaded/warmed/scored all three one at a time on Metal and offline CPU. Retained offline captures prove verified reuse; they are not independent transport-count evidence. All six readouts were finite. Evidence is under `docs/results/` and summarized in `docs/RESULTS.md`.
- [x] Fixed targeted cache review findings: portable safe repo/file validation before filesystem work; owned symlink containment; canonical regular-file import from HF relative snapshots; dangling destination handling; explicit owned-byte quarantine and snapshot rebuild; corrupt external bypass without modification; offline-repair miss behavior; failed-repair receipt invalidation; atomic Unix receipt replacement; and a true two-process one-fetch regression using tiny files only.
- [x] Adjudicated Qwen3-0.6B's nonidentical GGUF `57f1fd00…d0361` and native `a55ee1b1…74d8` templates as `reviewed-equivalent` only for two string system/user messages, no tools, generation prompt enabled, and thinking disabled. The manifest record is also keyed to artifact `9465e63a…031`; unseen hashes fail. Credited fixture templates, integrity tests, and the Jinja 3.1.4 oracle record 144 + 108 rows, all 252 reference prompt hashes, and four edge states without claiming broader equivalence.
- [x] Added create-only remediation captures for all three models on offline Metal and true CPU. All six rows pass with verified cache hits; original mismatch captures remain unchanged. `gpu_layers_actual=null` remains the honest safe-wrapper limitation.
- [x] Parent/Astra independently verified targeted fixes, all six passing smoke captures, and final gates (76 workspace tests) before the M2 commit.

Section 8 outcomes: architecture loading succeeded for all three (`qwen3`, `llama`, `qwen35`) on the pinned 0.1.156 source, so no unsupported `LLAMA_CPP_PATH`, dynamic-link, fork, mistralrs, or model-removal workaround was attempted. MiniCPM5 and Qwen3.5 are exact-template matches; Qwen3 is narrowly reviewed-equivalent, not exact. CPU/Metal flags and observed runtime evidence are recorded rather than inferred. Hybrid shared correctness remains M5 and no shared/batch/permutation implementation was added in M2.

The sentence above records M2's boundary at its commit. M3 has since been implemented locally as described below; no M4+ claim is implied.

### M3 implementation status (reviewed and approved)

- [x] Added production owner-thread `EngineHandle::score_direct` returning the full validated core `Readout`, with one clean prefill per decision, chunk-local final-logit retrieval, f64 postprocessing, no per-row warmup, no generation, no fallback, and mode-accurate timings/metadata. Direct readout `cache_hit` is now always `false` because every call creates a fresh context/full prefill; download-cache status remains separate cache/runner metadata. A model-free unit regression covers repeated direct metadata for a cached artifact, and the M3 runner now rejects any direct row that does not report `false`.
- [x] Reran offline fmt/check, workspace clippy/tests (78 passed), default-member clippy/tests (54 passed), and existing Metal/true-CPU native clippy/tests (26 unit tests each); diff check and unchanged-reference check passed. No model was resolved or benchmark rerun.
- [x] Preserved the M2 two-pass smoke path while exposing opt-in encoded prompt/reference validation for the integration gate. Generic all-slot/token-piece/collision/vocabulary/append-boundary verification remains mandatory.
- [x] Changed normative `ExecutionMetadata.gpu_layers_actual` to integer-or-null and added `gpu_layers_status`. CPU zero is reported only with offload disabled; Metal/CUDA unknown remains explicit null with retained native stderr, never inferred from requested layers. Added schema/code tests for explicit null and contradictory metadata.
- [x] Added production template status/equivalence evidence. Exact and the one artifact-keyed reviewed Qwen triple score; unseen/missing/mismatched templates are refused before decode. No equivalence scope was widened.
- [x] Added negative regressions for actual altered spacing, inserted BOS, and absolute-vs-chunk-local logits indexing without corrupting production.
- [x] Ran the create-only offline Metal gate from the existing Qwen cache/build. Mandatory authored144 passed 144/144 exact prompt SHA-256, input token count, ordered option IDs, answer token IDs, all-slot boundaries, and finite readouts. Extended perturbations108 separately passed 108/108 exact fields.
- [x] Retained full production row JSONL, one JSON stdout summary, reduced native stderr with original capture hash/size, and a report containing fixture/reference/output hashes, sizes, model/config, exact counts, numerical deltas, mismatch IDs, and reference/local margins.
- [x] Astra accepted the measured pinned-backend/Q8_0 versus native-BF16 baseline: authored first-argmax 140/144 (0.972222), logit MAE/RMSE/max 0.534313/0.664228/2.543209, probability MAE/RMSE/max 0.023046/0.068618/0.513266; extended first-argmax 107/108. This is measured baseline evidence, not numerical equivalence and not proof of quantization-only causation; no 98% gate or other guessed tolerance applies.
- [x] Parent/Astra approved corrected inference `cache_hit=false` semantics and final checks (78 workspace tests) for the M3 commit. The original 252 create-only rows stay unchanged, with their historical `cache_hit=true` defect explicitly annotated; their raw numerical fields remain the accepted baseline.

Section 8 status after M3: architecture support, restricted templates, serialization, accelerator truth, and licensing decisions remain closed with runtime evidence. The safe-wrapper layer-count gap is explicitly nullable/statused rather than guessed. Shared/hybrid copy correctness is intentionally still an M5 runtime gate, not a hidden M3 question; CUDA remains build-instruction scope until its later device gate. There are no other unresolved M3 design questions.

### M4 implementation status (reviewed, approved, and committed)

- [x] Production `decide`, `noul`, `score`, `ask`, `run`, and `models list|pull|path` execution, with thin primitive adapters over owner-thread direct scoring.
- [x] Exact text/file/stdin versus explicit structured-state ingestion; no trimming/guessing; explicit state never reads stdin; TTY absence errors.
- [x] Complete pre-load validation, duplicate-ID rejection, exact serialized shared-state comparison, ordered JSONL continuation on runtime row failures, stable exit 0/1/2 behavior, and create-only output summaries. Run rows are written and flushed before the next inference; sink failure stops later scoring and still shuts down. Output files are reserved before model startup, with documented empty/partial-file failure policy and no premature summary.
- [x] Visible serial full-prompt fallback for M4 shared/batch requests, including requested/effective modes, fallback reason and warning even under `--quiet`; `--require-shared` fails rather than pretending M5 exists.
- [x] Noul `p_yes`, finite Score level values/expectation/argmax, opt-in labelled normalized-margin confidence, and explicit rejection of unavailable M7 transforms.
- [x] Honest registered/custom model identity. Local files are hashed in place and may be `local-unverified`; custom Hub commits require caller SHA-256; all custom artifacts require an explicit profile and report `override-unverified` with no native-reference claim. Registered/custom Hub downloads share canonical mutation-parent preflight, and postfetch custom paths must remain owned regular files.
- [x] JSON-only help/version/results/model-path/write-summary envelopes, clap-tree-derived nested help metadata, command schema export, stderr-only native/progress/error logs, clean broken-pipe behavior, process tests, deterministic test-only scoring injection, and opt-in cached Qwen native CLI tests.
- [x] Parent/Astra code/evidence gate approved after targeted remediation; 97 workspace tests passed, native process capture 4/4. Ready for M4 milestone commit.

M4 intentionally does not implement KV prefix copy, independent sequence packing, probe eligibility, eval/bench, calibration, temperature scaling, or permutation averaging. Those remain M5–M7 at the M4 commit boundary.

### M5 implementation status (reviewed and approved; all measured profiles disabled)

- [x] Added immutable state-prefix prefill, full sequence copy into clean branches, ragged suffix scheduling with chunk-local output indices, checked branch clears and bounded wave reuse; no hybrid rollback or partial-prefix approximation.
- [x] Added independent full-prompt packed batching with distinct sequence IDs, bounded waves and stable input ordering.
- [x] Added exact configuration fingerprints and atomic process-contained probe receipts. Validation rejects malformed, failed, forged, reordered, incomplete, stale and nonmatching receipts; receipt data cannot relax the frozen `1e-3` logit / `1e-4` probability / identical-first-argmax gates.
- [x] Added whole-group fresh-context serial fallback after receipt absence/failure or runtime group failure, visible requested/effective metadata and stderr warnings even under `--quiet`; `--require-shared` fails instead of falling back.
- [x] Added model-free coverage for prefix construction, bounds, empty inputs, ragged scheduling, chunk-local indices, waves, copy/clear planning, receipt containment and forced child abort. Reprobe lifecycle tests now prove exact-key suspension before launch, durable fail-closed crash/malformed/nonzero/publication-failure transitions, parent-owned identity validation, and successful replacement. Native feature tests cover the compiled scheduler/receipt paths without claiming environment-gated model inference.
- [x] Ran all twelve create-only finalized probes for three exact models × Metal/true CPU × shared/batch from the existing cache and target directories. Every configuration failed the frozen numerical gate with unchanged first argmax, so none is enabled. The 21-way, changed-state and repeated-cycle cases are explicitly unrun after decisive failure, not passed. Astra accepted this as the intended serial-only user fallback with no numerical blocker and no tolerance relaxation.
- [x] Reran the cached Qwen release Metal native CLI process gate (5/5), including failed-receipt serial metadata, warning despite `--quiet`, `--require-shared`, and durable revocation after a probe-child crash with a synthetic pass confined to an isolated temporary cache. Reran the release Metal authored144 exact encoding gate (144/144).
- [x] Final fmt/check, workspace/default clippy/tests, Metal and true-CPU feature clippy/tests, diff/reference checks, exact counts and commands are recorded in `docs/PROGRESS.md`; no M6/M7 implementation, reference edit, model download, new native target, or commit was made.

No M5 shared/batch speedup is claimed. The large true-CPU shared deltas were inspected against direct/shared token, position, context, device and offload settings plus the pinned native source. No concrete implementation mismatch was found; decode-shape-dependent quantized kernels are a plausible but unproven explanation. Safe serial fallback is the reviewed supported outcome. Parent/Astra approved the lifecycle fix and 107-test workspace gate for the M5 commit; M6 must not claim shared speedup for these disabled configurations.

## 9. Later (v2+)

- `System1` trait crate shared with `gliner2-rs` (`choice/noul/score` →
  distribution), so callers can swap a 194M encoder for a 4B decoder per
  question.
- `--serve` HTTP/JSON residency is now implemented as a bounded Jev-compatible subset; a cross-request warm state-prefix cache is still future work. `--watch` REPL keeping one state prefilled remains future work.
- Reranker mode (`Qwen3-Reranker-4B` yes/no log-odds per option).
- Larger option cardinality (two-char slots), shortlist-then-choose for >16.
- Conformal / temperature calibration tooling with reliability diagrams.
- mistral.rs backend; WASM is out of scope (the site already covers it).

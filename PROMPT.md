# Agent prompt — plan and implement openjev-rs

Copy everything below the line into a fresh agent session started in
`~/projects/openjev-rs`.

---

You are building **openjev-rs**, a Rust port of the openjev.com / SemIf
"decision model" idea: read next-token option logits from a frozen open LLM
(GGUF via llama.cpp) in one forward pass and return Jev-style typed decisions
(`Choice` / `Noul` / `Score`) as JSON. It ships as a workspace of crates plus an
`openjev` CLI that takes state/question/options from args or stdin and prints
JSON to stdout.

Working directory: `~/projects/openjev-rs` (git repo, no Rust code yet).

## Read first, in this order
1. `todo.md` — the full brief: ground truth about the upstream project, goals,
   proposed layout, semantics that must match byte-for-byte, backend design,
   CLI spec, tests, ranked open questions. Treat it as the requirements doc.
2. `reference/semif-py/src/{core.py,direct.py,shared.py,serial.py,cli.py}` —
   the upstream Python implementation we mirror (prompt, slot checks, readout,
   shared-state KV reuse, output fields).
3. `reference/semif-py/manifests/models.json` (pinned model artifacts),
   `reference/semif-py/docs/METHOD.md`, `browser-model-ladder.json`,
   `browser-ladder-qwen3-0.6b.predictions.jsonl` (golden `prompt_sha256`,
   `input_tokens`, `option_logits` per authored144 row),
   `benchmarks/data/*.jsonl` (eval fixtures), `benchmarks/evaluate.py`.
4. `reference/semif-py/webgpu-demo/worker.js` — the browser variant (context
   only; we follow the Python prompt, not this one).
5. `reference/gliner2-rs-notes/jev-and-gliner.md` §5 — why this exists and how
   it will later sit beside `gliner2-rs` behind a `System1` trait.

## Model roles
- Use **GPT Astra 6** for planning, design review, and adjudicating open
  questions: produce `docs/PLAN.md` before any code, and re-review at each gate.
- Use **GPT 5.6 Sol** for implementation, tests, and benchmarks, working
  milestone by milestone against the plan.
- Run planning and implementation as separate subagents where the harness
  supports it; the implementer must receive the plan file, not a summary.

## Phase A — plan (Astra)
Produce `docs/PLAN.md` containing:
- Confirmed crate/API versions: check crates.io / the `utilityai/llama-cpp-rs`
  repo for `llama-cpp-2` (brief assumes 0.1.157, llama.cpp `e79e4bf6`) and
  verify the exact call signatures for: chat-template rendering (and whether
  `enable_thinking=false` / `chat_template_kwargs` can be passed), token↔str,
  `LlamaBatch` per-token logits + seq ids, `decode`, `get_logits_ith`,
  `copy_kv_cache_seq`, `clear_kv_cache_seq`, Metal/CUDA features.
- Decisions on the open questions in `todo.md` §8, with fallbacks.
- Final workspace layout, public types, CLI grammar, JSON output schema
  (must be a superset of the Python row schema, same field names).
- Milestones with acceptance checks (below), each small enough to land as one
  commit series.
Do not start Phase B until `docs/PLAN.md` exists and answers every §8 item.

## Phase B — implement (Sol), milestone gates
M1 **Skeleton + core**: workspace, `openjev-core` with types, Python-compatible
   JSON serialisation (`json.dumps` spacing/ordering/`ensure_ascii=False`),
   prompt builder for `direct-options-v1`, slot letters, softmax/confidence,
   eval metrics. Unit tests. `cargo clippy -D warnings` clean.
M2 **Backend smoke**: `openjev-llama` loads all three pinned GGUFs (download via
   hf-hub with revision pin into the cache dir); log architecture, n_ctx,
   template presence. If any model fails to load, record it in `docs/RESULTS.md`
   and continue with the ones that work — do not stall.
M3 **Direct readout parity**: `score_direct`. Golden test on Qwen3-0.6B Q8_0
   over authored144: `prompt_sha256` and `input_tokens` identical to the
   reference predictions for every row; report argmax agreement and logit
   deltas vs the BF16 reference. Fix prompt/template until hashes match — this
   is the gate, do not weaken it.
M4 **CLI v1**: `openjev decide|noul|score|ask|run|models` per `todo.md` §6;
   stdin handling; JSON to stdout, logs to stderr; exit codes. Include a
   `--help` that shows piped examples.
M5 **Shared-state mode**: `score_shared` via KV-cache sequence copy; assert
   logits match direct mode within tolerance on each model (hybrid Qwen3.5 is
   the risk — fall back to serial for models that fail and say so in output
   metadata). `decide` with several `--question` uses it automatically.
M6 **Eval + bench**: `openjev eval --fixture authored144|perturbations108`
   with balanced-accuracy math ported from `evaluate.py`; `openjev bench`
   direct vs shared on 1 state × 21 questions. Write measured numbers (this
   machine, Metal and CPU) into `docs/RESULTS.md` next to the published BF16
   reference; note the quantization gap honestly.
M7 **Polish**: `--permute N` and `--temperature T` (+ `calibrate`), README
   with install/usage/limitations (probabilities are conditional and
   uncalibrated; forced typed output can still be wrong), `THIRD_PARTY.md`
   crediting SemIf (MIT) and the TypeSafe non-affiliation note, CI workflow
   (macOS arm64 + Linux CPU, unit tests only; integration tests behind
   `OPENJEV_INTEGRATION=1`).

## Rules
- Never print anything but JSON to stdout from the CLI. Everything else →
  stderr.
- Match Python field names and the honesty strings (`readout`,
  `probability_status`) exactly; add fields, don't rename.
- No silent fallbacks in slot verification; fail with a clear error.
- Every milestone ends with: `cargo fmt`, `cargo clippy --all-targets -D
  warnings`, `cargo test`, a short entry appended to `docs/PROGRESS.md`
  (what was done, what was measured, what's next), and a git commit.
- Prefer reading the real crate source in `~/.cargo/registry` over guessing
  APIs; when a llama-cpp-2 API is missing, check the sys crate before writing a
  workaround.
- Large downloads (up to ~3 GB) go to the cache dir only once; reuse them.
- Keep `todo.md` in sync: tick items, add discovered issues under §8.

Start with Phase A. When `docs/PLAN.md` is written, summarise the decisions on
§8 in a few lines, then proceed to M1 without waiting for confirmation.

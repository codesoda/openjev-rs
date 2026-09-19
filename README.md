# openjev-rs

Rust port of the [openjev.com / SemIf](https://github.com/TheoLeeCJ/openjev)
idea: Jev-style typed decisions (`Choice` / `Noul` / `Score`) read directly from
the next-token option logits of a frozen open LLM (Qwen3-0.6B, MiniCPM5-2B,
Qwen3.5-4B GGUF via llama.cpp), in one forward pass, with no generation.

- `todo.md` — the implementation brief (start here).
- `PROMPT.md` — prompt for an agent session to plan + implement this repo.
- `reference/semif-py/` — upstream Python/JS source, fixtures, published
  results (MIT, © TheoLeeCJ).
- `reference/gliner2-rs-notes/` — background on Jev, use cases, and how this
  sits beside `gliner2-rs`.

Independent project. Not affiliated with or endorsed by TypeSafe AI or SemIf.

## Current status

M1 provides the backend-neutral core, exact restricted prompt renderers, schema,
evaluation subset, feature-disabled backend boundary, and JSON-only CLI parser.
M2 adds the pinned three-model registry, verified canonical cache, owner-thread
native loader, and a small JSONL direct-smoke example. All three exact GGUFs
loaded and produced finite readouts on Metal and true CPU. MiniCPM5/Qwen3.5 have
exact template hashes. Qwen3's GGUF and native templates are nonidentical;
parent/Astra approved a manifest-keyed `reviewed-equivalent` status only for the
restricted two-string-message, no-tools, disabled-thinking profile. Unseen hash
triples remain failures. The public production scoring CLI remains M4, so
ordinary scoring commands still return structured `backend_unavailable` rather
than exposing the smoke harness.

Raw JSON input through the explicit parser or serde_json's string, slice, and
reader routes accepts at most 128 nested arrays/objects per complete document
and preserves source text for strict parsing. Thus lexical integer `-0`
normalizes to `0`, while float spellings, overflow, and duplicate keys are
rejected consistently. `StateValue::try_from(serde_json::Value)` is the bounded
path for untrusted already-built trees; it validates depth iteratively but
cannot recover discarded duplicate keys or a numeric lexeme normalized by the
producer. Upstream generic operations that recursively serialize an
arbitrary-depth `Value` first—including
`serde_json::from_value::<StateValue>` with the `raw_value` feature and
`Value::to_string()`—are outside this depth guarantee.

The core carries these limitations into future readouts:

- A forced typed output can still be semantically wrong.
- Softmax over allowed tokens is conditional on the supplied alternatives; it
  is not calibrated operational confidence.

See `docs/PLAN.md` for the reviewed milestone contract,
`docs/RESULTS.md` for M2 evidence, and `schemas/readout-v1.schema.json` for the
normative emitted-readout schema.

## M2 native smoke commands

Metal and CPU must use separate target directories because Apple CMake defaults
Metal independently of Cargo's default features:

```sh
# Accelerated build: all model layers requested on Metal.
OPENJEV_INTEGRATION=1 GGML_METAL=ON CARGO_TARGET_DIR=target-m2-metal \
  cargo run -p openjev-llama --features metal --example m2_smoke -- \
  --all --device metal --gpu-layers all

# True CPU build: no Metal backend and no GPU/KQV/op offload.
OPENJEV_INTEGRATION=1 GGML_METAL=OFF CARGO_TARGET_DIR=target-m2-cpu \
  cargo run -p openjev-llama --features native --example m2_smoke -- \
  --all --offline --device cpu --gpu-layers 0
```

Integration runs are explicit and may download only when `--offline` is absent.
Ordinary `cargo test --workspace` never downloads a model. The canonical cache
is `--cache-dir` (API/harness), then `$OPENJEV_HOME`, then
`~/.cache/openjev`; GGUFs remain outside the repository.

The opt-in Qwen template oracle requires Jinja2 3.1.4 and performs no network
access:

```sh
python3 scripts/verify_qwen_template_equivalence.py
```

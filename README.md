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
Production inference and model downloads are intentionally not implemented yet;
scoring commands return a structured `backend_unavailable` error. Raw JSON
input through the explicit parser or serde_json's string, slice, and reader
routes accepts at most 128 nested arrays/objects per complete document and
preserves source text for strict parsing. Thus lexical integer `-0` normalizes
to `0`, while float spellings, overflow, and duplicate keys are rejected
consistently. `StateValue::try_from(serde_json::Value)` is the bounded path for
untrusted already-built trees; it validates depth iteratively but cannot recover
discarded duplicate keys or a numeric lexeme normalized by the producer.
Upstream generic operations that recursively serialize an arbitrary-depth
`Value` first—including `serde_json::from_value::<StateValue>` with the
`raw_value` feature and `Value::to_string()`—are outside this depth guarantee.

The core carries these limitations into future readouts:

- A forced typed output can still be semantically wrong.
- Softmax over allowed tokens is conditional on the supplied alternatives; it
  is not calibrated operational confidence.

See `docs/PLAN.md` for the reviewed milestone contract and
`schemas/readout-v1.schema.json` for the normative emitted-readout schema.

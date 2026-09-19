# M1 prompt oracles

`prompt-oracles.json` is a small offline oracle for the restricted
`direct-options-v1` renderers. The rendering behavior is ported from SemIf
(MIT, Copyright (c) 2026 TheoLeeCJ) and checked against the pinned upstream chat
templates listed in `docs/PLAN.md` §5.2. The synthetic decision text in this
file was authored for openjev-rs; no model weights are included.

Regenerate after an explicitly reviewed prompt/template change:

1. Render each `decision` with `openjev_core::prepare_prompt` for Qwen3,
   Qwen3.5, and MiniCPM5.
2. Independently render the same two-message input using the pinned Jinja
   artifacts and `enable_thinking=false` as described in `docs/PLAN.md` §5.2.
3. Require byte equality, then compute lowercase SHA-256 over UTF-8 prompt
   bytes and update the render/hash fields.
4. Run the 144 authored and 108 perturbation hash tests; never update this
   oracle merely to make a mismatch pass.

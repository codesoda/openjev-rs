# Embedded M6 evaluation assets

These three JSONL files are copied byte-for-byte from the preserved SemIf
reference snapshot in `reference/semif-py/` and embedded with `include_str!` so
`openjev eval` does not depend on the current working directory:

- `authored144.jsonl`
- `perturbations108.jsonl`
- `browser-ladder-qwen3-0.6b.predictions.jsonl`

They are MIT-licensed SemIf material, Copyright (c) 2026 TheoLeeCJ. Their frozen
SHA-256 values are checked in `openjev_core::fixtures`; regeneration means
copying the same pinned reference bytes, never editing either copy to satisfy a
test. See `THIRD_PARTY.md` for the full notice.

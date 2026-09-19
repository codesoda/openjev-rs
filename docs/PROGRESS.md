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

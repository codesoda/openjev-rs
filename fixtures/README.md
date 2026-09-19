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

## Qwen3 template-equivalence fixtures

`templates/qwen3-gguf-57f1fd00.jinja` is the 4,100-byte
`tokenizer.chat_template` extracted from the exact registered
`Qwen/Qwen3-0.6B-GGUF` artifact. `templates/qwen3-native-a55ee1b1.jinja` is the
4,168-byte template from pinned `Qwen/Qwen3-0.6B` tokenizer metadata. They are
credited to their Qwen/Hugging Face model sources and retained here only as
small integrity-pinned test fixtures; model weights are not included.

The templates are not identical. `qwen-template-equivalence.json` records the
parent/Astra-approved equivalence only for openjev's restricted profile:
exactly two string system/user messages, no tools,
`add_generation_prompt=true`, and `enable_thinking=false`. It records 144
authored rows, 108 perturbations, all 252 reference prompt hashes, and four
edge states. It makes no tool, multimodal, assistant-reasoning, or arbitrary
multi-turn claim. Reproduce with Jinja2 3.1.4 (an opt-in Python dependency):

```sh
python3 scripts/verify_qwen_template_equivalence.py
```

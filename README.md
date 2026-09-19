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

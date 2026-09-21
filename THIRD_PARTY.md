# Third-party material

openjev-rs is an independent implementation inspired by
[TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev), now named SemIf.
The exact direct prompt string, prompt/payload behavior, fixture-based test
oracles, and portions of the validation, numeric, and evaluation algorithms
were copied or ported from that project under its MIT license:

> Copyright (c) 2026 TheoLeeCJ
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The preserved upstream license is also at `reference/semif-py/LICENSE`.
For installed-binary M6 evaluation, byte-identical credited copies of SemIf's
`authored144.jsonl`, `perturbations108.jsonl`, and Qwen3-0.6B browser-ladder
prediction rows are embedded from `crates/openjev-core/assets/`; their frozen
hashes and provenance are recorded in that directory's README. The project-owned
benchmark fixtures under `fixtures/bench/` are original synthetic material, not
copied SemIf or hidden Jev data.

Other dependencies and external sources retain their own terms. This
repository distributes libraries only; a downstream binary distribution (for
example SystemOne's `s1`) is responsible for shipping complete license notice
bundles for its own pinned dependency graph and embedded native sources.
`THIRD_PARTY.md` is an attribution overview, not a substitute for such notices.

- [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT, pinned indirectly by
  `llama-cpp-2` to commit `e79e4bf660e19f2ad851e06c6913f7a8c5852621`.
  Its bundled llamafile SGEMM, cpp-httplib, nlohmann/json, base64, subprocess,
  and adapted ggml CPU/Metal sources carry their own notices, which binary
  distributions must preserve.
- [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs),
  MIT OR Apache-2.0; registry crates are pinned to 0.1.156.
- [hf-hub](https://github.com/huggingface/hf-hub), Apache-2.0, pinned to 1.0.0.
  Its pinned hf-xet dependency closure includes MPL-2.0-only
  [colored 3.1.1](https://github.com/mackwic/colored) and
  [option-ext 0.2.0](https://github.com/soc/option-ext); binary distributions
  must retain their full MPL-2.0 terms and source links. This is an explicit
  release-review item, not a general copyleft exception.
- The Rustls path uses statically linked
  [AWS-LC](https://github.com/aws/aws-lc-rs) through `aws-lc-sys` 0.45.0.
  Its compound license expression covers compiled-in BoringSSL/OpenSSL, Fiat,
  s2n-bignum, Jitter Entropy, and public-domain notices; AWS-LC expressly
  elects BSD-3-Clause, not GPL-2.0, for Jitter Entropy.
- The former Jev HTTP adapter and its SDK compatibility smoke (checked against
  [typesafe-ai/typesafe-sdk-js](https://github.com/typesafe-ai/typesafe-sdk-js)
  commit `66880ccded6cb642dc1809620c2b108c33730214`, npm 0.6.0, MIT, and Vercel
  AI commit `20dd00abba618d5a516e0fee40ccd3e18a2bd1fb`, Apache-2.0) now live in
  [SystemOne](https://github.com/codesoda/systemone).
- Qwen and MiniCPM tokenizer/model sources listed in `docs/PLAN.md` remain
  external and subject to their model cards and licenses. Two small Qwen3 chat
  template fixtures extracted from the exact registered GGUF and pinned native
  tokenizer metadata are retained under `fixtures/templates/` solely for the
  credited, integrity-pinned restricted-profile equivalence oracle.

## MPL-2.0 covered-source availability

The pinned `hf-hub` closure includes MPL-2.0 crates (`colored 3.1.1`,
`option-ext 0.2.0`). Any binary distribution built from these libraries must
ship the complete, unmodified crates.io source archives for the exact versions
in its `Cargo.lock`, together with the MPL-2.0 terms, and hash-pin them in its
release metadata. The previous release tooling that did this for the standalone
`openjev` binary was removed with the CLI; SystemOne owns that obligation for
`s1` releases.

No model weights are distributed or covered by the openjev-rs MIT license.
The material under `reference/` remains attributed to its upstream source.
openjev-rs is not affiliated with or endorsed by SemIf, TheoLeeCJ, TypeSafe,
or Jev. Names and marks belong to their respective owners.

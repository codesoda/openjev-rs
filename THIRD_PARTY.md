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

Other dependencies and external sources retain their own terms. Distributed
binary archives include `THIRD_PARTY_LICENSES.html`, which contains attributable
license/copyright text for both pinned Rust target graphs and for the native
sources embedded by llama.cpp/ggml. Archives separately include the unmodified
official Rust 1.95.0 compiler-payload `RUST-COPYRIGHT-library.html` for standard
library and compiler-builtins runtime attribution. `THIRD_PARTY.md` is an
attribution overview, not a substitute for those notice files.

- [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT, pinned indirectly by
  `llama-cpp-2` to commit `e79e4bf660e19f2ad851e06c6913f7a8c5852621`.
  The generated bundle also preserves the exact notices for its bundled
  llamafile SGEMM, cpp-httplib, nlohmann/json, base64, subprocess, and adapted
  ggml CPU/Metal code used by these release configurations.
- [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs),
  MIT OR Apache-2.0; registry crates are pinned to 0.1.156.
- [hf-hub](https://github.com/huggingface/hf-hub), Apache-2.0, pinned to 1.0.0.
  Its pinned hf-xet dependency closure includes MPL-2.0-only
  [colored 3.1.1](https://github.com/mackwic/colored) and
  [option-ext 0.2.0](https://github.com/soc/option-ext). Their full MPL-2.0
  terms, source links, and target applicability are retained in the generated
  bundle. This is an explicit release-review item, not a general copyleft
  exception.
- The Rustls path uses statically linked
  [AWS-LC](https://github.com/aws/aws-lc-rs) through `aws-lc-sys` 0.45.0.
  cargo-about cannot recover every copyright holder from its compound license
  expression, so the bundle additionally preserves the crate's exact aggregate
  license file, including its compiled-in BoringSSL/OpenSSL, Fiat,
  s2n-bignum, Jitter Entropy, and public-domain notices. AWS-LC expressly
  elects BSD-3-Clause, not GPL-2.0, for Jitter Entropy.
- [axum](https://github.com/tokio-rs/axum), MIT, pinned to 0.8.9, and
  [Tokio](https://github.com/tokio-rs/tokio), MIT, pinned to 1.53.1, provide
  the resident HTTP frontend/runtime.
- The Jev wire adapter and reproducible compatibility smoke were checked against
  [typesafe-ai/typesafe-sdk-js](https://github.com/typesafe-ai/typesafe-sdk-js)
  commit `66880ccded6cb642dc1809620c2b108c33730214`, npm package 0.6.0 (MIT).
  Two-decimal wire projection was additionally checked against Vercel AI commit
  `20dd00abba618d5a516e0fee40ccd3e18a2bd1fb`, files
  `packages/typesafe-ai/src/typesafe-ai-evaluation-model.ts` and
  `typesafe-ai-evaluation-api.ts` (Apache-2.0).
- Qwen and MiniCPM tokenizer/model sources listed in `docs/PLAN.md` remain
  external and subject to their model cards and licenses. Two small Qwen3 chat
  template fixtures extracted from the exact registered GGUF and pinned native
  tokenizer metadata are retained under `fixtures/templates/` solely for the
  credited, integrity-pinned restricted-profile equivalence oracle.

## MPL-2.0 covered-source availability

Each binary release includes `colored-3.1.1.crate` and
`option-ext-0.2.0.crate` at the archive root. These are the complete,
unmodified original crates.io source archives for the exact dependency
versions in `Cargo.lock`, not reconstructed source trees. Their archive
SHA-256 values are checked against the corresponding crates.io checksums in
`Cargo.lock`, recorded in `licenses/THIRD_PARTY_LICENSES.metadata.json`, and
included in `BUILD-INFO.json` with the other packaged-file hashes.

Recipients may extract, use, modify, and redistribute those covered-source
archives under the MPL-2.0 terms included inside each archive and reproduced in
`THIRD_PARTY_LICENSES.html`. Those terms apply to their covered files; the
larger openjev-rs program remains under its stated MIT license. This
source-availability provision is specific to these reviewed dependencies and
is not a blanket copyleft exception.

Regenerate the checked-in bundle with cargo-about 0.9.2 and the exact commands
recorded in the bundle:

```sh
python3 scripts/generate_third_party_licenses.py
python3 scripts/check_third_party_licenses.py
```

`about.toml` records license resolution and exact bindings-license
clarifications. `licenses/THIRD_PARTY_LICENSES.metadata.json` pins the
`Cargo.lock` hash, bundle hash, target/features, native commit, exact native
source hashes, the two covered-source archive identities and hashes, and the
Rust notice's SHA-256 plus `rustc -vV` release/commit provenance. Regeneration
copies that notice from the matching installed compiler after checking its
`rustc` component manifest; it does not synthesize notice text. CI runs the
network-free checker and fails if the lock, bundle, metadata, Rust notice,
workflow toolchain pin, or covered-source archive bytes disagree; generation
itself may fetch license text from Cargo/cargo-about sources and the hash-pinned
upstream URLs.

No model weights are distributed or covered by the openjev-rs MIT license.
The material under `reference/` remains attributed to its upstream source.
openjev-rs is not affiliated with or endorsed by SemIf, TheoLeeCJ, TypeSafe,
or Jev. Names and marks belong to their respective owners.

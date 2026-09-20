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

M1 provides the backend-neutral core, exact restricted prompt renderers and
schemas. M2 adds the pinned three-model registry, verified canonical cache and
owner-thread native loader. M3 adds the complete production direct `Readout`
and passed the strict cached Qwen3 authored144 plus perturbations108 prompt,
token and slot gates.

M4 exposes that production path through `openjev decide`, `noul`, `score`,
`ask`, `run`, and `models list|pull|path`. M5 adds exact shared-prefix KV copy,
independent packed batching, configuration-bound subprocess probes and local
eligibility receipts. Input is fully validated before model load; stdout is
JSON/JSONL only; native/progress/warning/error logs use stderr; file output is
create-only. `run` writes and flushes each success or ErrorRecord before scoring
the next row instead of retaining the run in memory. Noul and Score are
transparent direct-Choice adapters, and opt-in confidence is the labelled
uncalibrated normalized margin. A backend-disabled build parses and validates
inputs but returns structured `backend_unavailable` rather than fake
probabilities.

The user-requested resident HTTP extension adds `openjev --serve`: one verified
model is loaded and warmed once, then a bounded Jev-shaped API is served at
`/v1/systemone`. The async HTTP frontend never makes the non-Send scorer shared;
one dedicated synchronous owner thread retains it. See
[`docs/SERVE.md`](docs/SERVE.md) for auth, deadlines, cancellation limits, wire
mapping, conditional-probability disclosure, and the pinned official SDK smoke.

Native shared/batch execution is never enabled merely because it compiled. A
passing local receipt for the exact artifact, native pin, probe-suite version,
device/offload, threads, context/batch/sequence settings and prompt profile is
required. The
retained Metal and true-CPU probes for all three pinned profiles failed the
frozen numerical gates, so those twelve tested configurations intentionally
remain on fresh serial full-prompt scoring. Requested/effective mode, the
nonempty receipt failure, and `cache_hit=false` make that fallback visible;
stderr warns even with `--quiet`. `--require-shared` fails before inference when
no matching passing receipt exists. Eval/bench remain explicit M6
not-implemented surfaces, and calibration/permutation/nondefault temperature
remain M7.

MiniCPM5/Qwen3.5 have exact template hashes. Qwen3's GGUF and native templates
are nonidentical; parent/Astra approved a manifest-keyed `reviewed-equivalent`
status only for the restricted two-string-message, no-tools, disabled-thinking
profile. Unseen registered hash triples remain failures. Custom local/Hub GGUFs
require an explicit template profile and are labelled `override-unverified`,
with no fabricated native reference or golden claim.

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
`docs/RESULTS.md` for runtime evidence, `schemas/readout-v1.schema.json` for the
normative emitted-readout schema, and `schemas/commands-v1.schema.json` for M4
command envelopes.

## Tagged binary releases

The repository now contains a tag-triggered GitHub Actions pipeline for Apple
Silicon macOS 14+ (Metal) and x86-64 Ubuntu/glibc 2.35+ (CPU). A tag must be
exactly `v` plus the Cargo workspace version. Both native jobs must pass before
a GitHub Release is created; branch, pull-request, and manual runs upload only
ordinary workflow artifacts.

Each release archive contains the executable, the project license/attribution
file, a complete `THIRD_PARTY_LICENSES.html` dependency notice bundle, the
official Rust 1.95.0 library/runtime `RUST-COPYRIGHT-library.html` notices, the
complete unmodified `colored-3.1.1.crate` and `option-ext-0.2.0.crate`
MPL-2.0 covered-source archives, runtime documentation, and `BUILD-INFO.json`;
model weights and cache data are never packaged. The release also has
`SHA256SUMS`. CI executes the extracted
binary from a temporary directory outside the checkout and checks that native
dynamic dependencies resolve only to operating-system libraries. The macOS
binary embeds its Metal library, but is not Developer ID signed or Apple
notarized. Running either packaged binary requires no Python, CMake, compiler,
Xcode, or Homebrew.

See [`docs/RELEASE.md`](docs/RELEASE.md) for targets, system requirements,
checksum verification, package contents, and user-local installation. Release
publication and a real downloaded-artifact Metal HTTP/official-SDK smoke remain
separate gates: packaging CI alone is not evidence that the downloaded release
passed the latter.

## Build and install the M5 CLI

The ordinary workspace build deliberately excludes llama.cpp:

```sh
cargo build
# Parsing/help/models-list work; scoring returns backend_unavailable.
```

Use the existing device-specific target directory for a native CLI. On Apple
Silicon, the Metal build defaults to Metal with all layers requested. A true
CPU build defaults to CPU, requests zero GPU layers, and disables KQV/op
offload. Do not share one target directory between those native configurations.

```sh
# Metal
GGML_METAL=ON CARGO_TARGET_DIR=target-m2-metal \
  cargo build --release -p openjev-cli --features metal
install -m 0755 target-m2-metal/release/openjev "$HOME/.local/bin/openjev"

# True CPU on macOS
GGML_METAL=OFF CARGO_TARGET_DIR=target-m2-cpu \
  cargo build --release -p openjev-cli --features native
```

The product default model remains `minicpm5-2b`. The examples use cached
`qwen3-0.6b` for fast local exercise:

```sh
# State from a flag; strings are not trimmed.
openjev --offline --model qwen3-0.6b decide \
  --state 'customer cannot sign in' \
  --question 'Which queue?' --option 'Account access' --option Billing

# State from stdin. Structured state requires --state-json/--state-json-file.
printf 'suspicious message\n' | openjev --offline --model qwen3-0.6b noul \
  --question 'Is this phishing?'

openjev --offline --model qwen3-0.6b score --state-json '{"severity": 3}' \
  --question 'How urgent?' --level low --level medium --level high \
  --level-value 0 --level-value 5 --level-value 10

printf '%s\n' \
  '{"id":"d1","state":"ticket","question":"Queue?","options":[{"id":"access","description":"Account access"},{"id":"billing","description":"Billing"}]}' \
  | openjev --offline --model qwen3-0.6b ask

openjev --offline --model qwen3-0.6b run \
  --input decisions.jsonl --output new-results.jsonl
openjev models list
openjev --offline models path qwen3-0.6b

# Resident loopback API. The official SDK baseURL is this root URL; it appends /v1.
openjev --serve --offline --model qwen3-0.6b --host 127.0.0.1 --port 8080
```

The resident service exposes `POST /v1/systemone`, `GET /v1/models`,
`GET /healthz`, and `GET /readyz`. Loopback may run without configured auth;
non-loopback binding requires `--api-key-env NAME`. Bodies are capped at 1 MiB,
admission at 16 jobs, and the whole-request deadline defaults to 120 seconds.
The model is loaded exactly once and native decode remains noninterruptible.
See [`docs/SERVE.md`](docs/SERVE.md) for the supported Jev subset and
`scripts/sdk-compat/` for the exact `@typesafe-ai/sdk` 0.6.0 smoke source.

Exactly one state source is accepted for `decide`/`noul`/`score`:
`--state`, `--state-file`, `--state-json`, `--state-json-file`, or non-TTY
stdin. Explicit state never reads stdin. A TTY without state is an error.
`ask` takes one complete Decision object; `run` takes JSONL, ignores blank
lines, preserves row order, emits and flushes per-row runtime errors and
continues, then exits 1 if any row failed. Add `--compact --quiet` for a small
LLM/script-oriented decision projection while suppressing routine native INFO
logs:

```sh
openjev --compact --quiet --offline --model qwen3-0.6b decide \
  --state 'customer cannot sign in' \
  --question 'Which queue?' --option Access --option Billing
```

Full `openjev-readout-v1` remains the default. Compact output applies only to
`decide`, `noul`, `score`, `ask`, and `run`; it retains semantic option arrays,
the exact probability honesty label, primitive-specific values, requested
confidence, and a small execution object only on fallback. `--quiet` still
retains WARN/ERROR and explicit fallback warnings. See
[`docs/COMPACT.md`](docs/COMPACT.md) and
[`schemas/compact-v1.schema.json`](schemas/compact-v1.schema.json).

Fatal parse/validation errors exit 2 before native loading. For `run --output`, the create-only destination is
reserved after complete input validation but before model startup. A startup
failure therefore leaves a new empty file; later output failure leaves the
already flushed prefix, stops further inference, shuts down the owner worker,
and emits no completed write summary. The summary is written only after every
row and file sync complete. `--pretty` is only for a single object and is
rejected for JSONL.

Pinned pulls use the canonical cache (`--cache-dir`, then `$OPENJEV_HOME`, then
`~/.cache/openjev`) and verify complete size plus SHA-256. Registered and custom
Hub downloads preflight canonical containment of the Hub lock, repository,
blob, snapshot, and negative-cache parents before hf-hub can mutate them, then
require the returned regular file to remain inside the owned Hub. `models path`
emits a JSON envelope, never a bare path. Examples:

```sh
openjev models pull qwen3-0.6b
openjev --offline models path qwen3-0.6b

# Local custom artifact: expected hash is optional; omission is honestly
# labelled local-unverified. The file is hashed in place and never moved.
openjev --model /models/custom.gguf --template-profile qwen3 decide ...

# Remote custom artifact: commit, expected hash, and profile are mandatory.
openjev --model 'hf:owner/repo@0123456789abcdef0123456789abcdef01234567:model.gguf' \
  --model-sha256 64-lowercase-hex --template-profile qwen3 decide ...
```

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

M5 probes run in a subprocess. Before launch, the parent establishes and locks
the exact receipt key and suspends any prior authorization. A crash, malformed
or nonzero passing report, or publication failure leaves that key suspended;
only a fully validated exact passing child result replaces eligibility. Failed
receipts may remain as diagnostics but are not eligible. Probe both modes
independently because eligibility is mode-specific:

```sh
openjev --offline --device metal models probe qwen3-0.6b --mode shared
openjev --offline --device metal models probe qwen3-0.6b --mode batch
```

A failed probe exits 1 with an `openjev-probe-report-v1` JSON object on stdout;
a passing probe exits 0. Standard scoring never silently reprobes or relaxes
the frozen `1e-3` slot-logit / `1e-4` probability / identical-first-argmax
gates. See `docs/RESULTS.md` and `docs/results/m5/` for the twelve finalized
`*-final.json` reports, preserved pre-final captures, and exact reproduction
commands.

Integration runs are explicit and may download only when `--offline` is absent.
Ordinary `cargo test --workspace` never downloads a model. The canonical cache
is `--cache-dir` (API/harness), then `$OPENJEV_HOME`, then
`~/.cache/openjev`; GGUFs remain outside the repository.

The opt-in Qwen template oracle requires Jinja2 3.1.4 and performs no network
access:

```sh
python3 scripts/verify_qwen_template_equivalence.py
```

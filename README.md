<a id="readme-top"></a>

# openjev-rs

**Rust libraries for local typed decisions from a frozen language model's
next-token option logits.**

`openjev-core` and `openjev-llama` implement the OpenJev / SemIf decision
readout: score the supplied alternatives with one forward pass of a pinned,
checksum-verified GGUF model through llama.cpp, without generating an answer or
chain of thought. This repository ships **libraries only**. The command-line
interface and Jev-compatible HTTP API that used to live here have moved to
[SystemOne](https://github.com/codesoda/systemone) (`s1`), which pins these
crates by Git revision and hosts them behind a backend-neutral interface next to
other decision models.

This is an independent implementation inspired by
[SemIf / openjev](https://github.com/TheoLeeCJ/openjev). It is not affiliated
with or endorsed by TypeSafe AI or SemIf.

## Crates

| Crate | Purpose | Native build |
| --- | --- | --- |
| `openjev-core` | Backend-neutral types (`Decision`, `Question`, `StateValue`, `Readout`), Python-parity prompt rendering, answer-slot verification, f64 numerics, Noul/Score adapters, evaluation math and embedded fixtures | No |
| `openjev-llama` | Pinned model registry and verified cache, probe receipts for shared/batch execution, and (behind `native`/`metal`/`cuda`) the owner-thread llama.cpp engine: `EngineHandle::{spawn_resolved, score_direct, score_shared, score_batch, shutdown}` | Optional |

Default builds compile neither llama.cpp nor Hugging Face access: the registry,
cache integrity and all parity/unit tests run offline. Enable `native` (CPU),
`metal` (Apple Silicon) or `cuda` on `openjev-llama` to load and score models.

## Use from another workspace

Pin a reviewed revision; do not depend on a moving branch:

```toml
[dependencies]
openjev-core = { git = "https://github.com/codesoda/openjev-rs.git", rev = "<commit>" }
openjev-llama = { git = "https://github.com/codesoda/openjev-rs.git", rev = "<commit>", default-features = false }

[features]
native = ["openjev-llama/native"]
metal = ["native", "openjev-llama/metal"]
```

Minimal direct scoring (requires a native feature and a cached model):

```rust,ignore
use openjev_core::{Decision, DecisionOption, StateValue};
use openjev_llama::{CacheOptions, EngineHandle, EngineOptions, ModelCache, ModelRegistry, ModelSpec};

let registry = ModelRegistry::bundled()?;
let cache = ModelCache::new(home.join(".cache/openjev"));
let resolved = openjev_llama::resolve_model_spec(
    &registry, &cache, &ModelSpec::RegistryId("qwen3-0.6b".into()),
    CacheOptions { offline: true, repair: false },
)?;
let (model, artifact) = resolved.into_parts();
let engine = EngineHandle::spawn_resolved(model, artifact, EngineOptions { /* device, threads, ... */ })?;
let readout = engine.score_direct(Decision::new(
    "d1",
    StateValue::string("The customer was charged twice.")?,
    "Which team should handle this?",
    vec![
        DecisionOption { id: "billing".into(), description: "Billing".into() },
        DecisionOption { id: "support".into(), description: "Support".into() },
    ],
)?)?;
engine.shutdown()?;
```

`Readout` carries the full vendor evidence (option logits, token IDs, prompt
hash, execution mode, device, template status) and the honesty strings
`readout` / `probability_status`. Probabilities are conditional on the supplied
alternatives and are **not** calibrated confidence.

For a ready-made CLI/HTTP server (`s1 serve`, `s1 run`, `s1 openjev models pull`,
`s1 openjev probe`) use SystemOne.

## Models

| Model ID | Quantization | Approx. download | Notes |
| --- | --- | --- | --- |
| `qwen3-0.6b` | Q8_0 | 0.64 GB | Smallest; reviewed-equivalent template |
| `minicpm5-2b` | Q4_K_M | 1.56 GB | Registry default |
| `qwen3.5-4b` | Q4_K_M | 3.01 GB | Strongest on recorded fixtures |

Artifacts are pinned to exact Hugging Face commits with sizes and SHA-256 in
[`manifests/models.json`](manifests/models.json) and verified at every load.
Custom local or Hub GGUFs require an explicit template profile and never acquire
a fabricated native reference.

## Guarantees and limitations

- Prompt bytes and token counts are strict gates against the Python reference
  (`authored144` 144/144, `perturbations108` 108/108 exact).
- No truncation, implicit BOS, full-vocabulary softmax or silent model
  substitution. Requested accelerators fail explicitly rather than falling back.
- Shared-prefix and independent-batch execution are enabled only for an exact
  model/build/configuration with a passing probe receipt; otherwise callers get
  an explicit serial full-prompt fallback. Every measured profile so far fails
  the frozen numerical gate, so no speedup is claimed.
- Structured state must be integer-only JSON; floating-point numbers are
  rejected until a reviewed Python-`repr` serializer exists.

## Documentation

| Document | Contents |
| --- | --- |
| [Plan](docs/PLAN.md) / [brief](todo.md) | Architecture, source adjudications, milestone history (historical; mentions the former CLI) |
| [Progress](docs/PROGRESS.md) | Implementation and measurement history |
| [Results](docs/RESULTS.md) | Parity, runtime and release-acceptance evidence |
| [Readout schema](schemas/readout-v1.schema.json) | Machine-readable `Readout` shape |
| [Third-party notices](THIRD_PARTY.md) | Upstream credits and licenses |

## Contributing

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
# Native (compiles llama.cpp):
cargo clippy --workspace --all-targets --features openjev-llama/native -- -D warnings
cargo test --workspace --features openjev-llama/native
```

Ordinary tests do not download or load models. Model integration tests are
opt-in with the `integration` feature and `OPENJEV_INTEGRATION=1`. Preserve
pinned reference semantics and do not weaken parity gates to enable an
optimization. Changes to the public library surface should be coordinated with
SystemOne's [cross-repo plan](https://github.com/codesoda/systemone/blob/main/docs/plans/cross-repo.md).

## License

Project code is MIT ([LICENSE](LICENSE)). Model weights have their own terms;
upstream credits are in [THIRD_PARTY.md](THIRD_PARTY.md).

## Acknowledgments

- [TheoLeeCJ / SemIf](https://github.com/TheoLeeCJ/openjev) for the upstream
  decision-readout approach, Python reference, and evaluation fixtures.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and
  [llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs) for local inference.
- The Qwen and MiniCPM teams, and the GGUF publishers listed in
  [the model manifest](manifests/models.json).

[Back to top](#readme-top)

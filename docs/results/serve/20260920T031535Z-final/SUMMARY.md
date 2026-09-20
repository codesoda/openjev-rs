# Final OpenJev native server and official SDK smoke

UTC run: 2026-09-20T03:15:35Z–2026-09-20T03:17:34Z  
Repository HEAD: `1dcfad3743c86dbeaccb12236ac2c1ea574a219b`  
Scope: final requested native/SDK smoke only. Standard workspace and feature gates are owned by the parent and were not run here. This evidence makes no runtime-performance or benchmark claim.

## Artifact identity

Rebuilt release binary:

- Path: `target-m2-metal/release/openjev`
- SHA-256: `476a5b9414a702e0e2c7cfa792dce24bdd724ed1a825cdb470071f2b3582dcbb`

Relevant source SHA-256 values, captured before and after testing and verified unchanged:

- `crates/openjev-cli/src/server/mod.rs`: `cdb21679e5c9595066cdccbb496348c00bae7721e8606d638f519f028459461f`
- `crates/openjev-cli/src/server/jev.rs`: `846bd2decb4bd1639d17681689fa1a2dd8a3d6bf12cbaf2d64fbc40429ca3285`
- `crates/openjev-cli/tests/native_server.rs`: `96cfd0111586bf912d092d472c3cecdc28b27faa47d01c8b8c733bd1cbe80efe`
- `scripts/sdk-compat/smoke.mjs`: `bef177cf903c7280e45528239e36485f16b5d2eb08870019b505fc7be4ef9687`

No native/model-heavy process was present at preflight. The cached Qwen3-0.6B GGUF was used offline; no GGUF download was requested.

## Gate 1: native server integration — PASS

```sh
OPENJEV_INTEGRATION=1 CARGO_TARGET_DIR=target-m2-metal GGML_METAL=ON \
  cargo test --release -p openjev-cli --features metal,integration \
  --test native_server -- --nocapture
```

- Exit code: 0
- Timeout: false (1200-second subprocess bound)
- Tests: 1 passed, 0 failed, 0 ignored, 0 measured, 0 filtered out
- On macOS with the `metal` feature, the test's compile-time device selection is Metal.
- Two repeated mixed Choice/Noul/Score requests matched while one resident process remained alive.
- The test sent SIGTERM, required successful shutdown, and required empty stdout.
- Two disclosed fresh-serial fallback warnings were observed and are not test failures.

## Gate 2: real authenticated Metal server and strengthened SDK — PASS

Server:

```sh
OPENJEV_SMOKE_API_KEY=<redacted> target-m2-metal/release/openjev \
  --serve --offline --model qwen3-0.6b --device metal \
  --host 127.0.0.1 --port 64459 --request-timeout-secs 120 \
  --api-key-env OPENJEV_SMOKE_API_KEY
```

SDK setup and smoke:

```sh
cd scripts/sdk-compat
npm ci --ignore-scripts --no-audit --no-fund
OPENJEV_BASE_URL=http://127.0.0.1:64459 OPENJEV_API_KEY=<redacted> npm run smoke
```

- Readiness: HTTP 200
- Runtime explicitly requested `--device metal`; stderr records MTL0 / Apple M3 Pro selection and one pinned Qwen3-0.6B model load.
- `npm ci`: exit 0, timeout false, 1 package installed
- Strengthened official `@typesafe-ai/sdk@0.6.0` smoke: exit 0, timeout false
- SDK source commit declared by smoke: `66880ccded6cb642dc1809620c2b108c33730214`
- Response model: `qwen3-0.6b`; typed answers: `route`, `review`, `urgency`; usage: 338 input, 0 output tokens
- Passing strengthened assertions covered model fields; Choice labels/probabilities/confidence; Noul range; Score legend/range/probabilities/confidence; finite two-decimal values; usage; unknown-model HTTP 404; and unsupported-float HTTP 422.
- Authenticated success: SDK model-list, inference, and validation calls completed with the configured bearer key.
- Wrong-key failure path: HTTP 401 JSON with `error_type: authentication_error`.

## Gate 3: lifecycle and cleanup — PASS

- Sent SIGTERM only to captured server PID 98455.
- Server exit code: 0
- Server stdout: 0 bytes
- Captured server PID verified absent afterward
- `scripts/sdk-compat/node_modules` removed and verified absent
- Relevant source hashes unchanged throughout testing

## Final failure counts

- Native integration failures: 0
- npm installation failures: 0
- Strengthened SDK assertion failures: 0
- Authentication assertion failures: 0
- Shutdown/stdout/cleanup failures: 0
- Source-hash verification failures: 0
- Total observed failures: 0

Raw commands, bounded stdout/stderr captures, HTTP response, environment, binary/source hashes, and structured results are alongside this summary. `SHA256SUMS` hashes every evidence file other than itself.

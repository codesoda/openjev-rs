# OpenJev native server and official SDK smoke

UTC run start: 2026-09-20T03:00:40Z  
Repository HEAD: `1dcfad3743c86dbeaccb12236ac2c1ea574a219b`  
Scope: testing evidence only; no source edits, commits, or pushes by this test owner.

## Preflight

- Checked native/model-related processes before starting. No OpenJev, llama, Qwen, or Metal model process was running.
- Two unrelated Cargo test processes were present; the requested smoke was allowed to run alongside unrelated activity, so these results are not timing benchmarks.
- Used the cached Qwen3-0.6B GGUF with `--offline`; no GGUF download was requested.

## Native integration test

Command:

```sh
OPENJEV_INTEGRATION=1 CARGO_TARGET_DIR=target-m2-metal GGML_METAL=ON \
  cargo test --release -p openjev-cli --features metal,integration \
  --test native_server -- --nocapture
```

Result:

- Exit code: 0
- Timeout: false (1200-second subprocess bound)
- Tests: 1 passed, 0 failed, 0 ignored, 0 measured, 0 filtered out
- The test made two equal mixed Choice/Noul/Score requests, observed a live resident process, sent SIGTERM, required exit success, and required empty stdout.
- Stderr records two disclosed fresh-serial fallbacks; these are expected observability, not test failures.

## Real server and official JavaScript SDK

Server command (secret redacted in evidence):

```sh
OPENJEV_SMOKE_API_KEY=<redacted> target-m2-metal/release/openjev \
  --serve --offline --model qwen3-0.6b --device metal \
  --host 127.0.0.1 --port 64012 --request-timeout-secs 120 \
  --api-key-env OPENJEV_SMOKE_API_KEY
```

Client commands:

```sh
cd scripts/sdk-compat
npm ci --ignore-scripts --no-audit --no-fund
OPENJEV_BASE_URL=http://127.0.0.1:64012 OPENJEV_API_KEY=<redacted> npm run smoke
```

Results:

- Readiness: HTTP 200
- `npm ci`: exit 0, 1 package installed, timeout false (300-second bound)
- Official `@typesafe-ai/sdk@0.6.0` smoke: exit 0, timeout false (300-second bound)
- SDK source commit declared by smoke: `66880ccded6cb642dc1809620c2b108c33730214`
- SDK response model: `qwen3-0.6b`
- Typed answers returned: 3 (`route`, `review`, `urgency`)
- Usage: 338 input tokens, 0 output tokens
- Authenticated success: SDK model-list and inference calls completed with the configured bearer key.
- Authenticated failure: wrong bearer key returned HTTP 401 JSON with `error_type: authentication_error`.
- Server evidence counts: 1 pinned-model load, 1 ready message, 1 disclosed fresh-serial fallback, 1 graceful-stop message.
- Sent SIGTERM only to captured server PID 72973. Server exit code: 0.
- Server stdout: 0 bytes.
- `scripts/sdk-compat/node_modules`: removed and verified absent.

## Failure count

- Native integration test failures: 0
- npm install failures: 0
- SDK smoke failures: 0
- Auth assertions failed: 0
- Shutdown/stdout assertions failed: 0
- Total observed failures: 0

Raw command, stdout, stderr, HTTP response, environment, and structured result files are alongside this summary. `SHA256SUMS` hashes every evidence file other than itself.

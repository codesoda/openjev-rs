# Resident Jev-compatible HTTP service

`openjev serve` is a local, supported subset of the Jev System One wire API.
It loads and verifies one selected GGUF once, performs one disclosed warmup
Choice, then serves all requests through one synchronous inference owner. It is
not affiliated with TypeSafe AI and does not claim hosted Jev calibration,
quality, billing, latency, or numerical equivalence.

## Start

```sh
# Existing Metal binary and verified cache; loopback needs no configured auth.
openjev serve --offline --model qwen3-0.6b \
  --host 127.0.0.1 --port 8080 --request-timeout-secs 120

# A non-loopback listener is rejected unless a nonempty bearer secret is read
# indirectly from an environment variable.
export OPENJEV_API_KEY='replace-with-a-long-secret'
openjev serve --host 0.0.0.0 --api-key-env OPENJEV_API_KEY
```

The default listener is `127.0.0.1:8080`. TLS belongs at a trusted reverse
proxy. Wildcard CORS is not enabled. `serve` cannot be combined with another
command or decision-output flags. Put server options such as `--host` and
`--port` after `serve`. Since v0.2.0, the former `--serve` flag is not supported;
historical v0.1.0 release artifacts still use that flag. Existing model, cache, offline, device,
thread, context, batch, sequence, and `--require-shared` settings still apply.
Startup binds first, then loads and warms the model; readiness and accepting
begin only after load/warmup succeeds. stdout remains empty. Diagnostics go to
stderr.

SIGINT/SIGTERM closes admission, cancels queued work, and joins the scorer.
llama.cpp decode is not interruptible: a request that has already entered a
native decode keeps its admission slot until that call returns. Abandoned work
is stopped between decisions and queued abandoned work is skipped.

## Endpoints

- `POST /v1/systemone`
- `GET /v1/models`
- `GET /healthz`
- `GET /readyz`

Inference accepts `Content-Type: application/json` with optional parameters.
Bodies are capped at 1 MiB, including chunked bodies. To bound per-question
state cloning, Python-serialized state bytes multiplied by total question count
must also fit within 4 MiB; this conservative limit includes singleton questions
and returns 422 without inference when exceeded. At most 16 requests are
admitted across body reading, queueing, and inference. Overload returns 429 and
`Retry-After: 1`. The default 120-second whole-request deadline includes body
reading, queueing, and inference.

Health and readiness are unauthenticated. With `--api-key-env`, model and
inference routes require an exact `Authorization: Bearer ...` value. Loopback
without configured auth accepts the nonempty dummy bearer token that the SDK
always sends, but also permits direct unauthenticated local health/model calls.
Request bodies and secrets are not logged.

Errors are JSON objects with `error_type` and `message`; client input errors do
not include local paths or native details. Success headers disclose bounded
safe metadata:

- `X-OpenJev-Execution`
- `X-OpenJev-Fallback` when serial fallback occurred
- `X-OpenJev-Probability-Status`

## TypeSafe JavaScript SDK 0.6.0

The SDK `baseURL` is the server root, **not** a URL ending in `/v1`; the pinned
SDK appends `/v1/systemone` and `/v1/models` itself.

```js
import { TypeSafeClient, choice, noul, score } from "@typesafe-ai/sdk";

const client = new TypeSafeClient({
  apiKey: "local-dummy-token",
  baseURL: "http://127.0.0.1:8080",
  timeout: 120_000,
  retry: { maxRetries: 0 },
});

const result = await client.systemOne({
  state: { ticket: "Duplicate charge" },
  questions: {
    route: choice("Which queue?", { billing: null, support: null }),
    review: noul("Does a human need to review this?"),
    urgency: score("How urgent?", ["low", "medium", "high"]),
  },
});
```

A reproducible smoke source and npm lockfile are in `scripts/sdk-compat/`:

```sh
cd scripts/sdk-compat
npm ci --ignore-scripts --no-audit --no-fund
OPENJEV_BASE_URL=http://127.0.0.1:8080 npm run smoke
```

They pin `@typesafe-ai/sdk` 0.6.0, whose reviewed source is commit
`66880ccded6cb642dc1809620c2b108c33730214`.

## Supported wire subset

Request shape:

```json
{"model":"jev-latest","state":"...","questions":{"id":{"type":"noul","instructions":"..."}}}
```

- `state`, instructions, and descriptions accept a string, object, array, or
  null. JSON map insertion order is preserved. Duplicate keys, floats, integer
  overflow, and documents deeper than 128 containers are rejected as 422
  validation errors; malformed JSON (including lone surrogates) is 400. Empty/null
  top-level state is preserved under the explicit adapter
  envelope `{"value": <original>}` because the existing OpenJev core contract
  requires nonempty string/object/array state; nonempty state is passed through
  unchanged. Structured state is never converted to a JSON string.
- Omitted, null, or empty instructions become `Select the best option.`.
  Structured entries use the existing Python-compatible JSON renderer. A null
  Choice description uses its semantic label; other explicit null descriptions
  render as `null`.
- Choice accepts 1–16 insertion-ordered labels. A singleton is deterministic
  with probability/confidence 1 and no inference tokens. Score accepts 2–16
  ordered levels and reports the unnormalized expected index. Noul uses
  yes-first probability and optional `true`/`false` descriptions.
- Up to 64 questions are accepted. Arbitrary, empty, and Unicode external
  question IDs and Choice labels are mapped to private nonempty inference IDs
  and mapped back without entering HTTP headers.
- `model` may be omitted, `jev-latest`, or the one public loaded model identity.
  Other values return 404. Custom local paths are represented publicly as
  `openjev-local-<artifact hash prefix>` and are never exposed.
- `/v1/models` returns only the resident model. `release_date: "unknown"`
  honestly means the GGUF metadata/manifest does not provide a trustworthy
  publication date; no TypeSafe release is fabricated.
- Probabilities, Noul values, Score values, and Choice/Score confidence are
  rounded independently to two decimal places in the HTTP projection only.
  Argmax, expected Score, and normalized-margin confidence are computed from
  original core probabilities first. Rounded distributions are not
  renormalized and need not sum to one.
- `usage.input_tokens` sums logical full-prompt token counts per inferential
  question, not unique cached-prefix work or hosted billing units. Deterministic
  singleton Choice contributes zero. `output_tokens` is zero because OpenJev
  performs no generation.

Several inferential questions request the existing receipt-gated shared path.
If that exact native configuration is not eligible or fails, every tentative
row is discarded and the whole request is rescored as fresh serial full prompts.
The fallback is visible in headers and stderr. `--require-shared` fails closed:
requests with fewer than two inferential questions are rejected before scoring,
and multi-question requests without an eligible shared path never fall back.
Without that flag, one inferential question uses direct mode. Model selection and
shared-probe eligibility are startup configuration, not hot-reloaded: restart
the service after changing models, native settings, or probe receipts.

## Verified local smoke

On this Apple M3 Pro, the cached Qwen3-0.6B Metal server passed repeated mixed
HTTP requests and the pinned official SDK smoke, including authenticated
success, wrong-key 401, unknown-model 404, unsupported-float 422, one model load,
empty stdout, and clean SIGTERM shutdown. Final raw evidence and source/binary
hashes are under [`results/serve/20260920T031535Z-final/`](results/serve/20260920T031535Z-final/).
CPU and Metal build/unit checks also passed; this is not a Linux runtime test or
a performance benchmark, and shared execution still falls back to serial on
the tested profiles.

The machine-readable subset is `schemas/jev-http-v1.schema.json`. Runtime
validation is intentionally stricter than JSON Schema for duplicate keys,
numeric lexemes, nesting, and cross-field answer alignment.

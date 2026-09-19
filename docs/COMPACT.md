# Compact decision output

`--compact` is an opt-in projection for LLMs, scripts, and other callers that
need decision results without model/runtime diagnostics. Full
`openjev-readout-v1` output remains the default for backward compatibility and
Python parity.

```sh
openjev --compact --quiet --offline --model qwen3-0.6b decide \
  --state 'customer cannot sign in' \
  --question 'Which queue?' --option Access --option Billing
```

Compact rows use `schema: "openjev-compact-v1"` and always retain `id`,
`choice`, `option_ids`, `probabilities`, and the exact upstream
`probability_status`. Noul also includes `p_yes`; Score includes
`expected_value`, `argmax_level`, and `level_values`. `confidence` and
`confidence_status` appear only when `--confidence` was requested.

Model metadata, raw logits, token IDs, prompt metadata, timings, and other
execution diagnostics are omitted. If execution fell back from the requested
mode, the row includes only `execution.requested_mode`, `effective_mode`, and
`fallback_reason`; otherwise `execution` is omitted. The semantic fallback
warning remains on stderr even with `--quiet`.

`--compact` applies only to `decide`, `noul`, `score`, `ask`, and `run`.
Structured errors remain unchanged `openjev-error-v1` records. `run` remains
JSONL, flushes each row, continues after row errors, and preserves its existing
file summary and exit behavior. `--pretty` remains valid only for a single
row, never `run` or a multi-question `decide`.

`--quiet` suppresses routine native/tracing INFO and DEBUG logs, but retains
WARN/ERROR diagnostics and explicit semantic fallback warnings. Do not use it
as a mechanism to hide critical warnings.

See `schemas/compact-v1.schema.json` for the projection schema. It is separate
from `schemas/readout-v1.schema.json`; compact rows must not be validated as
full readouts.

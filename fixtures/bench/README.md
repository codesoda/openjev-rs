# M6 benchmark fixtures

These project-owned synthetic fixtures contain no copied incident, private data, or hidden Jev material.

- `short-state.txt`: 703-byte small-state coverage.
- `long-state.txt`: approximately 8,000-byte long-state coverage.
- `questions21.jsonl`: one fixed set of 21 validated questions, used with either state.

M6 benchmark reports must identify the exact state and questions SHA-256 values. Timed regions exclude file reads, model loading, warmup, validation, and report writes.

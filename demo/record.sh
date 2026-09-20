#!/usr/bin/env bash
# Real offline inference; the recorded session owns and stops its server/client.
set -euo pipefail
cd "$(dirname "$0")/.."
for tool in openjev vhs ttyd ffmpeg python3; do
  command -v "$tool" >/dev/null || { printf 'Missing dependency: %s\n' "$tool" >&2; exit 1; }
done
openjev demo --help >/dev/null
openjev serve --help >/dev/null
export DEMO_PORT="${DEMO_PORT:-18787}"
export DEMO_URL="http://127.0.0.1:$DEMO_PORT"
export OPENJEV_RECORDING_DIR="$PWD/out/demo-recording"
mkdir -p "$OPENJEV_RECORDING_DIR"
rm -f "$OPENJEV_RECORDING_DIR/responses.jsonl" "$OPENJEV_RECORDING_DIR/complete"
# The recorded server's actual bind remains authoritative if a race occurs.
python3 - "$DEMO_PORT" <<'PY'
import socket, sys
with socket.socket() as listener:
    listener.bind(('127.0.0.1', int(sys.argv[1])))
PY
if [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
  export DEMO_DEVICE=metal
else
  export DEMO_DEVICE=cpu
fi
vhs demo/readme.tape
ffmpeg -v error -sseof -0.1 -i docs/demo.mp4 -frames:v 1 -y "$OPENJEV_RECORDING_DIR/final.png"
python3 - <<'PY'
import json, os
from pathlib import Path
logs = Path(os.environ['OPENJEV_RECORDING_DIR'])
assert (logs/'complete').is_file(), 'Recording did not finish and shut down cleanly'
rows = [json.loads(line) for line in (logs/'responses.jsonl').read_text().splitlines()]
assert len(rows) == 8, 'Incomplete recording'
assert all(row['response']['model'] == 'qwen3-0.6b' for row in rows)
assert all(row['response']['usage']['output_tokens'] == 0 for row in rows)
assert all('state' in row['request'] and 'questions' in row['request'] for row in rows)
assert len(rows[-1]['response']['answers']) == 3
assert not (logs/'server.stdout').read_bytes()
assert not (logs/'client.stderr').read_bytes()
print('Verified real startup, eight request/response pairs, and clean shutdown.')
print('Created docs/demo.gif and docs/demo.mp4.')
PY

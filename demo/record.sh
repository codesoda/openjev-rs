#!/usr/bin/env bash
# Real offline inference; never attaches to or stops a pre-existing server.
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
rm -f "$OPENJEV_RECORDING_DIR/responses.jsonl"
# Fail before launch if the port is in use. The actual server bind is authoritative.
python3 - "$DEMO_PORT" <<'PY'
import socket, sys
with socket.socket() as listener:
    listener.bind(('127.0.0.1', int(sys.argv[1])))
PY
if [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
  device=metal
else
  device=cpu
fi
openjev serve --offline --model qwen3-0.6b --device "$device" --port "$DEMO_PORT" \
  >"$OPENJEV_RECORDING_DIR/server.stdout" 2>"$OPENJEV_RECORDING_DIR/server.stderr" &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python3 - "$server_pid" <<'PY'
import os, sys, time, urllib.request
for _ in range(600):
    os.kill(int(sys.argv[1]), 0)
    try:
        with urllib.request.urlopen(os.environ['DEMO_URL']+'/readyz', timeout=1) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(0.2)
else:
    raise SystemExit('Server not ready; see out/demo-recording/server.stderr')
PY
# No inference/downloads are hidden in the presenter; VHS runs the real demo client.
vhs demo/readme.tape
python3 - <<'PY'
import json, os
from pathlib import Path
rows = [json.loads(line) for line in (Path(os.environ['OPENJEV_RECORDING_DIR'])/'responses.jsonl').read_text().splitlines()]
assert len(rows) == 8, 'Incomplete recording'
assert all(row['response']['model'] == 'qwen3-0.6b' for row in rows)
assert all(row['response']['usage']['output_tokens'] == 0 for row in rows)
assert len(rows[-1]['response']['answers']) == 3
assert not (Path(os.environ['OPENJEV_RECORDING_DIR'])/'server.stdout').read_bytes()
print('Verified eight live responses. Created docs/demo.gif and docs/demo.mp4.')
PY

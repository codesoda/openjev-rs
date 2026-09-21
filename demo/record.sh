#!/usr/bin/env bash
# Render a staged walkthrough. No server, model, credentials, or network needed.
set -euo pipefail
cd "$(dirname "$0")/.."
for tool in vhs ttyd ffmpeg python3; do
  command -v "$tool" >/dev/null || { printf 'Missing dependency: %s\n' "$tool" >&2; exit 1; }
done
mkdir -p out/demo-recording
python3 -m unittest discover -s demo -p 'test_*.py'
vhs demo/readme.tape
ffmpeg -v error -ss 26 -i docs/demo.mp4 -frames:v 1 -y docs/demo-poster.png
ffmpeg -v error -sseof -0.1 -i docs/demo.mp4 -frames:v 1 -y out/demo-recording/final.png
printf 'Created docs/demo.gif, docs/demo.mp4, and docs/demo-poster.png (illustrative walkthrough).\n'

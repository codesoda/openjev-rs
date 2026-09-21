# README walkthrough (VHS)

The README GIF links to [`docs/demo.md`](../docs/demo.md), which contains the
video and explains both curl examples. This is a **staged illustration**, not a
live model recording. No server, model, credentials, or inference is needed to
render it.

## Re-record

Install VHS, ttyd, FFmpeg, Python 3, and a browser supported by VHS. On macOS:

```sh
brew install vhs ttyd ffmpeg
```

From the repository root:

```sh
bash demo/record.sh
```

Outputs:

- `docs/demo.gif` — clickable README animation.
- `docs/demo.mp4` — video on the demo page.
- `docs/demo-poster.png` — video poster.
- `out/demo-recording/final.png` — ignored final frame for review.

## Scenes

1. Type `openjev serve --model qwen3-0.6b`, then show illustrative loading and
   readiness messages. Explain that it stays open in its own terminal.
2. Clear the view. Type a real Choice curl request for support-ticket routing.
   Leave the inputs visible for 8 seconds, then display the illustrative JSON
   response for 10 seconds, keeping the command on screen.
3. Clear the view. Type a Noul curl request detecting an explicit refund request.
   Use the same input/result pauses.

`session.py` only prints: it does **not** execute the displayed commands.
`scenes.json` contains the copyable request bodies and illustrative responses.
The on-screen header and demo page label the simulation. JSON formatting is for
readability; normal curl output is compact unless piped through `jq`.

The tape uses the Discuss CLI-inspired Catppuccin Mocha theme, Menlo 20px, and a
1280×960 viewport. Adjust the pauses in `session.py` and the tape's final wait if
changing the scenes. Keep request bodies synchronized with `docs/demo.md`;
tests check that the displayed curl commands are valid and appear in the docs.

## Check

```sh
python3 -m unittest discover -s demo -p 'test_*.py'
shellcheck demo/record.sh
vhs validate demo/readme.tape
ffprobe -v error -show_entries format=duration,size -of json docs/demo.mp4
```

The earlier live-recording pipeline was replaced deliberately with this shorter
walkthrough. The real `openjev demo` command remains unchanged.

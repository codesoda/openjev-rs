# Run via bash demo/record.sh from the repo root (starts a real resident server).
Output docs/demo.gif
Output docs/demo.mp4

Require openjev
Require python3
Set Shell bash
Set FontSize 20
Set FontFamily "Menlo"
Set Width 1280
Set Height 860
Set Padding 28
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set BorderRadius 12
Set Framerate 15
Set TypingSpeed 25ms

Hide
Type `export PS1='$ '; clear; printf '\033[1;36mOPENJEV / local typed decisions\033[0m\n\nServer already loaded in another terminal:\n  openjev serve --offline --model qwen3-0.6b --port %s\n\nEight real HTTP requests. Compact view, paced for reading.\n\n' "$DEMO_PORT"`
Enter
Show
Sleep 2s
Type `openjev demo --base-url "$DEMO_URL" --quiet | python3 demo/present.py`
Enter
Sleep 26s
Wait+Screen /8 HTTP requests complete/
Sleep 3s
Screenshot out/demo-recording/final.png

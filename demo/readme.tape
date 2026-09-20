# Run bash demo/record.sh from the repo root. No model is preloaded.
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

Hide
Type `export PS1='$ '; clear`
Enter
Show
# Script types the exact commands it runs, and owns the real server/client.
Hide
Type "python3 demo/session.py"
Enter
Show
Sleep 135s
Wait+Screen /8 HTTP requests complete/
Sleep 5s

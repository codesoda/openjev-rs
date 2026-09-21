# Run bash demo/record.sh from the repo root. Staged; no inference is executed.
Output docs/demo.gif
Output docs/demo.mp4

Require python3
Set Shell bash
Set FontSize 20
Set FontFamily "Menlo"
Set Width 1280
Set Height 960
Set Padding 28
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set BorderRadius 12
Set Framerate 15

Hide
Type `export PS1='$ '; clear`
Enter
Show
# Script types illustrative commands and responses; it does not execute them.
Hide
Type "python3 demo/session.py"
Enter
Show
Sleep 56s
Wait+Screen /WALKTHROUGH COMPLETE/
Sleep 4s

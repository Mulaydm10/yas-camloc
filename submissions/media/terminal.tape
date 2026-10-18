# VHS take: strict held-out lap 4 + official scorer. Run from the repo root: vhs submissions/media/terminal.tape
Output submissions/build/out/terminal.mp4
Set Shell zsh
Set FontFamily "JetBrains Mono"
Set FontSize 29
Set Width 1920
Set Height 1080
Set Framerate 60
Set Padding 72
Set LineHeight 1.35
Set TypingSpeed 45ms
Set Theme { "name": "atelier-ink", "black": "#211E18", "red": "#E07A6E", "green": "#8CC79A", "yellow": "#E3B567", "blue": "#8FA9F0", "magenta": "#D99BB7", "cyan": "#8FC7C2", "white": "#E7E0D1", "brightBlack": "#8C8473", "brightRed": "#F0948A", "brightGreen": "#A6D9B2", "brightYellow": "#F0CB86", "brightBlue": "#AFC2F7", "brightMagenta": "#E8B5CC", "brightCyan": "#A9DAD5", "brightWhite": "#F9F6EE", "background": "#211E18", "foreground": "#F1ECE1", "selection": "#4E4A3F", "cursor": "#8FA9F0" }
Hide
Type "cd /Users/mulaydm10/Dhruv/national && setopt interactivecomments && export PATH=$PWD/.venv/bin:$PATH PYTHONWARNINGS=ignore PS1='%F{8}camloc%f %F{blue}$%f ' && clear"
Enter
Sleep 1s
Show
Sleep 1.5s
Type "# held-out lap 4: map + calibration built WITHOUT lap 4, no start pose"
Enter
Sleep 1.2s
Type "./submissions/media/heldout_lap4.sh submissions/build/lap4.csv"
Sleep 0.6s
Enter
Sleep 30s
Sleep 3s
Type "python -m camloc.evaluate submissions/build/lap4.csv --laps 4 --official | head -6"
Sleep 0.6s
Enter
Sleep 9s
Sleep 4s

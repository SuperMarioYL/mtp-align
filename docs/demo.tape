Output assets/demo.gif
Set FontSize 15
Set Width 1100
Set Height 540
Set Padding 20
Set Theme "Dracula"

# Verify install, then run the kill-gate benchmark (synthetic, GPU-free).
Type "mtp-align --version"
Enter
Sleep 1

Type "mtp-align bench"
Enter
Sleep 3

Type "python examples/benchmark.py"
Enter
Sleep 2

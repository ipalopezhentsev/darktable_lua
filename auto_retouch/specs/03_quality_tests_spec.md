There is script detect_dust.py that analyzes images and detects dust on them (described in dust_detection_spec.md).
There is UI (described in debug_ui_spec.md) which shows automatically detected dust spots and allows a human to recategorize errors in found places - missed dust and false positives.
It produces output of human analysis.

Using this UI and human remarking, we tuned (with the help of script analyze_debug_report.py which analyzes human remarkings) parameters of dust detection to good level.
I have a directory of freshly exported debug images and spots analysis (C:\Users\ilyap\AppData\Local\Temp\darktable_autoretouch_1771450557). I think the current state of detection is quite good to
serve as a baseline. 

I want to persist these results in the repo, not in a temp folder and create some test harness script which Claude Code can use to compare potential further improvements to the algo - i.e. when Claude Code changes some detection algo constants, it should be able to rerun dust detection on the same input images (persisted in our repo) and compare new detection results with persisted baseline "good" detection results (which we approved by me, a human). And if they differ significantly this would be a sign that optimization is going not in a good direction.
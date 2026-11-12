# Handover: possession pipeline (Xtenders)

Copy everything in this folder into the repository root, next to `games.py` (tests go in `tests/`).
Keep the tactical files in `data/tactical/`. Run from the repository root.

| File | Owner | What it is |
|---|---|---|
| `deadball.py` | Azzaliah | Dead-ball cues, the D score, the published pre-kickoff rule, Smart Tagging functions |
| `pipeline.py` | Elvis | Cleaning, `home_away_lean` (Terer's part), Viterbi, tidy-up, checks. Parameters in `P` |
| `run_full.py` | Elvis | One command that builds the submission |
| `rawatd.py` | shared | Fast, verified reader for the tracking file |
| `explore_deadball.py`, `weak_labels.py` | Azzaliah | Reports on the data; calibration of the cue weights from the restart tags |
| `tests/` | all | 26 tests; 4 need data files (the 2 broadcast ones skip if you only have the tactical files) |

## Commands
    python -m pytest -q                                  # 24 pass and 2 skip with only data/tactical present
    python run_full.py --out submission/<team_name>.csv  # build the file
    python evaluation/validate_submission.py submission/<team_name>.csv

Rerunning on the same data reproduces the file byte for byte.

## Add to .gitignore
    submission/
    deadball_report/
    *.npy

## Open decisions (nobody has decided these yet)
1. Smart Tagging team bonus: off by default (`--tag-w 0`). Even 0.1 changes 6% of graded frames.
2. `dead_bias` assumes 35% of play is dead. That is an assumption, not a measurement.
3. 2,934 second-half frames before the kick-off are forced to D, from numbers the organisers
   published. Confirm with them that this is allowed; if not, remove `force_dead=` in `run_full.py`.
4. `min_poss_secs` (1.5 s) in `pipeline.py` is Elvis's call.

## Known weak spots
Goal-kick stoppages are mostly missed. Dead-ball recall on the 5 public minutes was 0.25 (not used
to tune anything). Possession changes look about half a second early on those minutes.
Nothing has been scored on the full match: its answer key is private.

## Rules reminders
Never tune on the public 5 minutes (use them only to check the file is wired correctly). No hand
labels. No match data, derived frame-level files or videos in the repository.

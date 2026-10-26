# Team progress

This file records the team's current execution path. It does not replace the competition rules.

## Milestone 1: data handling and experiment policy

Keep the downloaded match bundle outside Git. The code accepts a data root through
`MTAH_DATA_DIR`; that root must contain the `tactical/` and `broadcast/` folders. On this checkout,
the local bundle is in `Challenge-Datasets/`.

The public five-minute labels are only for checking that the pipeline is wired correctly. Do not
use their scores or label patterns to choose features, thresholds, or model settings. Do not create
hand labels for this match or publish match data or derived tracking exports.

## Milestone 2: reproducible reference baseline

From the repository root in Windows PowerShell, point the loader at the local data bundle, generate
the smoothed 50-frame reference baseline outside the repository, and validate that exact file:

```powershell
$env:MTAH_DATA_DIR = "D:\path\to\Challenge-Datasets"
$baselineOut = Join-Path $env:TEMP "mtah2027-baseline"
python starter\baseline.py --window 50 --out $baselineOut
python evaluation\validate_submission.py (Join-Path $baselineOut "baseline_smooth50.csv")
```

The generation command reads the tactical ATD and metadata, then writes one label for each of the
199,319 tactical video frames. The validator checks the CSV contract; it does not check prediction
quality. This reference method assigns the nearest tracked player's team when the ball is visible
and labels missing-ball frames `D`, so it is a starting point rather than the team's final model.

The submission index is the tactical video frame: ATD `frame_id` is one-based and maps to
`video_frame = frame_id - 1`. Keep the tactical and broadcast timelines separate.

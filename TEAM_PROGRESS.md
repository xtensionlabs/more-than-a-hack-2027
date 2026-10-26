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

## Milestone 3: team-level feature audit

`starter/feature_audit.py` creates frame-aligned, team-level evidence without reading possession
labels. It records ball visibility and pitch bounds, ball speed, visible track-slot counts by team,
each team's nearest-player distance and speed, and the away-minus-home distance margin. Track slots
are aggregated by their metadata team; the audit does not treat them as persistent player IDs.

Run the audit from the repository root. The compressed arrays and aggregate JSON are written to a
temporary folder outside the repository and should remain local:

```powershell
$env:MTAH_DATA_DIR = "D:\path\to\Challenge-Datasets"
$featureOut = Join-Path $env:TEMP "mtah2027-feature-audit"
python starter\feature_audit.py --out $featureOut
```

The distance margin is positive when the nearest home track is closer, negative when the away track
is closer, and missing if either team has no visible track near an observed ball. An out-of-pitch
flag is only a geometric clue; it does not by itself label a frame as dead ball.

The full tactical audit completed with contiguous frame IDs and all 199,319 frame rows. Its outputs
were written to the local temporary directory; no feature archive or match-derived summary was
added to the repository. The audit uses tracking metadata only and does not load ground truth.

## Milestone 4: possession continuity decoder

`starter/possession_decoder.py` converts the team-distance evidence into a signed score, then uses
a two-state Viterbi decoder independently on each half. A fixed transition cost prevents weak or
brief opposing evidence from splitting a spell; missing-ball frames have neutral emissions, so the
decoder carries the best-supported team through short and long tracking gaps. The score combines
the home/away distance margin with absolute ball proximity, down-weighting frames where both teams
are far away. The default change penalty is conservative: brief or weak contrary evidence must
accumulate before a team switch. Scales and penalty are exposed for synthetic and sensitivity
checks, never for tuning against the public answer key.

The current candidate labels only `H`, `A`, and `X`; this milestone does not infer `D`. Run it after
the feature audit. The CSV, evidence archive, and diagnostics stay outside the repository:

```powershell
$decoderOut = Join-Path $env:TEMP "mtah2027-possession-decoder"
python starter\possession_decoder.py --out $decoderOut
python evaluation\validate_submission.py (Join-Path $decoderOut "possession_candidate.csv")
```

The validator checks only the submission format. Do not run the scorer on the public labels to
select decoder settings.

The synthetic checks passed for a sustained handover, a one-frame opposing spike, and a neutral
missing-ball gap. The full tactical candidate also passed the official format validator. No public
answer-key score was used; the output is an H/A continuity candidate, not the final submission.

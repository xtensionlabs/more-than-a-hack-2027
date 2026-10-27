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
labels. It records ball visibility and pitch bounds, raw and position-derived ball movement,
visible track-slot counts by team, team movement summaries, each team's nearest-player distance and
speed, and the away-minus-home distance margin. Track slots are aggregated by their metadata team;
the audit does not treat them as persistent player IDs.

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

## Milestone 5: dead-ball and restart evidence

`starter/dead_ball_decoder.py` combines the possession evidence with physical stoppage cues in a
three-state Viterbi decoder. It uses persistent out-of-pitch observations, prolonged low ball
movement, nearby-player distance, and team movement. Missing ball observations remain neutral.
Smart Tagging set-piece intervals are a weak context only; some span tens of seconds. A tag can
reinforce a corroborated slow-ball cue, but a tag by itself cannot emit `D` or copy its interval
into the submission.

The feature audit smooths pitch positions before estimating ball displacement because the raw ball
speed field is often absent. The stoppage detector requires repeated low-speed evidence and applies
a transition cost so brief tracking noise does not fragment possession spells. The default dead-ball
transition cost was checked with unlabeled sensitivity runs; it rejects single-frame dead-ball
artifacts. No answer-key scores were used.

Regenerate the extended features, decode with the Smart Tagging file, and validate the resulting
candidate:

```powershell
$env:MTAH_DATA_DIR = "D:\path\to\Challenge-Datasets"
$featureOut = Join-Path $env:TEMP "mtah2027-feature-audit"
$deadOut = Join-Path $env:TEMP "mtah2027-dead-ball-decoder"
python starter\feature_audit.py --out $featureOut
python starter\dead_ball_decoder.py --features (Join-Path $featureOut "tracking_features.npz") --data-dir $env:MTAH_DATA_DIR --out $deadOut
python evaluation\validate_submission.py (Join-Path $deadOut "possession_with_dead_ball.csv")
```

This is still a candidate: the physical cues and transition costs need independent validation, and
the final model must be checked for both stoppage quality and spell fragmentation without tuning
on the public answer key.

Synthetic checks passed for tag-only input, missing-ball gaps, isolated out-of-pitch observations,
and a sustained stoppage. The full tactical candidate passed format validation; its predictions and
diagnostics remain in the local temporary output folder. No public answer-key score was used.

## Milestone 6: label-free robustness checks

`starter/stress_test.py` runs deterministic synthetic checks plus controlled masking of visible
ball observations. It tests random dropout and contiguous gaps of different lengths, then reports
agreement on unmasked graded frames and per-half H/A/D spell counts and deltas. The report contains
no frame labels and is written outside the repository. These checks measure stability under
missing data; they do not estimate competition accuracy.

Run the extended feature audit first, then the stress harness. Keep the JSON report local:

```powershell
$env:MTAH_DATA_DIR = "D:\path\to\Challenge-Datasets"
$featureOut = Join-Path $env:TEMP "mtah2027-feature-audit"
$stressOut = Join-Path $env:TEMP "mtah2027-stress-test"
python starter\feature_audit.py --out $featureOut
python starter\stress_test.py --features (Join-Path $featureOut "tracking_features.npz") --data-dir $env:MTAH_DATA_DIR --out $stressOut
```

The deterministic run passed all seven synthetic checks. Ball visibility on scored frames was
59.307% before artificial masking. Removing 10%, 25%, and 40% of visible observations at random
left 99.590%, 97.733%, and 95.701% agreement on the remaining graded frames. Removing about 15%
in contiguous bursts of 25, 75, and 125 frames left 97.099%, 97.167%, and 95.785% agreement. The
125-frame burst case reduced predicted live spells by 26 in the first half and 29 in the second;
40% random masking reduced them by 72 and 46. Long missing stretches therefore remain a material
weakness even though most unmasked frame labels stayed stable. Agreement and spell counts are
structural diagnostics only, not accuracy estimates or tuning targets. The full report is local at
`%TEMP%\mtah2027-stress-test-m6\stress_report.json`.

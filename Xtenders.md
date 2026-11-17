# Xtenders - Initial Challenge Technical Report

**Team:** Xtenders | **Registered members:** 3

**Entry:** frame-level home possession (`H`), away possession (`A`), dead ball (`D`), or outside the graded windows (`X`)

## System overview

The entry converts the organizers' automatic tactical tracking into per-frame possession scores, then decodes each half as a temporally consistent state sequence:

1. Read the tactical ATD positions and the accompanying team metadata. Check frame IDs, row counts, and team-slot alignment.
2. Convert normalized coordinates to metres using the supplied pitch dimensions. Extract team-symmetric features for ball visibility and motion, nearest and second-nearest players, player speeds, counts within 2/5/10/20 metres, and short rolling summaries.
3. Apply two frozen logistic models: one estimates the live-ball owner and one estimates dead-ball probability. Team-mirrored training examples reduce home/away bias.
4. Combine the probabilities into `H/A/D` emissions and use a Viterbi decoder with owner and dead-state transition costs. Decode halves independently so smoothing does not cross halftime.
5. Write one row per video frame. The runner assigns `X` outside the two graded-half intervals.

This design uses both individual proximity and team movement, distinguishes inactive play from live possession, and adds temporal consistency instead of deciding each frame in isolation. Mirrored training examples make the owner model team-symmetric. A nearest-player-only rule misses collective evidence, stoppages, and sequence context; the external-only comparison below measures the gain over that simple reference.

The production path uses tactical tracking and its team metadata only. We did not use the tactical or broadcast video, Smart Tagging, or homography. The inference runner contains no fitting or calibration path.

## Possession decision rules

The output has no separate contested-ball or loose-ball class. `H` and `A` mean the model assigns live possession to that team; `D` means it assigns the ball inactive. The external DFL convention defines possession as the team of the player in control, and marks play inactive during referee interruptions. Our model learns that distinction from the supplied frame labels rather than adding a challenge-match rule.

At inference, the owner classifier compares both teams symmetrically using proximity, player counts, movement, and ball motion. The dead-ball classifier uses ball visibility, whether the ball is outside the pitch, ball movement, player movement, and nearby-player distances. The Viterbi decoder selects the H/A/D sequence with the lowest combined emission and transition cost. It penalizes an H/A owner switch by `12` and a live/dead change by `72`. There is no separate contested, loose-ball, or unknown state inside a graded half: an ambiguous frame is resolved as H, A, or D according to the model scores and sequence costs. When live scores are close, the transition cost can preserve the preceding owner through a brief ambiguous interval. There is no contact detector, restart-event input, fixed restart timer, or hand-authored frame rule. Restarts are recognized only through the learned live/dead scores and sequence transitions. This can misclassify deflections, prolonged challenges, or restart frames when tracking evidence is weak.

## External training data and method

The only labeled training dataset was seven IDSSE Bundesliga matches: `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ`, and `J03WR9`. The [IDSSE dataset](https://doi.org/10.6084/m9.figshare.28196177) distributes official DFL tracking and framewise `BallPossession` and `BallStatus` fields under CC BY 4.0. It is described by [Bassek et al. (2025)](https://www.nature.com/articles/s41597-025-04505-y); the [Floodlight DFL parser](https://github.com/floodlight-sports/floodlight/blob/main/floodlight/io/dfl.py) documents how the fields are read.

Training labels come from those external DFL fields: `BallStatus=0` is inactive/dead; `BallStatus=1` is active/live; when active, `BallPossession=1` maps to home and `BallPossession=2` to away. Missing or unrecognized codes are excluded from fitting. Across the seven matches, the final model uses 293,931 home, 259,174 away, and 449,539 dead-ball labeled frames, 1,002,644 labeled frames in total. These labels were supplied by the DFL dataset; Xtenders created no training labels. The paper describes quality checks for event data but does not name individual annotators or specify the production process for these two framewise fields, so we do not claim they were manually or automatically annotated by a particular person or system.

## Model configuration and tuning

The two classifiers use class-balanced logistic loss. Feature schema v3, L2 `0.0001`, owner transition cost `12`, dead-state transition cost `72`, and dead-logit offset `0.0` are frozen in [`starter/external_model.json`](starter/external_model.json), which also records weights, training provenance, and fold metrics. Six leave-one-match-out folds informed parameter selection using mean possession-spell F1 across IoU thresholds 0.3-0.7. The final weights were then fit on all seven matches. Refit and benchmark commands are in [README.md](README.md); both scripts consume external data only.

## Validation evidence

The Initial Challenge ranking is automatic: possession-spell F1 decides and macro-F1 breaks ties, according to the [official challenge overview](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/OVERVIEW.md). This report supports reproducibility rather than a judged pitch. Results below use the available evidence; external results do not predict rank against other teams.

| Evaluation set | Model spell-F1 (mean IoU 0.3-0.7) | Model spell-F1 (IoU 0.5) | Model macro-F1 | Reference |
|---|---:|---:|---:|---|
| Six external leave-one-match-out folds | 0.5292 | 0.5414 | 0.8985 | Nearest-player: 0.0953 / 0.0832 / 0.3701 |
| `J03WN1` development diagnostic | 0.4683 | 0.4806 | 0.8679 | Nearest-player spell-F1: 0.1167 |
| Public challenge five-minute window | 0.2732 | - | 0.7715 | Official `baseline_smooth50`: 0.3256 / - / 0.6207 |

The six-fold row averages the best setting selected within each fold; it is tuning evidence, not a score for one fixed setting on unseen matches. For the `J03WN1` diagnostic, that match was excluded from the diagnostic fit and parameter selection. It was later included in the final all-seven-match production fit. Its result was inspected during development, so it is not a pristine test estimate. The external comparison establishes an advantage over nearest-player assignment on those external matches only.

The public-window figure is one automatic check on 17 reference possessions. The evaluator's estimated noise width was about `0.23`; this sample cannot establish a reliable ranking difference. The frozen candidate scored below the published baseline on spell-F1, so we do not claim a demonstrated advantage on the challenge distribution or a top-20 position from the evidence available. The check was run after the candidate was frozen; no weights, parameters, or code were changed in response. No video comparison or private-ground-truth access was used.

## Compliance and limitations

The organizers' latest rules clarification supersedes earlier material. No challenge-match labels were hand-created; no challenge data was uploaded to AI services; and no training, fitting, calibration, or parameter tuning used the challenge ground truth or video. The public five-minute reference was used once for an automatic check after freezing the candidate. We did not inspect labels manually or change weights, parameters, code, or output in response. Challenge data and generated submissions are excluded from Git.

The external set covers seven matches from one competition and tracking provider. Its possession codes may differ from the challenge's interpretation of contested possession, loose balls, and stoppages. The ball is often unobserved, so long gaps depend on player movement and temporal decoding. The short public sample and the external folds cannot resolve this domain shift or establish competition rank. With more time, the next steps would be to add legally reusable matches from other providers, obtain external annotations that distinguish contacts and stoppages more precisely, and evaluate that frozen system on additional held-out matches before making any performance claim.

## Reproduction and output checks

The repository and command below follow the [official submission requirements](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md): install the pinned packages with `python -m pip install -r requirements-solution.txt`, set `MTAH_DATA_DIR` to the organizer bundle root, and run:

```powershell
$env:MTAH_DATA_DIR = 'D:\path\to\Challenge-Datasets'
python run_full.py --out submission/Xtenders.csv
python evaluation/validate_submission.py submission/Xtenders.csv
```

The runner checks contiguous one-based tracking IDs, expected length, both teams, and metadata alignment. It writes the required `frame,label` CSV with 199,319 rows, frames `0` through `199318`, and labels `H/A/D/X`. The validator checks format only; it does not score predictions. The frozen `Xtenders.csv` passed validation. Repeated inference on identical data and model produced SHA-256 `A6D8402FD9B4B8D5FC6728A78BE98432C23794EC704364B570F4B09B4FB5CCFE`.

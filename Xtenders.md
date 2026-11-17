# Xtenders — Initial Challenge Technical Report

**Team:** Xtenders · **Registered members:** 3  
**Entry:** frame-level home possession (`H`), away possession (`A`), dead ball (`D`), or outside the graded windows (`X`)

## System overview

The entry converts the organizers' automatic tactical tracking into per-frame possession scores, then decodes each half as a temporally consistent state sequence:

1. Read the tactical ATD positions and the accompanying team metadata. Check frame IDs, row counts, and team-slot alignment.
2. Convert normalized coordinates to metres using the supplied pitch dimensions. Extract team-symmetric features for ball visibility and motion, nearest and second-nearest players, player speeds, counts within 2/5/10/20 metres, and short rolling summaries.
3. Apply two frozen logistic models: one estimates the live-ball owner and one estimates dead-ball probability. Team-mirrored training examples reduce home/away bias.
4. Combine the probabilities into `H/A/D` emissions and use a Viterbi decoder with owner and dead-state transition costs. Decode halves independently so smoothing does not cross halftime.
5. Write one row per video frame. The runner assigns `X` outside the two graded-half intervals.

The production path uses tracking and team metadata only. It does not load video, Smart Tagging, or challenge possession labels. It contains no fitting or calibration path.

## External training data and method

The model was trained on seven IDSSE Bundesliga matches: `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ`, and `J03WR9`. The DFL tracking and machine-generated possession codes are distributed under CC BY 4.0 by the [IDSSE dataset](https://doi.org/10.6084/m9.figshare.28196177), described by [Bassek et al. (2025)](https://www.nature.com/articles/s41597-025-04505-y). The [Floodlight DFL parser](https://github.com/floodlight-sports/floodlight/blob/main/floodlight/io/dfl.py) documents the source code fields.

Training labels come from those external DFL fields: `BallStatus=0` is dead ball; `BallStatus=1` is live; `BallPossession=1` maps to home and `BallPossession=2` to away. Missing or unrecognized codes are excluded from fitting. Across the seven matches, the final model uses 293,931 home, 259,174 away, and 449,539 dead-ball labeled frames.

The two classifiers use class-balanced logistic loss. Feature schema v3, L2 `0.0001`, owner transition cost `12`, dead-state transition cost `72`, and dead-logit offset `0.0` are frozen in [`starter/external_model.json`](starter/external_model.json), which also records weights, training provenance, and fold metrics. Six leave-one-match-out folds informed parameter selection using mean possession-spell F1 across IoU thresholds 0.3–0.7. The final weights were then fit on all seven matches. Refit and benchmark commands are in [README.md](README.md); both scripts consume external data only.

## Validation evidence

The Initial Challenge ranks possession-spell F1, with macro-F1 as a tie-break, according to the [official challenge overview](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/OVERVIEW.md). Results below use the available evidence; external results do not predict rank against other teams.

| Evaluation set | Model spell-F1 (mean IoU 0.3–0.7) | Model spell-F1 (IoU 0.5) | Model macro-F1 | Reference |
|---|---:|---:|---:|---|
| Six external leave-one-match-out folds | 0.5292 | 0.5414 | 0.8985 | Nearest-player: 0.0953 / 0.0832 / 0.3701 |
| `J03WN1` development diagnostic | 0.4683 | 0.4806 | 0.8679 | Nearest-player spell-F1: 0.1167 |
| Public challenge five-minute window | 0.2732 | — | 0.7715 | Official `baseline_smooth50`: 0.3256 / — / 0.6207 |

The six-fold row averages the best setting selected within each fold; it is tuning evidence, not a score for one fixed setting on unseen matches. `J03WN1` was excluded from fitting and parameter selection, but its results were inspected during development, so it is not a pristine test estimate. The external comparison establishes an advantage over nearest-player assignment on those external matches only.

The public-window figure is one automatic check on 17 reference possessions. The evaluator's estimated noise width was about `0.23`; this sample cannot establish a reliable ranking difference. The frozen candidate scored below the published baseline on spell-F1, so we do not claim a demonstrated advantage on the challenge distribution or a top-20 position from the evidence available. The check was run after the candidate was frozen; no weights, parameters, or code were changed in response. No video comparison or private-ground-truth access was used.

## Compliance and limitations

The organizers' latest rules clarification supersedes earlier material. No challenge-match labels were hand-created; no challenge data was uploaded to AI services; and no training, fitting, calibration, or parameter tuning used the challenge ground truth or video. The one public-window check was automatic. Challenge data and generated submissions are excluded from Git.

The external set covers seven matches from one competition and tracking provider. Its possession codes may differ from the challenge's interpretation of contested possession, loose balls, and stoppages. The ball is often unobserved, so long gaps depend on player movement and temporal decoding. The short public sample and the external folds cannot resolve this domain shift or establish competition rank.

## Reproduction and output checks

Follow [README.md](README.md) to install the pinned dependencies, set `MTAH_DATA_DIR`, run `run_full.py`, and validate the output. The production run checks contiguous one-based tracking IDs, expected length, both teams, and metadata alignment. The validator checks the CSV format only; it does not score predictions. The frozen output contained 199,319 ordered rows and passed validation. Repeated inference on the same data and model produced the same SHA-256 digest: `A6D8402FD9B4B8D5FC6728A78BE98432C23794EC704364B570F4B09B4FB5CCFE`.

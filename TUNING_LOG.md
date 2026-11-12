# Tuning log

Rule we followed: no setting was chosen from the public 5 minutes of ground truth, and nothing was
labelled by hand. Evidence used instead: Smart Tagging restart tags as weak evidence (no ground
truth), D share and number of D stretches per half, possession counts, and agreement between cues.

## Final dead-ball settings (deadball.py, DEAD_P)
| Parameter | Value | Basis |
|---|---|---|
| out_margin | 0.5 m | ball more than 0.5 m outside the lines |
| still_range / still_secs | 1.0 m / 2 s | starting value; cue is almost noise |
| lost_secs / edge_dist / lost_min_players | 2 s / 5.0 m / 10 | edge_dist best of 1.5, 2.5, 5, 10 on weak-label separation |
| calm_speed / calm_secs | 1.3 m/s / 3 s | among the best of 0.6 to 2.0 on weak-label separation |
| w_out, w_still, w_lost, w_calm | 2.1, 0.4, 2.6, 1.1 | log-likelihood ratios (on minus off) from weak_labels.py |
| w_gapcalm | 0.6 | measured 1.2, halved on purpose (see experiment 3) |
| dead_bias | -2.1 | summed LLR off (-1.53) + log prior odds of dead (-0.62). The 35% dead prior is an ASSUMPTION |
| restart window | +1.0 from 8 s before to 3.5 s after each set-piece tag start | tags start about 3.5 s before the restart |
| live term | -2.0 from 5.5 s to 10 s after each tag start | ball stays unseen after restarts |
| forced D | 2,934 frames (110,808 to 113,741) | organiser-published (FAQ.md); confirm with organisers |

## Experiments (judged without labels)
| # | Change | Effect | Kept? |
|---|---|---|---|
| 1 | Fix absorb_short (it stopped after 10 merges) | Possessions 177 / 179 to 152 / 156 per half. D stretches 169 / 171 to 35 / 52; those under 2 s fell from 132 / 117 to 0 | Yes |
| 2 | Live term after restarts | D 2 to 3 s after a restart: 36% to 2% (by construction). Independent support: corner balls move at about 28 m/s 2 to 4 s after the tag starts | Yes |
| 3 | gap_calm weight 1.2 (full LLR) | D 30.4% / 38.1%; away possession share fell from 38% to 29% | No |
| 4 | gap_calm weight 0.6 | D 20.9% / 28.8%; D before kick-offs after goals 0% to 100% | Yes |
| 5 | Slow-ball cue for goal kicks | Restart windows 5.0% against 3.2% elsewhere at under 1 m/s: no separation | No |
| 6 | Smart Tagging team bonus, weight 0.1 / 0.3 | 6.4% / 9.2% of graded frames change; home share of second half up about 8 points | Off (open decision) |

## Results of the current file (xtenders.csv)
| | First half | Second half |
|---|---|---|
| H / A / D share | 41.3% / 37.8% / 20.9% | 33.7% / 37.5% / 28.8% |
| Possessions, median length | 159, 7.6 s | 157, 8.0 s |
| D stretches, median, longest | 40, 11.7 s, 56 s | 54, 11.1 s, 117 s |

Official validator: valid, 199,319 rows, 316 possessions, mean 15.1 s.

## Wiring check (public 5 minutes, scored once after the settings were frozen; not used to tune)
macroF1 0.650, spellF1 0.449 (D precision 0.61, recall 0.25). H and A swapped: spellF1 0.036.

## Add a row below for every change from now on
| # | Change | D share H1 / H2 | Possessions H1 / H2 | Kept? | Who |
|---|---|---|---|---|---|

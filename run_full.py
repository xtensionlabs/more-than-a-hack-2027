"""One end-to-end run: tracking -> dead-ball cues + home/away lean -> Viterbi -> submission CSV.

    python run_full.py [--out submission/xtenders.csv]

Dead-ball part: deadball.py (Azzaliah). Everything else is the Build Guide's starting version
from pipeline.py (cleaning, lean vote, decoder), to be improved by Elvis and Terer.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "starter"))
from games import GAME, half_video_blocks, require  # noqa: E402
import deadball as db  # noqa: E402
import pipeline as pl  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
from load_smart_tagging import load as load_tags  # noqa: E402
from rawatd import load_raw_atd, mean_player_speed, n_slots_from_meta  # noqa: E402


# +1: the tag's team is the team IN possession, -1: the team DEFENDING. Measured with
# db.tag_agreement() against the first full label file; codes near 0.5 carry no information.
SIGN_BY_CODE = {
    "BUILD UP": 1, "PROGRESSION": 1, "ATTACKING TRANSITION": 1, "LONG BALL": 1,
    "DEFENDING IN ATTACKING THIRD": -1, "DEFENDING IN MIDDLE THIRD": -1,
    "DEFENSIVE TRANSITION": -1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission/xtenders.csv")
    ap.add_argument("--tag-w", type=float, default=0.0,
                    help="weight of the verified Smart Tagging team bonus for H/A (0 = off)")
    args = ap.parse_args()

    meta, raw = require(["atd_meta", "atd_raw"])
    ids, players, ball = load_raw_atd(raw, n_slots_from_meta(meta))
    n = len(ids)
    assert ids[0] == 1 and ids[-1] == n == 199_319                  # row i == video frame i
    team = team_of_each_track(meta)
    home, away = np.flatnonzero(team == 0), np.flatnonzero(team == 1)
    print("slots:", len(team), "home", len(home), "away", len(away))

    xy = players[:, :, :2] * np.array([db.L, db.W])
    bx_raw, by_raw = ball[:, 0] * db.L, ball[:, 1] * db.W
    speed = mean_player_speed(players[:, :, 2])
    n_tracked = np.isfinite(players[:, :, 0]).sum(axis=1)

    # dead ball (calibrated on the raw ball, so use the raw ball)
    cues = db.dead_cues(bx_raw, by_raw, speed, n_tracked)
    tags = db.tag_table(load_tags())
    vote = db.dead_vote(cues, tag_bonus=db.restart_bonus(tags, n), force_dead=db.forced_dead_mask(n))

    # home / away lean from the cleaned ball and smoothed players
    bx, by, _ = pl.clean_ball(bx_raw, by_raw)
    xs = pl.smooth_players(xy)
    lean = pl.home_away_lean(bx, by, xs, home, away)

    scores = np.column_stack([lean / 2, -lean / 2, vote])
    if args.tag_w > 0:
        scores[:, :2] += db.tag_bonus_array(tags, n, SIGN_BY_CODE, w=args.tag_w)
    blocks = half_video_blocks(GAME)
    decoded = [pl.to_letters(pl.decode_half(scores[a:b + 1])) for _h, a, b in blocks]
    labels = pl.write_labels(n, blocks, decoded)
    pl.validate(labels, n, blocks)
    pl.summary(labels, blocks)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame({"frame": np.arange(n), "label": labels}).to_csv(args.out, index=False)
    print("wrote", args.out)

    # dead-ball plausibility: D share and stretches per half (no labels involved)
    for half, a, b in blocks:
        seg = labels[a:b + 1]
        runs = [(s, e) for s, e, v in pl.runs(seg) if v == "D"]
        lens = np.array([e - s for s, e in runs]) / db.FPS
        print("half {}: D share {:.1%}, {} D stretches, median {:.1f} s, longest {:.0f} s, under 2 s: {}".format(
            half, (seg == "D").mean(), len(runs), np.median(lens) if len(lens) else 0,
            lens.max() if len(lens) else 0, int((lens < 2).sum())))
    np.save(os.path.splitext(args.out)[0] + "_vote.npy", vote.astype(np.float32))


if __name__ == "__main__":
    main()

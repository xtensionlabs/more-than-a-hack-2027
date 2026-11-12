"""Label-free check of the dead-ball cues, using the Smart Tagging restarts as weak evidence.

No ground truth is read. "Near a restart" frames (the stoppage that precedes a set piece) are the
weak positives; frames far from every restart tag (and outside the published pre-kickoff stretch)
are the weak negatives. Tags are inferred, so this is a sanity and calibration tool, not truth:
dead balls that no tag covers (goals, injuries, offside restarts...) fall in the negatives and
bias every rate down together.

    python weak_labels.py [--pre 6 --post 3 --far 40]
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
from load_smart_tagging import load as load_tags  # noqa: E402
from rawatd import load_raw_atd, mean_player_speed, n_slots_from_meta  # noqa: E402


def auc(score, pos, neg):
    """Probability a random positive frame scores higher than a random negative one (ties 0.5)."""
    s = pd.Series(np.concatenate([score[pos], score[neg]]))
    r = s.rank().to_numpy()
    n1, n0 = pos.sum(), neg.sum()
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load_all(source="tactical"):
    meta, raw = require(["atd_meta", "atd_raw"], source=source)
    ids, players, ball = load_raw_atd(raw, n_slots_from_meta(meta))
    bx, by = ball[:, 0] * db.L, ball[:, 1] * db.W
    return ids, players, bx, by


def masks(tags, n, pre, post, far):
    graded = np.zeros(n, dtype=bool)
    for _h, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    restarts = db.restart_table(tags)
    restarts = restarts[~restarts["type"].str.startswith("kick off")]      # kick-offs are handled by the published rule
    pos = (db.restart_prior(restarts, n, pre, post) > 0) & graded
    near = db.restart_prior(restarts, n, far, far) > 0
    neg = graded & ~near & ~db.forced_dead_mask(n)
    return graded, pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", type=float, default=6)
    ap.add_argument("--post", type=float, default=3)
    ap.add_argument("--far", type=float, default=40)
    args = ap.parse_args()
    ids, players, bx, by = load_all()
    n = len(ids)
    speed = mean_player_speed(players[:, :, 2])
    n_tracked = np.isfinite(players[:, :, 0]).sum(axis=1)
    tags = db.tag_table(load_tags())
    graded, pos, neg = masks(tags, n, args.pre, args.post, args.far)
    cues = db.dead_cues(bx, by, speed, n_tracked)
    print("weak positives: {} frames ({:.1%} of graded)   weak negatives: {} frames ({:.1%})".format(
        pos.sum(), pos.sum() / graded.sum(), neg.sum(), neg.sum() / graded.sum()))
    print("\ncue              P(cue|pos)  P(cue|neg)   lift   LLR if on   LLR if off")
    for c in cues.columns:
        v = cues[c].to_numpy()
        p1, p0 = v[pos].mean(), v[neg].mean()
        eps = 1e-3
        print("{:<16} {:9.3f} {:11.3f} {:7.2f} {:10.2f} {:11.2f}".format(
            c, p1, p0, p1 / max(p0, eps), np.log((p1 + eps) / (p0 + eps)),
            np.log((1 - p1 + eps) / (1 - p0 + eps))))
    # Naive-Bayes weights: total log-odds = sum of LLR_off + sum over active cues of (LLR_on - LLR_off)
    print("\nsuggested weights (LLR_on - LLR_off) and the sum of LLR_off to fold into dead_bias:")
    total_off = 0.0
    for c in cues.columns:
        v = cues[c].to_numpy(); p1, p0 = v[pos].mean(), v[neg].mean(); eps = 1e-3
        on, off = np.log((p1 + eps) / (p0 + eps)), np.log((1 - p1 + eps) / (1 - p0 + eps))
        total_off += off
        print("  w_{:<15} {:5.2f}".format(c, on - off))
    print("  sum of LLR_off  {:5.2f}   (dead_bias = log prior odds of dead + this)".format(total_off))
    ok = np.isfinite(speed)
    print("\nAUC of the 3 s mean player speed (lower = more dead): {:.3f}".format(
        1 - auc(pd.Series(speed).rolling(75, center=True, min_periods=19).mean().fillna(99).to_numpy(), pos & ok, neg & ok)))
    print("n_tracked >= 10 on pos/neg: {:.2f} / {:.2f}".format((n_tracked[pos] >= 10).mean(), (n_tracked[neg] >= 10).mean()))


if __name__ == "__main__":
    main()

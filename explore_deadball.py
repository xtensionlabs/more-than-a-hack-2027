"""Look at dead ball in the real data: cue rates, co-occurrence, timelines, restart-tag alignment.

    python explore_deadball.py                      # tactical (what is submitted)
    python explore_deadball.py --source broadcast   # the other camera; for looking, not scoring

Nothing here reads a ground truth, and nothing is tuned: it only reports what the cues do on the
data so that thresholds can be chosen from the data's own behaviour. Windows for the timeline
figures are chosen by rule (the longest stretches the dead-ball vote is positive), not by hand.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "starter"))
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
import deadball as db  # noqa: E402
from rawatd import load_raw_atd, mean_player_speed, n_slots_from_meta  # noqa: E402


def longest_runs(mask, k):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    runs = sorted(zip(edges[::2], edges[1::2]), key=lambda r: r[0] - r[1])
    return runs[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tactical", choices=["tactical", "broadcast"])
    ap.add_argument("--out", default="deadball_report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta, raw = require(["atd_meta", "atd_raw"], source=args.source)
    ids, players, ball = load_raw_atd(raw, n_slots_from_meta(meta))
    n = len(ids)
    assert ids[0] == 1 and ids[-1] == n, "ATD ids are not 1..N"
    bx, by = ball[:, 0] * db.L, ball[:, 1] * db.W
    speed = mean_player_speed(players[:, :, 2])
    n_tracked = np.isfinite(players[:, :, 0]).sum(axis=1)
    cues = db.dead_cues(bx, by, speed, n_tracked)
    vote = db.dead_vote(cues)
    have = np.isfinite(bx)

    lines = ["source: {}   frames: {}".format(args.source, n),
             "ball visible: {:.1%}".format(have.mean())]
    if args.source == "tactical":
        graded = np.zeros(n, dtype=bool)
        for _h, a, b in half_video_blocks(GAME):
            graded[a:b + 1] = True
        lines.append("ball visible on graded frames: {:.1%}   (DATA.md: 59.3%)".format(
            have[graded].mean()))
    else:
        graded = np.ones(n, dtype=bool)
    lines.append("")
    lines.append("cue                share of graded frames")
    for name in cues.columns:
        lines.append("  {:<16} {:6.2%}".format(name, cues[name].to_numpy()[graded].mean()))
    lines.append("  vote > 0         {:6.2%}".format((vote[graded] > 0).mean()))
    lines.append("")
    lines.append("co-occurrence: P(row cue | column cue) over graded frames")
    cc = cues.to_numpy()[graded].astype(float)
    names = list(cues.columns)
    lines.append("  " + " " * 16 + "".join("{:>16}".format(c) for c in names))
    for i, a in enumerate(names):
        row = [(cc[:, i] * cc[:, j]).sum() / max(cc[:, j].sum(), 1) for j in range(len(names))]
        lines.append("  {:<16}".format(a) + "".join("{:16.2f}".format(v) for v in row))
    lines.append("")
    lines.append("frame-level mean player speed (m/s) quantiles 5/25/50/75/95: " +
                 " ".join("{:.2f}".format(v) for v in np.nanquantile(speed[graded],
                                                                     [.05, .25, .5, .75, .95])))
    lines.append("calm_speed currently {} m/s sits at the {:.0%} quantile".format(
        db.DEAD_P["calm_speed"], np.nanmean(speed[graded] < db.DEAD_P["calm_speed"])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # timelines around the longest positive-vote stretches (chosen by rule)
    runs = longest_runs((vote > 0) & graded, 3)
    for k, (a, b) in enumerate(runs, 1):
        lo, hi = max(0, a - 15 * db.FPS), min(n, b + 15 * db.FPS)
        t = (np.arange(lo, hi) - lo) / db.FPS
        fig, ax = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
        ax[0].plot(t, bx[lo:hi], ".", ms=2, label="ball x (m)"); ax[0].plot(t, by[lo:hi], ".", ms=2, label="ball y (m)")
        ax[0].legend(loc="upper right"); ax[0].set_ylabel("ball (m)")
        ax[1].plot(t, db.edge_distance(bx, by)[lo:hi], ".", ms=2); ax[1].axhline(0, c="k", lw=.5)
        ax[1].set_ylabel("edge dist (m)")
        ax[2].plot(t, speed[lo:hi]); ax[2].axhline(db.DEAD_P["calm_speed"], c="r", lw=.7)
        ax[2].set_ylabel("mean player speed")
        for i, c in enumerate(cues.columns):
            ax[3].fill_between(t, i, i + cues[c].to_numpy()[lo:hi] * .8, step="mid")
        ax[3].set_yticks(np.arange(len(cues.columns)) + .4); ax[3].set_yticklabels(cues.columns)
        ax[3].set_xlabel("seconds from frame {}".format(lo))
        fig.suptitle("{}: dead-ball vote positive, frames {}-{}".format(args.source, a, b))
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "timeline_{}_{}.png".format(args.source, k)), dpi=110)
        plt.close(fig)
        lines.append("timeline {}: vote>0 run frames {}-{} ({:.1f} s)".format(k, a, b, (b - a) / db.FPS))

    # Smart Tagging alignment (tactical only)
    if args.source == "tactical":
        from load_smart_tagging import load as load_tags, check_time_base
        events = load_tags()
        tags = db.tag_table(events)
        rep = check_time_base(events)
        lines.append("")
        lines.append("Smart Tagging: {} instances; time base agrees with kickoffs: {}".format(
            len(tags), rep["agrees"]))
        lines.append("codes: " + ", ".join("{} {}".format(k, v) for k, v in tags["code"].value_counts().items()))
        lines.append("restart types: " + ", ".join("{} {}".format(k, v) for k, v in
                                                   db.restart_table(tags)["type"].value_counts().items()))
        restarts = db.restart_table(tags)
        for anchor in ("f0", "f1"):
            prof = db.cue_profile(cues, restarts, anchor=anchor)
            prof.to_csv(os.path.join(args.out, "cue_profile_{}.csv".format(anchor)))
        lines.append("cue_profile_f0.csv / cue_profile_f1.csv written: mean cue value per second around each "
                     "restart type -- read them to see where tags sit relative to stoppages")
        forced = db.forced_dead_mask(n)
        lines.append("forced D (organiser-published second-half pre-kickoff): {} frames".format(int(forced.sum())))
        lines.append("  cue behaviour inside them: " + ", ".join(
            "{} {:.0%}".format(c, cues[c].to_numpy()[forced].mean()) for c in cues.columns))

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "report_{}.txt".format(args.source)), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()

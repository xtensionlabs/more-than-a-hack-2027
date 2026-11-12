"""pipeline.py -- the shared, generic part of the pipeline (Elvis owns this file).

Cleaning, the home/away lean (Terer's part), the Viterbi decoder, the tidy-up and the checks.
The dead-ball cues and the D score live in deadball.py. run_full.py ties them together.
Every number in P is a starting value that has not been validated against any ground truth.
"""
import numpy as np
import pandas as pd

FPS, L, W = 25, 105.3, 68.0
P = dict(
    ball_max_speed=45.0,       # m/s, faster = tracker glitch
    ball_gap_fill=12,          # frames (0.5 s): fill only ball gaps up to this
    player_median=5,           # frames of median smoothing on players
    ctrl_radius=3.0,           # m: nearest player must be this close to count as controlling
    tau=1.5,                   # m: how fast the home/away lean saturates
    pen_switch=20.0, pen_to_dead=10.0, pen_from_dead=6.0,   # decoder costs (about 50 x seconds of evidence)
    min_poss_secs=1.5, min_dead_secs=2.0,
)


def fill_short_gaps(x, max_gap):
    s = pd.Series(x)
    isn = s.isna()
    grp = (isn != isn.shift()).cumsum()
    run = isn.groupby(grp).transform("sum")
    filled = s.interpolate(limit_area="inside")
    return np.where(isn & (run <= max_gap), filled, s)


def clean_ball(bx, by, p=P):
    bx, by = bx.copy(), by.copy()
    sp = np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan)) * FPS
    bad = sp > p["ball_max_speed"]
    bx[bad] = np.nan
    by[bad] = np.nan
    observed = ~np.isnan(bx)
    bx = fill_short_gaps(bx, p["ball_gap_fill"])
    by = fill_short_gaps(by, p["ball_gap_fill"])
    return bx, by, observed


def smooth_players(xy, p=P):  # xy: (N, K, 2)
    out = np.empty_like(xy)
    for k in range(xy.shape[1]):
        out[:, k, :] = (
            pd.DataFrame(xy[:, k, :])
            .rolling(p["player_median"], center=True, min_periods=1)
            .median()
            .values
        )
    return out


def nearest_dist(bx, by, xy, slots):
    d = np.hypot(xy[:, slots, 0] - bx[:, None], xy[:, slots, 1] - by[:, None])
    return np.where(np.isnan(d), np.inf, d).min(axis=1)


def home_away_lean(bx, by, xs, home, away, p=P):
    """Per-frame lean toward the team whose nearest player is clearly closer to the ball.

    Positive = home, negative = away, 0 where the ball is missing or nobody is within
    ctrl_radius of it (so the decoder keeps the previous team). bx, by in metres (cleaned),
    xs (N, K, 2) smoothed player positions in metres, home / away = slot index arrays.
    """
    have = np.isfinite(bx) & np.isfinite(by)
    b0x, b0y = np.nan_to_num(bx, nan=-999), np.nan_to_num(by, nan=-999)
    d_home = np.where(have, nearest_dist(b0x, b0y, xs, home), np.inf)
    d_away = np.where(have, nearest_dist(b0x, b0y, xs, away), np.inf)
    ctrl = have & (np.minimum(d_home, d_away) < p["ctrl_radius"])
    with np.errstate(invalid="ignore"):
        lean = np.clip((d_away - d_home) / p["tau"], -4, 4)
    return np.where(ctrl & np.isfinite(lean), lean, 0.0)


def viterbi(S, p=P):
    pen = np.array(
        [
            [0, p["pen_switch"], p["pen_to_dead"]],
            [p["pen_switch"], 0, p["pen_to_dead"]],
            [p["pen_from_dead"], p["pen_from_dead"], 0],
        ],
        float,
    )
    T = len(S)
    back = np.zeros((T, 3), np.int8)
    v = S[0].copy()
    for t in range(1, T):
        cand = v[:, None] - pen
        back[t] = cand.argmax(0)
        v = cand.max(0) + S[t]
    path = np.empty(T, np.int8)
    path[-1] = v.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def runs(a):
    a = np.asarray(a)
    idx = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1]
    return list(zip(idx, np.append(idx[1:], len(a)), a[idx]))


def absorb_short(path, p=P):
    """Merge every run shorter than its minimum into the label before it, repeating until no
    short run is left (the first version stopped after 10 merges and left hundreds behind)."""
    path = path.copy()
    while True:
        changed = False
        for s, e, v in runs(path):
            lim = p["min_dead_secs"] if v == 2 else p["min_poss_secs"]
            if (e - s) < lim * FPS and s > 0:
                path[s:e] = path[s - 1]
                changed = True
        if not changed:
            return path


def decode_half(S, p=P):
    return absorb_short(viterbi(S, p), p)


def to_letters(path):
    return np.array(list("HAD"))[path]


def write_labels(n_total, blocks, decoded):
    labels = np.full(n_total, "X", dtype="<U1")
    for (half, a, b), lab in zip(blocks, decoded):
        labels[a : b + 1] = lab
    return labels


def validate(labels, n_total=199_319, blocks=()):
    assert len(labels) == n_total, f"expected {n_total} rows, got {len(labels)}"
    assert set(np.unique(labels)) <= set("HADX"), "illegal label"
    for _, a, b in blocks:
        assert (labels[a : b + 1] != "X").all(), "X inside a graded half"
    print("OK", dict(zip(*np.unique(labels, return_counts=True))))


def spell_list(lab):
    return [(s, e, v) for s, e, v in runs(np.asarray(lab)) if v in ("H", "A")]


def approx_spell_f1(pred, ref, thr=0.5):
    P_, R_ = spell_list(pred), spell_list(ref)
    used = set()
    tp = 0
    for s, e, v in R_:
        best, bj = 0, None
        for j, (s2, e2, v2) in enumerate(P_):
            if v2 != v or j in used:
                continue
            inter = max(0, min(e, e2) - max(s, s2))
            iou = inter / (max(e, e2) - min(s, s2))
            if iou > best:
                best, bj = iou, j
        if best >= thr:
            tp += 1
            used.add(bj)
    prec, rec = tp / max(len(P_), 1), tp / max(len(R_), 1)
    return 0 if tp == 0 else 2 * prec * rec / (prec + rec)


def summary(labels, blocks):
    for half, a, b in blocks:
        seg = labels[a : b + 1]
        shares = {k: round(float((seg == k).mean()), 3) for k in "HAD"}
        lens = np.array([e - s for s, e, _ in spell_list(seg)]) / FPS
        print(
            f"half {half}: {shares} possessions={len(lens)} "
            f"median={np.median(lens):.1f}s share<2s={(lens < 2).mean():.0%}"
        )


def mask_ball(bx, by, frac, rng, chunk=50):
    """Hide the ball in whole 2-second chunks (real gaps are runs, not single frames)."""
    hide = np.repeat(rng.random(len(bx) // chunk + 1) < frac, chunk)[: len(bx)]
    return np.where(hide, np.nan, bx), np.where(hide, np.nan, by)


def sweep(name, values, run):
    """Change ONE parameter, re-run, and see how much the answer moves.
    run(p) must return the H/A/D letters for one half."""
    ref = run(P)
    for v in values:
        lab = run({**P, name: v})
        print(
            f"{name}={v:<6} D={np.mean(lab == 'D'):.3f} "
            f"possessions={len(spell_list(lab)):4d} "
            f"same_as_current={approx_spell_f1(lab, ref):.2f}"
        )

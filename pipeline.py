"""pipeline.py - Xtenders Phase 1 possession pipeline.

Functions come from the Phase 1 Build Guide (Part Four to Part Six).
Constants and parameter values marked GUESS are starting values only.
They have NOT been tested on the real match and must be tuned (Part Six).
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants
L, W = 105.3, 68.0  # pitch length and width in metres (positions run 0..1)
FPS = 25

# Maps the lower-case team name in a Smart Tagging row to a state:
# 0 = home (H), 1 = away (A). Fill it in run_pipeline.py AFTER printing the
# real team names from the tag file. Empty = tags unused.
TEAM_TO_STATE = {}

P = {
    # inputs
    "ball_max_speed": 45.0,  # m/s, faster jumps are removed (guide)
    "ball_gap_fill": 12,  # frames, longest ball gap filled (guide)
    "player_median": 5,  # frames, median filter width (GUESS)
    # evidence
    "still_speed": 0.5,  # m/s, ball counts as still below this (GUESS)
    "still_secs": 2.0,  # guide
    "lost_secs": 2.0,  # guide: missing 2+ seconds
    "edge_dist": 3.0,  # m, "near a line" (GUESS)
    "calm_secs": 3.0,  # guide: players calm for 3 seconds
    "calm_speed": 1.3,  # m/s, guide
    # scores
    "ctrl_radius": 3.0,  # m, guide
    "tau": 1.5,  # m, guide
    "dead_bias": -2.0,  # guide
    "w_out": 4.0,
    "w_still": 2.0,
    "w_lost": 2.5,
    "w_calm": 1.5,  # guide
    # decoder
    "pen_switch": 20.0,
    "pen_to_dead": 10.0,
    "pen_from_dead": 6.0,  # guide
    # clean-up
    "min_poss_secs": 1.5,
    "min_dead_secs": 2.0,  # guide
}


# ------------------------------------------------------ Step 2: clean (Elvis)
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


# ------------------------------------------------- Step 3: features (Terer/Azzaliah)
def nearest_dist(bx, by, xy, slots):
    d = np.hypot(xy[:, slots, 0] - bx[:, None], xy[:, slots, 1] - by[:, None])
    return np.where(np.isnan(d), np.inf, d).min(axis=1)


def run_length_of(mask):
    s = pd.Series(mask.astype(int))
    grp = (s != s.shift()).cumsum()
    return (s.groupby(grp).cumsum() * s).values


def make_features(bx, by, speed_mean, xy, home, away, p=P):
    f = {}
    have = ~np.isnan(bx)
    bx0, by0 = np.nan_to_num(bx, nan=-999), np.nan_to_num(by, nan=-999)
    f["d_home"] = np.where(have, nearest_dist(bx0, by0, xy, home), np.inf)
    f["d_away"] = np.where(have, nearest_dist(bx0, by0, xy, away), np.inf)
    f["have"] = have
    bs = np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan)) * FPS
    f["ball_speed"] = (
        pd.Series(bs).rolling(FPS, center=True, min_periods=5).median().values
    )
    margin = 0.5
    f["out"] = have & (
        (bx < -margin) | (bx > L + margin) | (by < -margin) | (by > W + margin)
    )
    still = have & (np.nan_to_num(f["ball_speed"], nan=99) < p["still_speed"])
    f["still"] = run_length_of(still) >= p["still_secs"] * FPS
    lx = pd.Series(bx).ffill().values
    ly = pd.Series(by).ffill().values  # last seen position
    edge = np.minimum.reduce([lx, L - lx, ly, W - ly])
    f["lost_near_line"] = (run_length_of(~have) >= p["lost_secs"] * FPS) & (
        edge < p["edge_dist"]
    )
    f["calm"] = (
        pd.Series(speed_mean)
        .rolling(int(p["calm_secs"] * FPS), center=True, min_periods=10)
        .mean()
        .values
        < p["calm_speed"]
    )
    return pd.DataFrame(f)


# ------------------------------------------------------ Step 4: scores
def make_scores(F, p=P, tag_bonus=None):
    ctrl = F.have & (np.minimum(F.d_home, F.d_away) < p["ctrl_radius"])
    lean = np.clip((F.d_away - F.d_home) / p["tau"], -4, 4)
    lean = np.where(ctrl & np.isfinite(lean), lean, 0.0)
    dead = (
        p["dead_bias"]
        + p["w_out"] * F.out
        + p["w_still"] * F.still
        + p["w_lost"] * F.lost_near_line
        + p["w_calm"] * F.calm
    )
    S = np.column_stack([lean / 2, -lean / 2, dead])
    if tag_bonus is not None:
        S[:, 0] += tag_bonus[:, 0]
        S[:, 1] += tag_bonus[:, 1]
    return S


# ------------------------------------------------------ Step 5: Smart Tagging
def tag_frames(ev):
    ev = ev.copy()
    ev["code_u"] = ev["code"].str.upper()
    ev["team_l"] = ev["Team"].astype(str).str.lower()
    ev["f0"] = (ev["timestamp"].dt.total_seconds() * FPS).round().astype(int)
    ev["f1"] = (ev["end_timestamp"].dt.total_seconds() * FPS).round().astype(int)
    return ev


def tag_bonus_array(ev, n, sign_by_code, w=0.3):
    """sign_by_code: +1 if the tag's team is the team IN possession, -1 if it is
    the team DEFENDING, 0 to ignore. Verify these with tag_agreement() first."""
    B = np.zeros((n, 3))
    for r in ev.itertuples():
        st = TEAM_TO_STATE.get(r.team_l)
        sg = sign_by_code.get(r.code_u, 0)
        if st is None or sg == 0:
            continue
        B[max(r.f0, 0) : min(r.f1, n), st if sg > 0 else 1 - st] += w
    return B


def tag_agreement(ev, labels):
    out = {}
    for code, g in ev.groupby("code_u"):
        agree = tot = 0
        for r in g.itertuples():
            st = TEAM_TO_STATE.get(r.team_l)
            if st is None:
                continue
            seg = labels[r.f0 : r.f1]
            seg = seg[(seg == "H") | (seg == "A")]
            agree += (seg == "HA"[st]).sum()
            tot += len(seg)
        if tot:
            out[code] = round(agree / tot, 2)
    return out


# ------------------------------------------------------ Steps 6-7: decode (Elvis)
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
    """Merge possessions shorter than min_poss_secs and stoppages shorter than
    min_dead_secs into the label before them.
    CHANGE FROM THE GUIDE: the guide's loop stopped after 10 merges in total,
    which is far too few for a whole half. This version repeats until nothing
    is left to merge (capped for safety)."""
    path = path.copy()
    for _ in range(100_000):
        changed = False
        for s, e, v in runs(path):
            lim = p["min_dead_secs"] if v == 2 else p["min_poss_secs"]
            if (e - s) < lim * FPS and s > 0:
                path[s:e] = path[s - 1]
                changed = True
                break
        if not changed:
            break
    return path


def decode_half(S, p=P):
    return absorb_short(viterbi(S, p), p)


def to_letters(path):
    return np.array(list("HAD"))[path]


def decode_frames(bx, by, sm, xs, home, away, tag_bonus=None, p=P):
    """Everything from cleaned arrays to H/A/D letters, for ONE half."""
    F = make_features(bx, by, sm, xs, home, away, p)
    return to_letters(decode_half(make_scores(F, p, tag_bonus), p))


# ------------------------------------------------------ Step 8: write + check (Elvis)
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


# ------------------------------------------------------ checks and tuning tools
def spell_list(lab):
    return [(s, e, v) for s, e, v in runs(np.asarray(lab)) if v in ("H", "A")]


def summary(labels, blocks):
    for half, a, b in blocks:
        seg = labels[a : b + 1]
        shares = {k: round(float((seg == k).mean()), 3) for k in "HAD"}
        lens = np.array([e - s for s, e, _ in spell_list(seg)]) / FPS
        print(
            f"half {half}: {shares} possessions={len(lens)} "
            f"median={np.median(lens):.1f}s share<2s={(lens < 2).mean():.0%}"
        )


def approx_spell_f1(pred, ref, thr=0.5):
    """Development stand-in only. NOT necessarily the official matching rule."""
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


def mask_ball(bx, by, frac, rng, chunk=50):
    """Hide the ball in whole 2-second chunks (real gaps are runs, not single frames)."""
    hide = np.repeat(rng.random(len(bx) // chunk + 1) < frac, chunk)[: len(bx)]
    return np.where(hide, np.nan, bx), np.where(hide, np.nan, by)


def sweep(name, values, run, base=P):
    """Change ONE parameter, re-run, see how much the answer moves.
    run(p) must return the H/A/D letters for one half."""
    ref = run(base)
    for v in values:
        lab = run({**base, name: v})
        print(
            f"{name}={v:<6} D={np.mean(lab == 'D'):.3f} "
            f"possessions={len(spell_list(lab)):4d} "
            f"same_as_current={approx_spell_f1(lab, ref):.2f}"
        )

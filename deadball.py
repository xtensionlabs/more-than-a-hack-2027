"""deadball.py -- when has play stopped?  (Azzaliah's part of the pipeline)

Everything here is a plain function of arrays, so it runs on the tactical ATD (what gets
submitted) and on the broadcast ATD (useful for looking at, never aligned to the submission).
Constants named in the docstrings come from the organisers' own files and are cited.

What the data says about dead ball (all measured on the delivered files, see the work pack):
  * the ATD has no ball-state field at all (kloppy returns ball_state = None), so "dead" has to
    be inferred from where the ball is, whether it moves, and what the players are doing;
  * the ball does appear outside the lines (3.6% of visible broadcast frames);
  * the ball is missing in long runs, not flicker (99.5% of missing frames sit in runs >= 2 s).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks  # noqa: E402

FPS = GAMES[GAME]["fps"]
L, W = GAMES[GAME]["pitch"]          # 105.3 x 68.0 m, measured from the corrected tracking

# Starting values, not facts. Every one is a general rule in seconds / metres, none is tied to
# a frame of this match. Merge into the shared parameter dictionary P (Elvis owns that).
DEAD_P = dict(
    out_margin=0.5,                  # m beyond a line before the ball counts as outside
    still_range=1.0, still_secs=2.0, still_min_obs=10,
    lost_secs=2.0, edge_dist=5.0, lost_min_players=10,
    calm_speed=1.3, calm_secs=3.0,
    # Weights = LLR_on - LLR_off measured by weak_labels.py on the tactical data (restart tags as
    # weak evidence, no ground truth). dead_bias = log prior odds of dead (ASSUMED 35% dead,
    # -0.62: a team judgement, not measured) + the summed LLR_off (-1.53) = -2.15.
    # w_gapcalm is deliberately HALF its measured 1.2: at full weight it turned slow live play
    # (ball hidden among players, players calm) into D. At 0.6 it only tips a tie, e.g. before a
    # kick-off after a goal where the ball was last seen far from any line.
    w_out=2.1, w_still=0.4, w_lost=2.6, w_calm=1.1, w_gapcalm=0.6, dead_bias=-2.1,
    # Restart prior: a set-piece tag starts about 3.5 s BEFORE the restart (88 and 82 frames on the
    # two published kick-off tags; goal kicks, free kicks and corners move at +2..+4 s in the
    # data). The stoppage runs from some seconds before the tag start to that restart.
    restart_pre_s=8.0, restart_post_s=3.5, w_restart=1.0,
    # After the restart the ball is live again, but a ball that left the tracker near a line stays
    # unseen for a few seconds (a corner is measured at ~28 m/s two to four seconds after the tag
    # start while the ball is still missing). Counter the gap cue there with a live term.
    restart_live_from_s=5.5, restart_live_to_s=10.0, w_live=2.0,
)
FORCE = 50.0                         # per-frame vote that no other evidence can outweigh

# Organiser-published kickoff frames (FAQ.md, DATA.md and starter/load_smart_tagging.py):
# the first half is graded from its kickoff; the second half is graded from frame 110,808 but
# kicks off at 113,742, and FAQ.md states those 2,934 graded frames are all D.
KICKOFF_VIDEO_FRAMES = {1: 11185, 2: 113742}


# ----------------------------------------------------------------------------- run helpers
def run_length_of(mask):
    """Length so far of the run of True each frame is in (0 where False). 1 at a run's start."""
    mask = np.asarray(mask, dtype=bool)
    index = np.arange(len(mask))
    last_false = np.maximum.accumulate(np.where(~mask, index, -1))
    return np.where(mask, index - last_false, 0)


def run_total_length(mask):
    """Full length of the run of True each frame belongs to (0 where False)."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(len(mask), dtype=int)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    for start, stop in zip(edges[::2], edges[1::2]):
        out[start:stop] = stop - start
    return out


# --------------------------------------------------------------------------------- the cues
def edge_distance(bx, by):
    """Metres from the ball to the nearest touchline or goal line; negative when outside."""
    return np.minimum.reduce([bx, L - bx, by, W - by])


def dead_cues(bx, by, speed_mean, n_tracked=None, p=DEAD_P):
    """Five boolean dead-ball cues per frame. bx, by in metres, NaN where the ball is missing.

    out             ball is more than out_margin metres outside the lines
    still           ball stays within still_range metres for still_secs (centred window), judged
                    from the frames where it IS seen, so gaps do not break the test
    lost_near_line  ball missing for lost_secs or more AND the last or the next time it is seen
                    it is within edge_dist of a line -- both ends of the gap are real data --
                    AND at least lost_min_players players are tracked in that frame. If nobody is
                    tracked the camera or tracker has lost the pitch, which says nothing about the
                    ball (on the broadcast file, 74% of the frames inside gaps of 2 s or more
                    have fewer than 10 players tracked)
    calm            players' mean speed stays under calm_speed for calm_secs
    gap_calm        ball missing for lost_secs or more while the players are calm and enough are
                    tracked: nobody is moving and the ball is not in play (e.g. before a kick-off
                    after a goal, where the ball was last seen far from any line)
    """
    bx, by = np.asarray(bx, float), np.asarray(by, float)
    have = np.isfinite(bx) & np.isfinite(by)
    margin = p["out_margin"]
    out = have & ((bx < -margin) | (bx > L + margin) | (by < -margin) | (by > W + margin))

    window = int(round(p["still_secs"] * FPS))
    roll_x = pd.Series(np.where(have, bx, np.nan)).rolling(window, center=True, min_periods=1)
    roll_y = pd.Series(np.where(have, by, np.nan)).rolling(window, center=True, min_periods=1)
    spread = np.maximum((roll_x.max() - roll_x.min()).to_numpy(),
                        (roll_y.max() - roll_y.min()).to_numpy())
    seen = pd.Series(have.astype(float)).rolling(window, center=True, min_periods=1).sum().to_numpy()
    still = have & (spread <= p["still_range"]) & (seen >= p["still_min_obs"])

    edge = np.where(have, edge_distance(bx, by), np.nan)
    last_edge = pd.Series(edge).ffill().to_numpy()
    next_edge = pd.Series(edge).bfill().to_numpy()
    long_gap = run_total_length(~have) >= p["lost_secs"] * FPS
    near = (last_edge < p["edge_dist"]) | (next_edge < p["edge_dist"])    # NaN compares False
    lost_near_line = long_gap & near
    if n_tracked is not None:
        lost_near_line &= np.asarray(n_tracked) >= p["lost_min_players"]

    window = int(round(p["calm_secs"] * FPS))
    smooth = pd.Series(speed_mean).rolling(window, center=True, min_periods=max(1, window // 4))
    calm = (smooth.mean() < p["calm_speed"]).to_numpy()

    gap_calm = long_gap & calm
    if n_tracked is not None:
        gap_calm &= np.asarray(n_tracked) >= p["lost_min_players"]

    return pd.DataFrame({"out": out, "still": still, "lost_near_line": lost_near_line,
                         "calm": calm, "gap_calm": gap_calm})


def dead_vote(cues, p=DEAD_P, tag_bonus=None, force_dead=None):
    """The D score per frame: starts at dead_bias, each active cue adds its weight."""
    vote = (p["dead_bias"] + p["w_out"] * cues["out"].to_numpy()
            + p["w_still"] * cues["still"].to_numpy()
            + p["w_lost"] * cues["lost_near_line"].to_numpy()
            + p["w_calm"] * cues["calm"].to_numpy()
            + p["w_gapcalm"] * cues["gap_calm"].to_numpy())
    if tag_bonus is not None:
        vote = vote + tag_bonus
    if force_dead is not None:
        vote = np.where(force_dead, FORCE, vote)
    return vote


# ------------------------------------------------------------- facts the organisers published
def forced_dead_mask(n_frames, game=GAME, kickoffs=KICKOFF_VIDEO_FRAMES):
    """Graded frames before a half's kickoff. For this match: 110,808 .. 113,741 (2,934 frames).

    FAQ.md: "Those 2,934 frames are graded and all of them are D." Computed from the published
    half boundaries and kickoff frames, nothing estimated. The first half has none because it is
    graded from the kickoff itself.
    """
    mask = np.zeros(n_frames, dtype=bool)
    for half, first, _last in half_video_blocks(game):
        kickoff = kickoffs.get(half)
        if kickoff is not None and kickoff > first:
            mask[first:kickoff] = True
    return mask


# ------------------------------------------------------------------------------ Smart Tagging
TEAM_TO_STATE = {"barcelona": 0, "atletic club": 1, "athletic club": 1}   # 0 = H, 1 = A
RESTART_TYPES = ("throw-in", "corner kick", "goal kick", "free kick",
                 "kick off (after goal)", "kick off (start)")


def tag_table(events):
    """Smart Tagging as a table, from the organisers' loader: starter.load_smart_tagging.load().

    Columns: code (upper), type (lower), half, team (0 home, 1 away, NaN for N/A), f0, f1 (video
    frames, f1 is the first frame after the phase). Compare case-insensitively, as DATA.md warns.
    """
    rows = []
    for event in events:
        tags = event["tags"]
        rows.append({
            "code": (event["code"] or "").strip().upper(),
            "type": (tags.get("Type") or "").strip().lower(),
            "half": tags.get("Half"),
            "team": TEAM_TO_STATE.get((tags.get("Team") or "").strip().lower(), np.nan),
            "f0": int(event["start_frame"]),
            "f1": int(event["end_frame"]),
        })
    return pd.DataFrame(rows, columns=["code", "type", "half", "team", "f0", "f1"])


def restart_table(tags):
    """The set-piece instances, in time order."""
    return tags[tags["type"].isin(RESTART_TYPES)].sort_values("f0").reset_index(drop=True)


def cue_profile(cues, restarts, before_s=20, after_s=10, anchor="f0"):
    """Average of each cue, second by second around every restart tag, per restart type.

    This is how to learn -- from the data, with no labels -- where a restart tag sits relative
    to the stoppage it ends. anchor is "f0" (tag start) or "f1" (tag end). Rows are offsets in
    seconds (negative = before the anchor).
    """
    offsets = np.arange(-before_s, after_s + 1)
    n = len(cues)
    out = {}
    for rtype, group in restarts.groupby("type"):
        for name in cues.columns:
            values = cues[name].to_numpy(dtype=float)
            matrix = []
            for frame in group[anchor].to_numpy():
                idx = frame + offsets * FPS
                row = np.full(len(offsets), np.nan)
                inside = (idx >= 0) & (idx < n)
                row[inside] = values[idx[inside]]
                matrix.append(row)
            out[(rtype, name)] = np.nanmean(np.array(matrix), axis=0)
    return pd.DataFrame(out, index=offsets)


def restart_prior(restarts, n_frames, before_s, after_s, types=RESTART_TYPES, anchor="f0"):
    """1.0 on frames from before_s before to after_s after each chosen restart, else 0.

    Multiply by a small weight and pass as tag_bonus to dead_vote. Choose before_s / after_s
    from cue_profile output, not by eye on single frames.
    """
    prior = np.zeros(n_frames)
    for frame in restarts.loc[restarts["type"].isin(types), anchor].to_numpy():
        prior[max(0, frame - int(before_s * FPS)):min(n_frames, frame + int(after_s * FPS) + 1)] = 1.0
    return prior


def restart_bonus(tags, n_frames, p=DEAD_P):
    """Net D bonus per frame from the restart tags (kick-offs included). Pass as tag_bonus.

    +w_restart from restart_pre_s before the tag start to restart_post_s after it (the stoppage up
    to the restart), then -w_live from restart_live_from_s to restart_live_to_s after the tag start
    (play is live again; stops a gap cue from holding D after the restart).
    """
    restarts = restart_table(tags)
    dead = restart_prior(restarts, n_frames, p["restart_pre_s"], p["restart_post_s"])
    live = restart_prior(restarts, n_frames, -p["restart_live_from_s"], p["restart_live_to_s"])
    return p["w_restart"] * dead - p["w_live"] * live


def tag_bonus_array(tags, n_frames, sign_by_code, w=0.3):
    """(n, 2) bonus for H and A from the Team tag. sign_by_code: +1 where the tag's team is the
    team in possession, -1 where it is the team defending, 0 to ignore. Fill it from
    tag_agreement(), never from assumption. N/A teams (36 instances) are skipped."""
    bonus = np.zeros((n_frames, 2))
    for row in tags.itertuples():
        sign = sign_by_code.get(row.code, 0)
        if sign == 0 or not np.isfinite(row.team):
            continue
        side = int(row.team) if sign > 0 else 1 - int(row.team)
        bonus[max(row.f0, 0):min(row.f1, n_frames), side] += w
    return bonus


def tag_agreement(tags, labels):
    """Per code: how often the tag's team matches your H/A label while the ball is alive.
    Near 1.0 = tag team is the possessing team, near 0.0 = the defending team, near 0.5 = noise."""
    labels = np.asarray(labels)
    out = {}
    for code, group in tags.groupby("code"):
        agree = total = 0
        for row in group.itertuples():
            if not np.isfinite(row.team):
                continue
            seg = labels[row.f0:row.f1]
            seg = seg[(seg == "H") | (seg == "A")]
            agree += int((seg == "HA"[int(row.team)]).sum())
            total += len(seg)
        if total:
            out[code] = round(agree / total, 3)
    return out

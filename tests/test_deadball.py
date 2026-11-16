"""Tests for deadball.py. Two kinds, kept apart on purpose:

  * real-data tests read the delivered ATD files and compare against numbers the organisers
    published in DATA.md / FAQ.md. They skip themselves when the files are not in data/.
  * logic tests use a few hand-written frames (test fixtures, not data) to pin down edge cases
    that real data cannot isolate.
"""
import os

import numpy as np
import pandas as pd
import pytest

import deadball as db
from games import GAME, asset_paths, half_video_blocks
from rawatd import load_raw_atd, mean_player_speed, n_slots_from_meta


def _have(source):
    paths = asset_paths(source=source)
    return os.path.isfile(paths["atd_raw"]) and os.path.isfile(paths["atd_meta"])


def _load(source):
    paths = asset_paths(source=source)
    return load_raw_atd(paths["atd_raw"], n_slots_from_meta(paths["atd_meta"]))


# ------------------------------------------------------------------------- real data
@pytest.mark.skipif(not _have("broadcast"), reason="broadcast ATD not in data/")
def test_broadcast_matches_data_md():
    ids, players, ball = _load("broadcast")
    assert len(ids) == 164_488 and players.shape[1] == 51          # DATA.md table
    assert ids[0] == 1 and ids[-1] == 164_488
    assert round(np.isfinite(ball[:, 0]).mean(), 3) == 0.505       # DATA.md: 50.5% visible


@pytest.mark.skipif(not _have("tactical"), reason="tactical ATD not in data/")
def test_tactical_matches_published_counts():
    ids, players, ball = _load("tactical")
    assert len(ids) == 199_319 and players.shape[1] == 46
    visible = np.isfinite(ball[:, 0])
    assert visible.sum() == 113_488                                # DATA.md, exact
    graded = np.zeros(len(ids), dtype=bool)
    for _h, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    assert graded.sum() == 158_857 and visible[graded].sum() == 94_214


@pytest.mark.skipif(not _have("broadcast"), reason="broadcast ATD not in data/")
def test_cues_on_real_broadcast_are_consistent():
    _ids, players, ball = _load("broadcast")
    bx, by = ball[:, 0] * db.L, ball[:, 1] * db.W
    n_tracked = np.isfinite(players[:, :, 0]).sum(axis=1)
    cues = db.dead_cues(bx, by, mean_player_speed(players[:, :, 2]), n_tracked)
    have = np.isfinite(bx)
    assert len(cues) == len(bx)
    # out is an independent recount
    recount = have & ((bx < -.5) | (bx > db.L + .5) | (by < -.5) | (by > db.W + .5))
    assert (cues["out"].to_numpy() == recount).all()
    # cues that need a visible ball never fire without one, and the gap cue the opposite
    assert not cues["out"].to_numpy()[~have].any() and not cues["still"].to_numpy()[~have].any()
    assert not cues["lost_near_line"].to_numpy()[have].any()
    # the player-count rule removes the camera-away gaps
    assert not cues["lost_near_line"].to_numpy()[n_tracked < 10].any()


def test_kickoff_constants_match_the_organiser_loader():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(db.__file__), "starter"))
    from load_smart_tagging import KICKOFF_VIDEO_FRAMES as official
    assert db.KICKOFF_VIDEO_FRAMES == {1: official["1st Half"], 2: official["2nd Half"]}


def test_published_second_half_pre_kickoff_is_2934_frames():
    mask = db.forced_dead_mask(199_319)
    assert mask.sum() == 2_934                                     # FAQ.md
    assert np.flatnonzero(mask)[0] == 110_808 and np.flatnonzero(mask)[-1] == 113_741
    assert not mask[11_185:90_652].any()                           # first half starts at kickoff


# ------------------------------------------------------------------------ logic fixtures
def test_run_helpers():
    m = [0, 1, 1, 0, 1, 1, 1]
    assert run_list(db.run_length_of(m)) == [0, 1, 2, 0, 1, 2, 3]
    assert run_list(db.run_total_length(m)) == [0, 2, 2, 0, 3, 3, 3]
    assert run_list(db.run_length_of([1, 1])) == [1, 2]


def run_list(a):
    return [int(v) for v in a]


def test_out_cue_inside_and_outside():
    bx = np.array([50.0, -0.4, -0.6, db.L + 0.6, 50.0, np.nan])
    by = np.array([30.0, 30.0, 30.0, 30.0, db.W + 1.0, 30.0])
    out = db.dead_cues(bx, by, np.full(6, 3.0))["out"].to_numpy()
    assert out.tolist() == [False, False, True, True, True, False]


def test_still_survives_gaps_and_rejects_motion():
    n = 200
    bx = np.full(n, 40.0); by = np.full(n, 20.0)
    bx[::2] = np.nan                                               # seen on half the frames only
    still = db.dead_cues(bx, by, np.full(n, 3.0))["still"].to_numpy()
    assert still[101] and not still[100]          # judged where the ball is seen, not where it is not
    moving = np.linspace(10, 60, n)
    assert not db.dead_cues(moving, np.full(n, 20.0), np.full(n, 3.0))["still"].to_numpy().any()


def test_lost_near_line_needs_long_gap_edge_and_players():
    n = 400
    bx = np.full(n, 50.0); by = np.full(n, 34.0)
    bx[100:200] = np.nan; by[100:200] = np.nan                     # 4 s gap, ball mid-pitch
    far = db.dead_cues(bx, by, np.full(n, 3.0), np.full(n, 20))["lost_near_line"]
    assert not far.any()
    bx[99] = 1.0                                                   # last seen 1 m from the line
    near = db.dead_cues(bx, by, np.full(n, 3.0), np.full(n, 20))["lost_near_line"].to_numpy()
    assert near[100:200].all() and not near[:100].any() and not near[200:].any()
    nobody = db.dead_cues(bx, by, np.full(n, 3.0), np.zeros(n))["lost_near_line"]
    assert not nobody.any()
    bx2 = bx.copy(); by2 = by.copy()
    bx2[130:135] = 50.0; by2[130:135] = 34.0                       # gap broken: 30 frames < 2 s
    broken = db.dead_cues(bx2, by2, np.full(n, 3.0), np.full(n, 20))["lost_near_line"].to_numpy()
    assert not broken[100:130].any()


def test_gap_calm_needs_gap_calm_and_players():
    n = 400
    bx = np.full(n, 50.0); by = np.full(n, 34.0)                   # last seen mid-pitch: not near a line
    bx[100:200] = np.nan; by[100:200] = np.nan
    calm = np.full(n, 0.5)
    on = db.dead_cues(bx, by, calm, np.full(n, 20))
    assert on["gap_calm"].to_numpy()[130:170].all() and not on["lost_near_line"].any()
    assert not db.dead_cues(bx, by, np.full(n, 3.0), np.full(n, 20))["gap_calm"].any()   # moving
    assert not db.dead_cues(bx, by, calm, np.zeros(n))["gap_calm"].any()                 # nobody tracked


def test_calm_follows_player_speed():
    speed = np.concatenate([np.full(200, 0.5), np.full(200, 3.0)])
    calm = db.dead_cues(np.full(400, 50.0), np.full(400, 30.0), speed)["calm"].to_numpy()
    assert calm[50] and not calm[350]


def test_dead_vote_weights_and_force():
    cues = pd.DataFrame({"out": [True, False], "still": [True, False],
                         "lost_near_line": [False, False], "calm": [True, False],
                         "gap_calm": [True, False]})
    p = db.DEAD_P
    vote = db.dead_vote(cues)
    assert vote[0] == pytest.approx(p["dead_bias"] + p["w_out"] + p["w_still"] + p["w_calm"]
                                    + p["w_gapcalm"])
    assert vote[1] == pytest.approx(p["dead_bias"])
    forced = db.dead_vote(cues, force_dead=np.array([False, True]))
    assert forced[1] == db.FORCE and forced[0] == vote[0]


EVENTS = [  # loader format of starter/load_smart_tagging.py, hand-written fixtures
    {"code": "SET PIECES", "start_frame": 100, "end_frame": 200,
     "tags": {"Team": "N/A", "Half": "1st Half", "Type": "Throw-in"}},
    {"code": "Build Up", "start_frame": 300, "end_frame": 400,
     "tags": {"Team": "Barcelona", "Half": "1st Half"}},
    {"code": "BUILD UP", "start_frame": 500, "end_frame": 600,
     "tags": {"Team": "Atletic Club", "Half": "1st Half"}},
]


def test_tag_table_case_and_na():
    tags = db.tag_table(EVENTS)
    assert tags["code"].tolist() == ["SET PIECES", "BUILD UP", "BUILD UP"]
    assert np.isnan(tags.loc[0, "team"]) and tags.loc[1, "team"] == 0 and tags.loc[2, "team"] == 1
    assert db.restart_table(tags)["type"].tolist() == ["throw-in"]


def test_tag_bonus_and_agreement():
    tags = db.tag_table(EVENTS)
    bonus = db.tag_bonus_array(tags, 700, {"BUILD UP": 1}, w=0.3)
    assert bonus[350].tolist() == [0.3, 0.0] and bonus[550].tolist() == [0.0, 0.3]
    assert bonus[150].tolist() == [0.0, 0.0]                      # N/A is skipped
    flipped = db.tag_bonus_array(tags, 700, {"BUILD UP": -1}, w=0.3)
    assert flipped[350].tolist() == [0.0, 0.3]
    labels = np.array(["X"] * 700)
    labels[300:400] = "H"; labels[500:600] = "H"
    assert db.tag_agreement(tags, labels) == {"BUILD UP": 0.5}


def test_restart_bonus_window():
    events = [{"code": "SET PIECES", "start_frame": 1000, "end_frame": 1875,
               "tags": {"Team": "N/A", "Half": "1st Half", "Type": "Throw-in"}}]
    p = db.DEAD_P
    bonus = db.restart_bonus(db.tag_table(events), 3000)
    pre, post = int(p["restart_pre_s"] * db.FPS), int(p["restart_post_s"] * db.FPS)
    lo, hi = int(p["restart_live_from_s"] * db.FPS), int(p["restart_live_to_s"] * db.FPS)
    assert bonus[1000 - pre] == p["w_restart"] and bonus[1000 + post] == p["w_restart"]
    assert bonus[1000 - pre - 1] == 0 and bonus[1000 + post + 1] == 0
    assert bonus[1000 + lo] == -p["w_live"] and bonus[1000 + hi] == -p["w_live"]
    assert bonus[1000 + hi + 1] == 0 and bonus[2500] == 0


@pytest.mark.skipif(not _have("tactical"), reason="tactical files not in data/")
def test_kickoff_tags_start_about_3_5_s_before_the_kickoff():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(db.__file__), "starter"))
    from load_smart_tagging import load, KICKOFF_VIDEO_FRAMES
    tags = db.tag_table(load())
    start = tags[tags["type"] == "kick off (start)"].sort_values("f0")
    gaps = [kick - f0 for kick, f0 in zip((KICKOFF_VIDEO_FRAMES["1st Half"], KICKOFF_VIDEO_FRAMES["2nd Half"]), start["f0"])]
    assert gaps == [88, 82]                                        # frames; ~3.5 s at 25 fps
    assert (tags[tags["type"].isin(db.RESTART_TYPES)].eval("f1 - f0").isin([125, 875])).all()


def test_cue_profile_and_prior():
    tags = db.tag_table(EVENTS)
    restarts = db.restart_table(tags)
    cues = pd.DataFrame({"out": np.zeros(1000, bool)})
    cues.loc[100:199, "out"] = True
    prof = db.cue_profile(cues, restarts, before_s=2, after_s=2)
    assert prof[("throw-in", "out")].loc[0] == 1.0 and prof[("throw-in", "out")].loc[-2] == 0.0
    prior = db.restart_prior(restarts, 1000, before_s=1, after_s=1)
    assert prior[100] == 1.0 and prior[75] == 1.0 and prior[74] == 0.0 and prior[126] == 0.0


def test_pipeline_absorb_short_leaves_no_short_run():
    import pipeline as pl
    path = np.array(([0] * 60 + [2] + [0] * 60 + [1] * 3 + [0] * 60) * 40)
    assert min(e - s for s, e, _v in pl.runs(pl.absorb_short(path))) >= 37

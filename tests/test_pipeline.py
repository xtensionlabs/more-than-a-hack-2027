import numpy as np
import pytest
from pipeline import *


def test_short_gap_filled_long_gap_kept():
    x = np.array([0, 1, np.nan, np.nan, 4, 5] + [np.nan] * 30 + [40.0])
    out = fill_short_gaps(x, max_gap=3)
    assert out[2] == pytest.approx(2) and out[3] == pytest.approx(3)
    assert np.isnan(out[10])  # long gap untouched


def test_viterbi_ignores_one_frame_flicker():
    S = np.tile([1.0, -1.0, -5.0], (200, 1))  # clearly H
    S[100] = [-1.0, 1.0, -5.0]  # one-frame "A" glitch
    assert (viterbi(S) == 0).all()


def test_viterbi_follows_sustained_change():
    S = np.vstack([np.tile([1, -1, -5.0], (100, 1)), np.tile([-1, 1, -5.0], (100, 1))])
    path = viterbi(S)
    assert (path[:95] == 0).all() and (path[-95:] == 1).all()


def test_absorb_short_removes_tiny_runs():
    path = np.array([0] * 100 + [1] * 5 + [0] * 100)
    assert (absorb_short(path) == 0).all()


def test_absorb_short_handles_many_tiny_runs():
    # more than 10 tiny runs: the guide's original loop would leave some behind
    parts = [np.array([0] * 100)]
    for _ in range(30):
        parts += [np.array([1] * 5), np.array([0] * 100)]
    path = np.concatenate(parts)
    assert (absorb_short(path) == 0).all()


def test_validate_catches_wrong_length_and_x_in_half():
    good = np.array(list("HHAD" * 5))
    validate(good, 20, [(1, 0, 19)])
    with pytest.raises(AssertionError):
        validate(good, 21)
    bad = good.copy()
    bad[3] = "X"
    with pytest.raises(AssertionError):
        validate(bad, 20, [(1, 0, 19)])


def test_spell_f1_perfect_and_swapped():
    ref = np.array(list("H" * 100 + "A" * 100))
    assert approx_spell_f1(ref, ref) == 1.0
    assert approx_spell_f1(np.where(ref == "H", "A", "H"), ref) == 0


def test_end_to_end_runs_on_synthetic_data():
    rng = np.random.default_rng(0)
    n = 1500
    xy = rng.uniform(0, 1, (n, 6, 2)) * [L, W]
    home, away = np.array([0, 1, 2]), np.array([3, 4, 5])
    bx = np.full(n, 20.0)
    by = np.full(n, 30.0)
    xy[:, 0, 0], xy[:, 0, 1] = 20.5, 30.0  # home player on the ball
    bx[600:700] = np.nan
    sm = np.full(n, 3.0)
    lab = decode_frames(bx, by, sm, xy, home, away)
    assert len(lab) == n and set(np.unique(lab)) <= set("HAD")
    assert (lab == "H").mean() > 0.8

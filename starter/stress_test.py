"""Run label-free synthetic and tracking-dropout checks on the current candidate model.

The report measures structural stability only. It does not load ground truth or calculate a
competition score, and its output is written outside the repository.
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks
from starter.dead_ball_decoder import (
    decode as decode_with_dead_ball,
    load_features,
    midpoint_missing_ball_handovers,
    rolling_count_by_half,
    stoppage_evidence,
    viterbi_three_state,
)
from starter.feature_audit import require_output_outside_repository
from starter.load_smart_tagging import load as load_smart_tagging
from starter.possession_decoder import viterbi_decode

HOME, AWAY, DEAD = 0, 1, 2
BALL_FEATURES = (
    "ball_seen",
    "ball_x_norm",
    "ball_y_norm",
    "ball_outside_pitch",
    "home_nearest_distance_m",
    "away_nearest_distance_m",
    "ball_displacement_speed_mps",
)


def graded_mask(n_frames, game=GAME):
    mask = np.zeros(n_frames, dtype=np.bool_)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


def synthetic_checks():
    handover = np.r_[np.full(40, 0.6), np.full(40, -0.6)]
    handover_states = viterbi_decode(handover)
    if not (np.all(handover_states[:40] == HOME)
            and np.all(handover_states[40:] == AWAY)):
        raise AssertionError("sustained team handover failed")

    gap_length = 100
    gap_evidence = np.r_[np.full(40, 0.8), np.zeros(gap_length),
                         np.full(40, -0.8)]
    gap_ball_seen = np.r_[np.ones(40, dtype=np.bool_),
                          np.zeros(gap_length, dtype=np.bool_),
                          np.ones(40, dtype=np.bool_)]
    gap_states = viterbi_three_state(gap_evidence, np.zeros(len(gap_evidence)))
    centered_states, centered_count = midpoint_missing_ball_handovers(
        gap_states, gap_evidence, gap_ball_seen)
    gap_start = 40
    expected_boundary = gap_start + gap_length // 2
    changes = np.flatnonzero(centered_states[1:] != centered_states[:-1]) + 1
    if centered_count != 1 or not np.array_equal(changes, [expected_boundary]):
        raise AssertionError("supported handover was not centered in a neutral ball gap")

    reverse_evidence = -gap_evidence
    reverse_states = viterbi_three_state(
        reverse_evidence, np.zeros(len(reverse_evidence)))
    centered_reverse, reverse_count = midpoint_missing_ball_handovers(
        reverse_states, reverse_evidence, gap_ball_seen)
    reverse_changes = np.flatnonzero(
        centered_reverse[1:] != centered_reverse[:-1]) + 1
    if reverse_count != 1 or not np.array_equal(reverse_changes, [expected_boundary]):
        raise AssertionError("reverse handover was not centered in a neutral ball gap")

    dead_gap_states = gap_states.copy()
    dead_gap_states[gap_start + 40:gap_start + 60] = DEAD
    unchanged_states, unchanged_count = midpoint_missing_ball_handovers(
        dead_gap_states, gap_evidence, gap_ball_seen)
    if unchanged_count or not np.array_equal(unchanged_states, dead_gap_states):
        raise AssertionError("a ball gap with a dead-ball state was midpoint-adjusted")

    same_team_evidence = np.r_[np.full(40, 0.8), np.zeros(gap_length),
                               np.full(40, 0.8)]
    same_team_states = viterbi_three_state(
        same_team_evidence, np.zeros(len(same_team_evidence)))
    unchanged_states, unchanged_count = midpoint_missing_ball_handovers(
        same_team_states, same_team_evidence, gap_ball_seen)
    if unchanged_count or not np.array_equal(unchanged_states, same_team_states):
        raise AssertionError("same-team continuity was changed across a ball gap")

    spike = np.full(80, 0.6)
    spike[40] = -0.99
    if not np.all(viterbi_decode(spike) == HOME):
        raise AssertionError("one-frame opposing spike caused a false handover")

    gap = np.r_[np.full(30, 0.6), np.zeros(30), np.full(30, 0.6)]
    if not np.all(viterbi_decode(gap) == HOME):
        raise AssertionError("neutral tracking gap changed the owner")

    n_frames = 150
    features = {
        "ball_seen": np.ones(n_frames, dtype=np.bool_),
        "ball_displacement_speed_mps": np.full(n_frames, 5.0, dtype=np.float32),
        "ball_outside_pitch": np.zeros(n_frames, dtype=np.bool_),
        "home_nearest_distance_m": np.full(n_frames, 2.0, dtype=np.float32),
        "away_nearest_distance_m": np.full(n_frames, 8.0, dtype=np.float32),
        "all_player_speed_median_mps": np.full(n_frames, 1.2, dtype=np.float32),
        "player_speed_fraction_below_0_5_mps": np.zeros(n_frames, dtype=np.float32),
        "home_track_count": np.full(n_frames, 11, dtype=np.int16),
        "away_track_count": np.full(n_frames, 11, dtype=np.int16),
    }
    context = np.ones(n_frames, dtype=np.bool_)
    dead_score, _, _ = stoppage_evidence(features, context)
    if np.any(viterbi_three_state(np.full(n_frames, 0.8), dead_score) == DEAD):
        raise AssertionError("set-piece context alone caused a dead-ball label")

    features["ball_seen"][:] = False
    features["ball_displacement_speed_mps"][:] = np.nan
    dead_score, _, _ = stoppage_evidence(features, context)
    if np.any(dead_score > 0.0) or np.any(viterbi_three_state(
            np.full(n_frames, 0.8), dead_score) == DEAD):
        raise AssertionError("active player movement in a ball gap caused a dead-ball label")

    features["ball_seen"][:] = True
    features["ball_displacement_speed_mps"][:] = 5.0
    features["ball_outside_pitch"][50] = True
    dead_score, _, _ = stoppage_evidence(features, np.zeros(n_frames, dtype=np.bool_))
    if np.any(viterbi_three_state(np.full(n_frames, 0.8), dead_score) == DEAD):
        raise AssertionError("isolated out-of-pitch observation caused a dead-ball label")

    features["ball_outside_pitch"][:] = False
    features["ball_displacement_speed_mps"][30:60] = 0.2
    features["home_nearest_distance_m"][30:60] = 7.0
    features["away_nearest_distance_m"][30:60] = 9.0
    features["ball_outside_pitch"][30:37] = True
    features["all_player_speed_median_mps"][30:60] = 0.6
    features["player_speed_fraction_below_0_5_mps"][30:60] = 0.5
    dead_score, _, _ = stoppage_evidence(
        features, np.zeros(n_frames, dtype=np.bool_))
    states = viterbi_three_state(np.full(n_frames, 0.8), dead_score)
    if not np.any(states[30:60] == DEAD):
        raise AssertionError("corroborated sustained stoppage was missed")

    features["ball_seen"][:] = False
    features["ball_displacement_speed_mps"][:] = np.nan
    features["all_player_speed_median_mps"][:] = 1.2
    features["player_speed_fraction_below_0_5_mps"][:] = 0.0
    features["all_player_speed_median_mps"][45:125] = 0.4
    features["player_speed_fraction_below_0_5_mps"][45:125] = 0.8
    dead_score, _, _ = stoppage_evidence(
        features, np.zeros(n_frames, dtype=np.bool_))
    states = viterbi_three_state(np.full(n_frames, 0.8), dead_score)
    if not np.all(states[70:110] == DEAD):
        raise AssertionError("sustained player stop during a ball gap was missed")
    if np.any(states[:30] == DEAD):
        raise AssertionError("player stoppage evidence leaked outside its sustained interval")
    if np.any(states[140:150] == DEAD):
        raise AssertionError("player activity failed to end a stoppage during a ball gap")

    first_graded = half_video_blocks(GAME)[0][1]
    boundary_candidate = np.zeros(GAMES[GAME]["n_video_frames"], dtype=np.bool_)
    boundary_candidate[first_graded - 35:first_graded] = True
    boundary_counts = rolling_count_by_half(boundary_candidate, 50, GAME)
    if np.any(boundary_counts[first_graded:first_graded + 10]):
        raise AssertionError("pre-match stillness leaked into the first scored half")

    return {
        "sustained_handover": "pass",
        "supported_switch_centered_in_ball_gap": "pass",
        "reverse_switch_centered_in_ball_gap": "pass",
        "dead_ball_gap_is_not_midpoint_adjusted": "pass",
        "same_team_gap_is_unchanged": "pass",
        "one_frame_opponent_spike": "pass",
        "neutral_ball_gap": "pass",
        "tag_only_is_neutral": "pass",
        "active_ball_gap_does_not_trigger_dead_ball": "pass",
        "isolated_out_of_pitch_is_rejected": "pass",
        "corroborated_sustained_stop_is_detected": "pass",
        "sustained_player_stop_during_ball_gap": "pass",
        "player_activity_ends_ball_gap_stoppage": "pass",
        "pre_match_player_stop_does_not_leak_into_half": "pass",
    }


def apply_ball_mask(features, mask):
    masked = dict(features)
    for name in BALL_FEATURES:
        values = features[name].copy()
        if values.dtype == np.bool_:
            values[mask] = False
        else:
            values[mask] = np.nan
        masked[name] = values
    return masked


def random_mask(features, graded, rate, random_state):
    observed = np.flatnonzero(features["ball_seen"] & graded)
    count = int(round(len(observed) * rate))
    mask = np.zeros(len(graded), dtype=np.bool_)
    if count:
        mask[random_state.choice(observed, size=count, replace=False)] = True
    return mask


def burst_mask(features, graded, burst_frames, removed_fraction, random_state):
    observed = features["ball_seen"] & graded
    target = int(round(np.sum(observed) * removed_fraction))
    mask = np.zeros(len(graded), dtype=np.bool_)
    blocks = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    masked_count = 0
    attempts = 0
    max_attempts = max(1000, target * 10)

    while masked_count < target and attempts < max_attempts:
        first, stop = blocks[int(random_state.integers(len(blocks)))]
        if stop - first <= burst_frames:
            attempts += 1
            continue
        start = int(random_state.integers(first, stop - burst_frames + 1))
        end = start + burst_frames
        new_mask = observed[start:end] & ~mask[start:end]
        mask[start:end] |= observed[start:end]
        masked_count += int(np.sum(new_mask))
        attempts += 1
    if masked_count < target:
        raise RuntimeError("could not place enough non-overlapping dropout frames")
    return mask


def prediction_profile(labels, game=GAME):
    halves = []
    for half, first, last in half_video_blocks(game):
        part = labels[first:last + 1]
        dead = part == "D"
        starts = np.flatnonzero(dead & ~np.r_[False, dead[:-1]])
        stops = np.flatnonzero(dead & ~np.r_[dead[1:], False]) + 1
        live = np.isin(part, ("H", "A"))
        live_starts = live & ~np.r_[False, live[:-1]]
        live_starts[1:] |= live[1:] & live[:-1] & (part[1:] != part[:-1])
        halves.append({
            "half": half,
            "H_frames": int(np.sum(part == "H")),
            "A_frames": int(np.sum(part == "A")),
            "D_frames": int(np.sum(dead)),
            "live_spells": int(np.sum(live_starts)),
            "dead_spells": int(len(starts)),
            "median_dead_seconds": float(np.median(stops - starts) / 25.0)
            if len(starts) else None,
        })
    return halves


def compare_mask_scenario(base_labels, scenario_labels, mask, graded, name,
                          original_visible_ball_frames):
    comparable = graded & ~mask
    agreement = float(np.mean(base_labels[comparable] == scenario_labels[comparable]))
    graded_frame_count = int(np.sum(graded))
    masked_count = int(np.sum(mask))
    remaining_visible = original_visible_ball_frames - masked_count
    remaining_missing_pct = 100.0 * (
        graded_frame_count - remaining_visible) / graded_frame_count
    baseline_missing_pct = 100.0 * (
        graded_frame_count - original_visible_ball_frames) / graded_frame_count

    baseline_profile = prediction_profile(base_labels)
    scenario_profile = prediction_profile(scenario_labels)
    profile_delta = []
    for baseline_half, scenario_half in zip(baseline_profile, scenario_profile):
        profile_delta.append({
            "half": baseline_half["half"],
            "H_frames": scenario_half["H_frames"] - baseline_half["H_frames"],
            "A_frames": scenario_half["A_frames"] - baseline_half["A_frames"],
            "D_frames": scenario_half["D_frames"] - baseline_half["D_frames"],
            "live_spells": scenario_half["live_spells"] - baseline_half["live_spells"],
            "dead_spells": scenario_half["dead_spells"] - baseline_half["dead_spells"],
        })
    return {
        "scenario": name,
        "masked_ball_frames": masked_count,
        "masked_pct_of_original_ball_observations": round(
            100.0 * masked_count / max(1, original_visible_ball_frames), 3),
        "baseline_ball_missing_pct_graded": round(baseline_missing_pct, 3),
        "remaining_ball_missing_pct_graded": round(remaining_missing_pct, 3),
        "agreement_on_unmasked_graded_frames": round(agreement, 5),
        "profile_delta_from_baseline": profile_delta,
        "prediction_profile": scenario_profile,
    }


def run(features_path, data_dir=None, output_dir=None, seed=2027):
    features = load_features(features_path)
    events = load_smart_tagging(game=GAME, data_dir=data_dir)
    n_frames = len(features["frame"])
    graded = graded_mask(n_frames)
    original_visible_ball_frames = int(np.sum(features["ball_seen"] & graded))
    synthetic = synthetic_checks()
    base_labels, _, _, base_diagnostics = decode_with_dead_ball(features, events)
    random_state = np.random.default_rng(seed)

    scenarios = []
    for rate in (0.1, 0.25, 0.4):
        mask = random_mask(features, graded, rate, random_state)
        masked_features = apply_ball_mask(features, mask)
        labels, _, _, _ = decode_with_dead_ball(masked_features, events)
        scenarios.append(compare_mask_scenario(
            base_labels, labels, mask, graded, "random_{:.0f}pct".format(rate * 100),
            original_visible_ball_frames))

    for burst_frames in (25, 75, 125):
        mask = burst_mask(features, graded, burst_frames, 0.15, random_state)
        masked_features = apply_ball_mask(features, mask)
        labels, _, _, _ = decode_with_dead_ball(masked_features, events)
        scenarios.append(compare_mask_scenario(
            base_labels, labels, mask, graded,
            "bursts_{}frames_15pct".format(burst_frames),
            original_visible_ball_frames))

    report = {
        "purpose": "label-free structural robustness; no ground truth or competition score loaded",
        "seed": seed,
        "graded_frames": int(np.sum(graded)),
        "original_ball_visible_graded_frames": original_visible_ball_frames,
        "baseline_ball_missing_pct_graded": round(
            100.0 * (np.sum(graded) - original_visible_ball_frames) / np.sum(graded), 3),
        "synthetic_checks": synthetic,
        "baseline_profile": prediction_profile(base_labels),
        "baseline_cue_counts": base_diagnostics["cue_frame_counts"],
        "dropout_scenarios": scenarios,
    }

    output_dir = require_output_outside_repository(
        output_dir or os.path.join(tempfile.gettempdir(), "mtah2027-stress-test"))
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "stress_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report_path, report


if __name__ == "__main__":
    default_features = os.path.join(
        tempfile.gettempdir(), "mtah2027-feature-audit", "tracking_features.npz")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", default=default_features,
                        help="feature archive generated by feature_audit.py")
    parser.add_argument("--data-dir", default=None,
                        help="asset root containing the tactical/ folder")
    parser.add_argument("--out", default=None,
                        help="report folder (must be outside this repository)")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    report_path, report = run(args.features, args.data_dir, args.out, args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nstress report: {}".format(report_path))

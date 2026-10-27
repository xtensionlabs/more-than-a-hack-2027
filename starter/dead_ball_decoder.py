"""Add a conservative dead-ball state to the possession sequence candidate.

Dead-ball evidence comes from persistent out-of-pitch tracking, sustained low ball movement,
player movement during ball gaps, and weak corroboration from automatic set-piece tags. A tag
interval alone never labels a frame as dead. Outputs are written outside the repository by default.
"""

import argparse
import csv
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks
from starter.feature_audit import require_output_outside_repository
from starter.load_smart_tagging import load as load_smart_tagging
from starter.possession_decoder import signed_evidence

HOME, AWAY, DEAD = 0, 1, 2
STATE_LABELS = np.array(["H", "A", "D"])
PLAYER_STOP_WINDOW_FRAMES = 50
MINIMUM_PLAYER_STOP_FRAMES = 35
PLAYER_ACTIVITY_WINDOW_FRAMES = 15
MINIMUM_PLAYER_ACTIVITY_FRAMES = 10
MINIMUM_VISIBLE_PER_TEAM = 6
PLAYER_STOP_SCORE = 0.9
SET_PIECE_STOP_BONUS = 0.3
PLAYER_RESTART_SCORE = 0.75


def rolling_count(mask, window):
    values = np.asarray(mask, dtype=np.int16)
    before = window // 2
    after = window - 1 - before
    padded = np.pad(values, (before, after))
    return np.convolve(padded, np.ones(window, dtype=np.int16), mode="valid")


def rolling_count_by_half(mask, window, game=None):
    """Count a frame-level cue without letting evidence cross half boundaries."""
    if game is None:
        return rolling_count(mask, window)
    counts = np.zeros(len(mask), dtype=np.int16)
    for _, first, last in half_video_blocks(game):
        first = max(0, first)
        last = min(len(mask) - 1, last)
        if first <= last:
            counts[first:last + 1] = rolling_count(mask[first:last + 1], window)
    return counts


def restart_context(events, n_frames):
    """Mark set-piece context for corroboration, not as a dead-ball label."""
    context = np.zeros(n_frames, dtype=np.bool_)
    event_count = 0
    for event in events:
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        first = max(0, int(event["start_frame"]))
        last = min(n_frames, int(event["end_frame"]))
        if first < last:
            context[first:last] = True
            event_count += 1
    return context, event_count


def stoppage_evidence(features, set_piece_context, slow_speed_threshold=1.0,
                      slow_fraction_threshold=0.75, slow_window_frames=25,
                      minimum_slow_observations=8, close_distance_m=1.5,
                      far_distance_m=3.0, fast_speed_threshold=3.0,
                      player_stop_window_frames=PLAYER_STOP_WINDOW_FRAMES,
                      minimum_player_stop_frames=MINIMUM_PLAYER_STOP_FRAMES,
                      player_activity_window_frames=PLAYER_ACTIVITY_WINDOW_FRAMES,
                      minimum_player_activity_frames=MINIMUM_PLAYER_ACTIVITY_FRAMES,
                      minimum_visible_per_team=MINIMUM_VISIBLE_PER_TEAM, game=None):
    """Build dead-ball evidence from ball motion or sustained player inactivity.

    A missing ball is not itself evidence for a stoppage. During a ball gap, sustained low team
    motion and a high fraction of nearly stationary players can provide independent evidence;
    a set-piece tag only adds weak support after that physical cue is already present.
    """
    ball_seen = features["ball_seen"].astype(np.bool_, copy=False)
    ball_speed = features["ball_displacement_speed_mps"]
    valid_speed = ball_seen & np.isfinite(ball_speed)
    slow = valid_speed & (ball_speed <= slow_speed_threshold)
    observed_count = rolling_count(valid_speed, slow_window_frames)
    slow_count = rolling_count(slow, slow_window_frames)
    slow_fraction = np.zeros(len(ball_seen), dtype=np.float32)
    enough_observations = observed_count >= minimum_slow_observations
    slow_fraction[enough_observations] = (
        slow_count[enough_observations] / observed_count[enough_observations])
    persistent_slow = (
        ball_seen & enough_observations & (slow_fraction >= slow_fraction_threshold))

    outside = ball_seen & features["ball_outside_pitch"]
    outside_count = rolling_count(outside, 7)
    persistent_outside = outside & (outside_count >= 3)

    home_distance = features["home_nearest_distance_m"]
    away_distance = features["away_nearest_distance_m"]
    home_available = np.isfinite(home_distance)
    away_available = np.isfinite(away_distance)
    nearest_distance = np.full(len(ball_seen), np.nan, dtype=np.float32)
    both_available = home_available & away_available
    nearest_distance[both_available] = np.minimum(
        home_distance[both_available], away_distance[both_available])
    only_home = home_available & ~away_available
    only_away = away_available & ~home_available
    nearest_distance[only_home] = home_distance[only_home]
    nearest_distance[only_away] = away_distance[only_away]
    far_from_players = ball_seen & np.isfinite(nearest_distance) & (
        nearest_distance >= far_distance_m)
    close_to_player = ball_seen & np.isfinite(nearest_distance) & (
        nearest_distance <= close_distance_m)

    player_median = features["all_player_speed_median_mps"]
    player_stationary_fraction = features["player_speed_fraction_below_0_5_mps"]
    home_track_count = features["home_track_count"]
    away_track_count = features["away_track_count"]
    low_player_motion = np.isfinite(player_median) & (player_median <= 0.8)
    many_stationary_players = np.isfinite(player_stationary_fraction) & (
        player_stationary_fraction >= 0.45)
    player_stop_candidate = (
        ~ball_seen
        & np.isfinite(player_median)
        & (player_median <= 0.5)
        & np.isfinite(player_stationary_fraction)
        & (player_stationary_fraction >= 0.65)
        & (home_track_count >= minimum_visible_per_team)
        & (away_track_count >= minimum_visible_per_team)
    )
    player_stop_window_count = rolling_count_by_half(
        player_stop_candidate, player_stop_window_frames, game)
    sustained_player_stop = player_stop_candidate & (
        player_stop_window_count >= minimum_player_stop_frames)
    player_restart_candidate = (
        ~ball_seen
        & np.isfinite(player_median)
        & (player_median >= 1.0)
        & np.isfinite(player_stationary_fraction)
        & (player_stationary_fraction <= 0.35)
        & (home_track_count >= minimum_visible_per_team)
        & (away_track_count >= minimum_visible_per_team)
    )
    player_restart_window_count = rolling_count_by_half(
        player_restart_candidate, player_activity_window_frames, game)
    sustained_player_restart = player_restart_candidate & (
        player_restart_window_count >= minimum_player_activity_frames)
    fast_ball = valid_speed & (ball_speed >= fast_speed_threshold) & ~outside

    dead_score = np.zeros(len(ball_seen), dtype=np.float32)
    dead_score[outside] += 1.25
    dead_score[persistent_outside] += 0.75
    dead_score[persistent_slow & far_from_players] += 0.8
    dead_score[persistent_slow & set_piece_context] += 0.25
    dead_score[persistent_slow & low_player_motion] += 0.25
    dead_score[persistent_slow & many_stationary_players] += 0.2
    dead_score[persistent_slow & close_to_player] -= 0.65
    dead_score[fast_ball] -= 0.8
    moving_players_at_exit = persistent_outside & np.isfinite(player_median) & (
        player_median >= 1.5)
    dead_score[moving_players_at_exit] += 0.2
    dead_score[sustained_player_stop] += PLAYER_STOP_SCORE
    dead_score[sustained_player_stop & set_piece_context] += SET_PIECE_STOP_BONUS
    dead_score[sustained_player_restart] -= PLAYER_RESTART_SCORE
    np.clip(dead_score, -1.5, 2.5, out=dead_score)

    cue_counts = {
        "ball_observed": int(np.sum(ball_seen)),
        "ball_outside_pitch": int(np.sum(outside)),
        "persistent_outside": int(np.sum(persistent_outside)),
        "persistent_slow_ball": int(np.sum(persistent_slow)),
        "slow_and_far_from_players": int(np.sum(persistent_slow & far_from_players)),
        "slow_inside_set_piece_context": int(np.sum(persistent_slow & set_piece_context)),
        "slow_with_many_stationary_players": int(np.sum(
            persistent_slow & many_stationary_players)),
        "slow_near_a_player_suppressed": int(np.sum(persistent_slow & close_to_player)),
        "fast_ball_suppressed": int(np.sum(fast_ball)),
        "sustained_player_stop_during_ball_gap": int(np.sum(sustained_player_stop)),
        "player_stop_with_set_piece_context": int(np.sum(
            sustained_player_stop & set_piece_context)),
        "sustained_player_activity_during_ball_gap": int(np.sum(
            sustained_player_restart)),
    }
    return dead_score, cue_counts, slow_fraction


def viterbi_three_state(owner_evidence, dead_score, handover_penalty=10.0,
                        dead_transition_penalty=6.0, initial_dead_penalty=4.0):
    """Decode H/A/D while retaining the possession model's owner evidence."""
    owner_evidence = np.asarray(owner_evidence, dtype=np.float64)
    dead_score = np.asarray(dead_score, dtype=np.float64)
    if owner_evidence.ndim != 1 or len(owner_evidence) == 0:
        raise ValueError("owner_evidence must be a non-empty one-dimensional sequence")
    if owner_evidence.shape != dead_score.shape:
        raise ValueError("owner and dead-ball evidence must have matching shapes")
    if not np.all(np.isfinite(owner_evidence)) or not np.all(np.isfinite(dead_score)):
        raise ValueError("decoder evidence must contain only finite values")
    if handover_penalty < 0 or dead_transition_penalty < 0 or initial_dead_penalty < 0:
        raise ValueError("transition penalties must be non-negative")

    emissions = np.column_stack((
        -owner_evidence + 0.5 * dead_score,
        owner_evidence + 0.5 * dead_score,
        -dead_score,
    ))
    transitions = np.array((
        (0.0, handover_penalty, dead_transition_penalty),
        (handover_penalty, 0.0, dead_transition_penalty),
        (dead_transition_penalty, dead_transition_penalty, 0.0),
    ), dtype=np.float64)

    costs = np.empty_like(emissions)
    previous_state = np.zeros(emissions.shape, dtype=np.int8)
    costs[0] = emissions[0]
    costs[0, DEAD] += initial_dead_penalty
    for frame in range(1, len(owner_evidence)):
        candidate_costs = costs[frame - 1, :, None] + transitions
        best_previous = np.argmin(candidate_costs, axis=0)
        previous_state[frame] = best_previous
        costs[frame] = emissions[frame] + candidate_costs[
            best_previous, np.arange(3)]

    states = np.empty(len(owner_evidence), dtype=np.int8)
    states[-1] = int(np.argmin(costs[-1]))
    for frame in range(len(owner_evidence) - 1, 0, -1):
        states[frame - 1] = previous_state[frame, states[frame]]
    return states


def state_summary(states, fps):
    counts = np.bincount(states, minlength=3)
    dead = states == DEAD
    starts = np.flatnonzero(dead & ~np.r_[False, dead[:-1]])
    stops = np.flatnonzero(dead & ~np.r_[dead[1:], False]) + 1
    durations = (stops - starts) / float(fps)
    return {
        "frames": {"H": int(counts[HOME]), "A": int(counts[AWAY]), "D": int(counts[DEAD])},
        "dead_spells": int(len(durations)),
        "dead_spell_median_seconds": float(np.median(durations)) if len(durations) else None,
        "dead_spell_p90_seconds": float(np.percentile(durations, 90)) if len(durations) else None,
    }


def decode(features, events, game=GAME, handover_penalty=10.0,
           dead_transition_penalty=6.0, slow_speed_threshold=1.0,
           slow_fraction_threshold=0.75):
    n_frames = GAMES[game]["n_video_frames"]
    expected_frames = np.arange(n_frames, dtype=np.int32)
    if not np.array_equal(features["frame"], expected_frames):
        raise ValueError("feature archive is not aligned to the tactical video frames")

    set_piece_context, set_piece_count = restart_context(events, n_frames)
    dead_score, cue_counts, slow_fraction = stoppage_evidence(
        features, set_piece_context,
        slow_speed_threshold=slow_speed_threshold,
        slow_fraction_threshold=slow_fraction_threshold,
        game=game)
    owner_evidence = signed_evidence(features)
    labels = np.full(n_frames, "X", dtype="<U1")
    halves = []

    for half, first, last in half_video_blocks(game):
        states = viterbi_three_state(
            owner_evidence[first:last + 1], dead_score[first:last + 1],
            handover_penalty, dead_transition_penalty)
        labels[first:last + 1] = STATE_LABELS[states]
        halves.append({
            "half": half,
            "frames": int(last - first + 1),
            "states": state_summary(states, GAMES[game]["fps"]),
            "slow_ball_confirmed_frames": int(np.sum(
                slow_fraction[first:last + 1] >= slow_fraction_threshold)),
            "mean_dead_score": float(np.mean(dead_score[first:last + 1])),
        })

    diagnostics = {
        "method": "three-state Viterbi with geometric possession and physical stoppage evidence",
        "set_piece_events_used_as_context": set_piece_count,
        "set_piece_context_frames": int(np.sum(set_piece_context)),
        "tag_policy": "context only; tag intervals alone do not emit D",
        "cue_frame_counts": cue_counts,
        "parameters": {
            "handover_penalty": handover_penalty,
            "dead_transition_penalty": dead_transition_penalty,
            "slow_speed_threshold_mps": slow_speed_threshold,
            "slow_fraction_threshold": slow_fraction_threshold,
            "player_gap_stop_window_frames": PLAYER_STOP_WINDOW_FRAMES,
            "minimum_player_stop_frames": MINIMUM_PLAYER_STOP_FRAMES,
            "player_gap_activity_window_frames": PLAYER_ACTIVITY_WINDOW_FRAMES,
            "minimum_player_activity_frames": MINIMUM_PLAYER_ACTIVITY_FRAMES,
            "minimum_visible_players_per_team": MINIMUM_VISIBLE_PER_TEAM,
            "player_gap_stop_score": PLAYER_STOP_SCORE,
            "set_piece_stop_bonus": SET_PIECE_STOP_BONUS,
            "player_gap_restart_score": PLAYER_RESTART_SCORE,
        },
        "halves": halves,
    }
    return labels, dead_score, set_piece_context, diagnostics


def load_features(path):
    if not os.path.isfile(path):
        raise IOError("feature archive not found: {}".format(path))
    required = (
        "frame", "ball_seen", "ball_x_norm", "ball_y_norm", "ball_outside_pitch",
        "home_nearest_distance_m", "away_nearest_distance_m",
        "ball_displacement_speed_mps", "all_player_speed_median_mps",
        "player_speed_fraction_below_0_5_mps", "home_track_count", "away_track_count",
    )
    with np.load(path) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError("feature archive is missing arrays: {}".format(", ".join(missing)))
        return {name: archive[name] for name in required}


def write_outputs(labels, dead_score, set_piece_context, diagnostics, output_dir):
    output_dir = require_output_outside_repository(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "possession_with_dead_ball.csv")
    evidence_path = os.path.join(output_dir, "dead_ball_evidence.npz")
    diagnostics_path = os.path.join(output_dir, "dead_ball_summary.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("frame", "label"))
        writer.writerows(enumerate(labels.tolist()))

    np.savez_compressed(
        evidence_path,
        frame=np.arange(len(labels), dtype=np.int32),
        dead_score=dead_score,
        set_piece_context=set_piece_context,
        state=np.where(labels == "H", HOME, np.where(labels == "A", AWAY,
                                                       np.where(labels == "D", DEAD, -1))).astype(
                                                           np.int8),
    )
    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return csv_path, evidence_path, diagnostics_path


def run(features_path, output_dir, data_dir=None, game=GAME,
        handover_penalty=10.0, dead_transition_penalty=6.0,
        slow_speed_threshold=1.0, slow_fraction_threshold=0.75):
    features = load_features(features_path)
    events = load_smart_tagging(game=game, data_dir=data_dir)
    labels, dead_score, context, diagnostics = decode(
        features, events, game, handover_penalty, dead_transition_penalty,
        slow_speed_threshold, slow_fraction_threshold)
    paths = write_outputs(labels, dead_score, context, diagnostics, output_dir)
    return paths, diagnostics


if __name__ == "__main__":
    default_features = os.path.join(
        tempfile.gettempdir(), "mtah2027-feature-audit", "tracking_features.npz")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", default=default_features,
                        help="feature archive generated by feature_audit.py")
    parser.add_argument("--out", default=None, help="output folder (must be outside this repo)")
    parser.add_argument("--data-dir", default=None,
                        help="asset root containing the tactical/ folder")
    parser.add_argument("--handover-penalty", type=float, default=10.0)
    parser.add_argument("--dead-transition-penalty", type=float, default=6.0)
    parser.add_argument("--slow-speed-threshold", type=float, default=1.0)
    parser.add_argument("--slow-fraction-threshold", type=float, default=0.75)
    args = parser.parse_args()

    output_dir = args.out or os.path.join(
        tempfile.gettempdir(), "mtah2027-dead-ball-decoder")
    output_paths, diagnostics = run(
        args.features, output_dir, args.data_dir,
        handover_penalty=args.handover_penalty,
        dead_transition_penalty=args.dead_transition_penalty,
        slow_speed_threshold=args.slow_speed_threshold,
        slow_fraction_threshold=args.slow_fraction_threshold)
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print("\nsubmission candidate: {}".format(output_paths[0]))
    print("evidence archive: {}".format(output_paths[1]))
    print("diagnostics: {}".format(output_paths[2]))

"""Decode stable home/away possession candidates from the label-free feature audit.

The decoder uses signed geometric evidence and a per-half Viterbi path with a fixed cost for each
team switch. Missing or weak observations therefore carry the sequence state instead of creating
new spells. Dead-ball classification is deliberately left to a later stage.
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

HOME, AWAY = 0, 1
STATE_LABELS = np.array(["H", "A"])


def signed_evidence(features, margin_scale_m=2.5, proximity_scale_m=5.0,
                    single_team_weight=0.35):
    """Positive evidence favors home; negative evidence favors away.

    Team distance differences are strongest when a tracked player is close to the ball. Frames
    with no observed ball produce zero evidence and are resolved by sequence continuity.
    """
    if margin_scale_m <= 0 or proximity_scale_m <= 0:
        raise ValueError("distance scales must be positive")
    if not 0.0 <= single_team_weight <= 1.0:
        raise ValueError("single_team_weight must be between 0 and 1")

    n_frames = len(features["frame"])
    evidence = np.zeros(n_frames, dtype=np.float32)
    ball_seen = features["ball_seen"].astype(np.bool_, copy=False)
    home = features["home_nearest_distance_m"]
    away = features["away_nearest_distance_m"]
    home_seen = ball_seen & np.isfinite(home)
    away_seen = ball_seen & np.isfinite(away)
    both_seen = home_seen & away_seen
    only_home = home_seen & ~away_seen
    only_away = away_seen & ~home_seen

    if np.any(both_seen):
        home_distance = home[both_seen]
        away_distance = away[both_seen]
        closest_distance = np.minimum(home_distance, away_distance)
        margin = away_distance - home_distance
        evidence[both_seen] = (
            np.tanh(margin / margin_scale_m)
            * np.exp(-closest_distance / proximity_scale_m)
        )

    if np.any(only_home):
        evidence[only_home] = single_team_weight * np.exp(
            -home[only_home] / proximity_scale_m)
    if np.any(only_away):
        evidence[only_away] = -single_team_weight * np.exp(
            -away[only_away] / proximity_scale_m)
    return evidence


def viterbi_decode(evidence, change_penalty=10.0):
    """Find the minimum-cost two-state sequence for one continuous half."""
    evidence = np.asarray(evidence, dtype=np.float64)
    if evidence.ndim != 1 or not len(evidence):
        raise ValueError("evidence must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(evidence)):
        raise ValueError("evidence must contain only finite values")
    if change_penalty < 0:
        raise ValueError("change_penalty must be non-negative")

    costs = np.empty((len(evidence), 2), dtype=np.float64)
    previous_state = np.zeros((len(evidence), 2), dtype=np.int8)
    costs[0, HOME] = -evidence[0]
    costs[0, AWAY] = evidence[0]

    for frame in range(1, len(evidence)):
        stay_home = costs[frame - 1, HOME]
        switch_to_home = costs[frame - 1, AWAY] + change_penalty
        if stay_home <= switch_to_home:
            costs[frame, HOME] = -evidence[frame] + stay_home
            previous_state[frame, HOME] = HOME
        else:
            costs[frame, HOME] = -evidence[frame] + switch_to_home
            previous_state[frame, HOME] = AWAY

        stay_away = costs[frame - 1, AWAY]
        switch_to_away = costs[frame - 1, HOME] + change_penalty
        if stay_away <= switch_to_away:
            costs[frame, AWAY] = evidence[frame] + stay_away
            previous_state[frame, AWAY] = AWAY
        else:
            costs[frame, AWAY] = evidence[frame] + switch_to_away
            previous_state[frame, AWAY] = HOME

    states = np.empty(len(evidence), dtype=np.int8)
    states[-1] = HOME if costs[-1, HOME] <= costs[-1, AWAY] else AWAY
    for frame in range(len(evidence) - 1, 0, -1):
        states[frame - 1] = previous_state[frame, states[frame]]
    return states


def spell_summary(labels, fps):
    """Summarise predicted live-team runs without consulting any labels."""
    if not len(labels):
        return {"count": 0, "home": 0, "away": 0, "median_seconds": None,
                "p10_seconds": None, "p90_seconds": None}

    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(labels)]))
    durations = (stops - starts) / float(fps)
    teams = labels[starts]
    return {
        "count": int(len(starts)),
        "home": int(np.sum(teams == "H")),
        "away": int(np.sum(teams == "A")),
        "median_seconds": float(np.median(durations)),
        "p10_seconds": float(np.percentile(durations, 10)),
        "p90_seconds": float(np.percentile(durations, 90)),
    }


def decode(features, game=GAME, margin_scale_m=2.5, proximity_scale_m=5.0,
           change_penalty=10.0):
    """Decode each half independently; leave non-play frames as X."""
    n_frames = GAMES[game]["n_video_frames"]
    expected_frames = np.arange(n_frames, dtype=np.int32)
    if not np.array_equal(features["frame"], expected_frames):
        raise ValueError("feature archive is not aligned to the tactical video frames")

    evidence = signed_evidence(features, margin_scale_m, proximity_scale_m)
    labels = np.full(n_frames, "X", dtype="<U1")
    halves = []

    for half, first, last in half_video_blocks(game):
        half_evidence = evidence[first:last + 1]
        informative = np.abs(half_evidence) > 0.05
        if np.any(informative):
            states = viterbi_decode(half_evidence, change_penalty)
            half_labels = STATE_LABELS[states]
            labels[first:last + 1] = half_labels
            halves.append({
                "half": half,
                "frames": int(last - first + 1),
                "informative_frames": int(np.sum(informative)),
                "evidence_abs_p50": float(np.percentile(np.abs(half_evidence), 50)),
                "evidence_abs_p90": float(np.percentile(np.abs(half_evidence), 90)),
                "spells": spell_summary(half_labels, GAMES[game]["fps"]),
            })
        else:
            halves.append({
                "half": half,
                "frames": int(last - first + 1),
                "informative_frames": 0,
                "evidence_abs_p50": 0.0,
                "evidence_abs_p90": 0.0,
                "spells": {"count": 0, "home": 0, "away": 0},
            })

    diagnostics = {
        "method": "signed proximity evidence with per-half Viterbi switching cost",
        "parameters": {
            "margin_scale_m": margin_scale_m,
            "proximity_scale_m": proximity_scale_m,
            "change_penalty": change_penalty,
        },
        "dead_ball_model": "not included in milestone 4",
        "halves": halves,
    }
    return labels, evidence, diagnostics


def load_features(path):
    if not os.path.isfile(path):
        raise IOError("feature archive not found: {}".format(path))
    required = (
        "frame", "ball_seen", "home_nearest_distance_m", "away_nearest_distance_m")
    with np.load(path) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError("feature archive is missing arrays: {}".format(", ".join(missing)))
        return {name: archive[name] for name in required}


def write_outputs(labels, evidence, diagnostics, output_dir):
    output_dir = require_output_outside_repository(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "possession_candidate.csv")
    evidence_path = os.path.join(output_dir, "possession_evidence.npz")
    diagnostics_path = os.path.join(output_dir, "possession_decoder_summary.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("frame", "label"))
        writer.writerows(enumerate(labels.tolist()))

    state_codes = np.full(len(labels), -1, dtype=np.int8)
    state_codes[labels == "H"] = HOME
    state_codes[labels == "A"] = AWAY
    np.savez_compressed(
        evidence_path,
        frame=np.arange(len(labels), dtype=np.int32),
        evidence=evidence,
        state=state_codes,
    )
    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return csv_path, evidence_path, diagnostics_path


def run(features_path=None, output_dir=None, game=GAME, margin_scale_m=2.5,
        proximity_scale_m=5.0, change_penalty=10.0):
    temp_root = tempfile.gettempdir()
    features_path = features_path or os.path.join(
        temp_root, "mtah2027-feature-audit", "tracking_features.npz")
    output_dir = output_dir or os.path.join(temp_root, "mtah2027-possession-decoder")
    features = load_features(features_path)
    labels, evidence, diagnostics = decode(
        features, game, margin_scale_m, proximity_scale_m, change_penalty)
    paths = write_outputs(labels, evidence, diagnostics, output_dir)
    return paths, diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", default=None, help="feature archive from feature_audit.py")
    parser.add_argument("--out", default=None, help="output folder (must be outside this repo)")
    parser.add_argument("--margin-scale-m", type=float, default=2.5)
    parser.add_argument("--proximity-scale-m", type=float, default=5.0)
    parser.add_argument("--change-penalty", type=float, default=10.0)
    args = parser.parse_args()

    output_paths, diagnostics = run(
        args.features, args.out, margin_scale_m=args.margin_scale_m,
        proximity_scale_m=args.proximity_scale_m, change_penalty=args.change_penalty)
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print("\nsubmission candidate: {}".format(output_paths[0]))
    print("evidence archive: {}".format(output_paths[1]))
    print("diagnostics: {}".format(output_paths[2]))

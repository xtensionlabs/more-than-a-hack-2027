"""Extract per-frame team-level tracking evidence without using possession labels.

Run from the repository root with the match data configured through MTAH_DATA_DIR. The compressed
feature archive and its aggregate summary are written outside the repository by default.
"""

import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require
from starter.baseline import team_of_each_track

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"
HOME, AWAY = 0, 1


def parse_observation(group, frame_id):
    """Return normalised x/y and speed in m/s, or None for a missing track."""
    if not group or group[0] == MISSING:
        return None

    values = group.split(",")
    if len(values) < 2:
        raise ValueError("malformed observation on ATD frame {}".format(frame_id))

    try:
        x = float(values[0])
        y = float(values[1])
        speed = float(values[2]) if len(values) > 2 else float("nan")
    except ValueError:
        raise ValueError("malformed observation on ATD frame {}".format(frame_id))

    if not math.isfinite(x) or not math.isfinite(y):
        return None
    if not math.isfinite(speed):
        speed = float("nan")
    return x, y, speed


def empty_features(n_frames):
    """Allocate frame-aligned aggregate evidence arrays."""
    return {
        "frame": np.arange(n_frames, dtype=np.int32),
        "ball_seen": np.zeros(n_frames, dtype=np.bool_),
        "ball_x_norm": np.full(n_frames, np.nan, dtype=np.float32),
        "ball_y_norm": np.full(n_frames, np.nan, dtype=np.float32),
        "ball_speed_mps": np.full(n_frames, np.nan, dtype=np.float32),
        "ball_outside_pitch": np.zeros(n_frames, dtype=np.bool_),
        "home_track_count": np.zeros(n_frames, dtype=np.int16),
        "away_track_count": np.zeros(n_frames, dtype=np.int16),
        "home_nearest_distance_m": np.full(n_frames, np.nan, dtype=np.float32),
        "away_nearest_distance_m": np.full(n_frames, np.nan, dtype=np.float32),
        "home_nearest_speed_mps": np.full(n_frames, np.nan, dtype=np.float32),
        "away_nearest_speed_mps": np.full(n_frames, np.nan, dtype=np.float32),
        "distance_margin_m": np.full(n_frames, np.nan, dtype=np.float32),
        "nearest_team": np.full(n_frames, -1, dtype=np.int8),
    }


def extract_features(raw_path, team_of_track, n_frames, pitch):
    """Read ATD once and derive ball and team-level proximity evidence."""
    length, width = pitch
    features = empty_features(n_frames)
    expected_frame = 0

    with open(raw_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            if len(parts) != 3:
                raise ValueError("expected three ATD fields on line {}".format(line_number))

            try:
                frame_id = int(parts[0])
            except ValueError:
                raise ValueError("invalid ATD frame id on line {}".format(line_number))
            if frame_id != expected_frame + 1:
                raise ValueError(
                    "ATD alignment breaks on line {}: expected frame_id {}, found {}".format(
                        line_number, expected_frame + 1, frame_id))
            if expected_frame >= n_frames:
                raise ValueError("ATD contains more than {} frames".format(n_frames))

            players = parts[1].split(GROUP_SEP)
            if len(players) != len(team_of_track):
                raise ValueError(
                    "ATD frame {} has {} track slots; metadata has {}".format(
                        frame_id, len(players), len(team_of_track)))

            ball = parse_observation(parts[2], frame_id)
            if ball is not None:
                ball_x, ball_y, ball_speed = ball
                features["ball_seen"][expected_frame] = True
                features["ball_x_norm"][expected_frame] = ball_x
                features["ball_y_norm"][expected_frame] = ball_y
                features["ball_speed_mps"][expected_frame] = ball_speed
                features["ball_outside_pitch"][expected_frame] = (
                    ball_x < 0.0 or ball_x > 1.0 or ball_y < 0.0 or ball_y > 1.0)
                ball_x_m = ball_x * length
                ball_y_m = ball_y * width
            else:
                ball_x_m = ball_y_m = None

            track_counts = [0, 0]
            nearest_squared = [float("inf"), float("inf")]
            nearest_speed = [float("nan"), float("nan")]

            for slot, group in enumerate(players):
                observation = parse_observation(group, frame_id)
                if observation is None:
                    continue

                player_x, player_y, player_speed = observation
                team = int(team_of_track[slot])
                track_counts[team] += 1
                if ball_x_m is None:
                    continue

                dx = player_x * length - ball_x_m
                dy = player_y * width - ball_y_m
                distance_squared = dx * dx + dy * dy
                if distance_squared < nearest_squared[team]:
                    nearest_squared[team] = distance_squared
                    nearest_speed[team] = player_speed

            features["home_track_count"][expected_frame] = track_counts[HOME]
            features["away_track_count"][expected_frame] = track_counts[AWAY]

            if ball_x_m is not None:
                for team, name in ((HOME, "home"), (AWAY, "away")):
                    if math.isfinite(nearest_squared[team]):
                        distance = math.sqrt(nearest_squared[team])
                        features[name + "_nearest_distance_m"][expected_frame] = distance
                        features[name + "_nearest_speed_mps"][expected_frame] = nearest_speed[team]

                home_distance = features["home_nearest_distance_m"][expected_frame]
                away_distance = features["away_nearest_distance_m"][expected_frame]
                if math.isfinite(float(home_distance)) and math.isfinite(float(away_distance)):
                    features["distance_margin_m"][expected_frame] = away_distance - home_distance
                    if home_distance < away_distance:
                        features["nearest_team"][expected_frame] = HOME
                    elif away_distance < home_distance:
                        features["nearest_team"][expected_frame] = AWAY

            expected_frame += 1

    if expected_frame != n_frames:
        raise ValueError("ATD has {} frames; expected {}".format(expected_frame, n_frames))
    return features


def distribution(values, selected):
    """Finite-value count and percentiles for a boolean-selected subset."""
    data = values[selected & np.isfinite(values)]
    if not len(data):
        return {"count": 0, "median": None, "p90": None}
    median, p90 = np.percentile(data, [50, 90])
    return {"count": int(len(data)), "median": float(median), "p90": float(p90)}


def count_distribution(values, selected):
    data = values[selected]
    if not len(data):
        return {"count": 0, "mean": None, "p95": None}
    mean, p95 = np.mean(data), np.percentile(data, 95)
    return {"count": int(len(data)), "mean": float(mean), "p95": float(p95)}


def build_summary(features, game=GAME):
    """Summarise missingness and team-level evidence without consulting labels."""
    n_frames = len(features["frame"])
    graded = np.zeros(n_frames, dtype=np.bool_)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True

    ball_seen = features["ball_seen"]
    graded_ball_seen = graded & ball_seen
    graded_ball_frames = int(np.sum(graded))
    observed_all = int(np.sum(ball_seen))
    observed_graded = int(np.sum(graded_ball_seen))
    outside = features["ball_outside_pitch"]
    both_teams_nearest = graded_ball_seen & np.isfinite(features["distance_margin_m"])
    nearest_team = features["nearest_team"]

    return {
        "frames": n_frames,
        "graded_frames": graded_ball_frames,
        "ball": {
            "observed_all": observed_all,
            "observed_pct_all": 100.0 * observed_all / n_frames,
            "observed_graded": observed_graded,
            "observed_pct_graded": 100.0 * observed_graded / graded_ball_frames,
            "outside_pitch_observed_all": int(np.sum(outside)),
            "outside_pitch_observed_graded": int(np.sum(graded & outside)),
        },
        "visible_track_slots_per_graded_frame": {
            "home": count_distribution(features["home_track_count"], graded),
            "away": count_distribution(features["away_track_count"], graded),
        },
        "nearest_distance_m_on_ball_seen_graded": {
            "home": distribution(features["home_nearest_distance_m"], graded_ball_seen),
            "away": distribution(features["away_nearest_distance_m"], graded_ball_seen),
        },
        "distance_margin_m_away_minus_home": distribution(
            features["distance_margin_m"], both_teams_nearest),
        "nearest_team_on_ball_seen_graded": {
            "home": int(np.sum(graded_ball_seen & (nearest_team == HOME))),
            "away": int(np.sum(graded_ball_seen & (nearest_team == AWAY))),
            "tie_or_unavailable": int(np.sum(graded_ball_seen & (nearest_team < 0))),
        },
    }


def require_output_outside_repository(path):
    absolute_path = os.path.abspath(path)
    try:
        common = os.path.commonpath((ROOT, absolute_path))
    except ValueError:
        return absolute_path
    if os.path.normcase(common) == os.path.normcase(ROOT):
        raise ValueError("feature outputs must be written outside the repository")
    return absolute_path


def run(data_dir=None, output_dir=None, game=GAME):
    n_frames = GAMES[game]["n_video_frames"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    output_dir = require_output_outside_repository(
        output_dir or os.path.join(tempfile.gettempdir(), "mtah2027-feature-audit"))

    features = extract_features(
        raw_path, team_of_each_track(meta_path), n_frames, GAMES[game]["pitch"])
    summary = build_summary(features, game)
    os.makedirs(output_dir, exist_ok=True)

    archive_path = os.path.join(output_dir, "tracking_features.npz")
    summary_path = os.path.join(output_dir, "tracking_feature_summary.json")
    np.savez_compressed(archive_path, **features)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return archive_path, summary_path, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="output folder (must be outside this repo)")
    args = parser.parse_args()

    archive_path, summary_path, summary = run(output_dir=args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nfeature archive: {}".format(archive_path))
    print("summary: {}".format(summary_path))

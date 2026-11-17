"""Runtime feature and inference code for the externally trained possession model.

Training is kept in ``train_external_model.py`` and uses the licensed IDSSE matches only. This
module has no fit path and never reads labels or evaluation data.
"""

import json
import os

import numpy as np
import pandas as pd


FEATURE_VERSION = 3
OWNER_FEATURE_NAMES = (
    "team_distance_margin", "absolute_distance_margin", "nearest_distance",
    "farthest_distance", "margin_times_nearest", "margin_times_farthest",
    "nearest_player_speed_difference", "team_median_speed_difference",
    "tracked_player_count_difference", "ball_speed", "ball_seen",
    "proximity_difference_2m", "proximity_difference_5m", "proximity_difference_10m",
    "rolling_margin_11f", "rolling_margin_31f", "rolling_speed_difference_11f",
    "second_distance_margin", "absolute_second_distance_margin",
    "second_margin_times_nearest", "proximity_difference_20m",
    "near_player_count_difference_2m", "near_player_count_difference_5m",
    "near_player_count_difference_10m", "near_player_count_difference_20m",
)
OWNER_MIRROR_INDICES = tuple(
    index for index, name in enumerate(OWNER_FEATURE_NAMES)
    if name not in ("absolute_distance_margin", "nearest_distance", "farthest_distance",
                    "absolute_second_distance_margin", "ball_speed", "ball_seen")
)
DEAD_FEATURE_NAMES = (
    "ball_seen", "ball_outside_pitch", "ball_speed", "ball_speed_squared",
    "nearest_distance", "farthest_distance", "median_player_speed",
    "median_player_speed_squared", "stationary_player_fraction", "tracked_player_fraction",
    "ball_seen_times_nearest_distance", "ball_seen_times_ball_speed",
    "rolling_median_player_speed_25f", "rolling_stationary_fraction_25f",
)


def _row_median(values, valid):
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    count = valid.sum(axis=1)
    ordered = np.sort(np.where(valid, values, np.inf), axis=1)
    rows = np.arange(len(values))
    result = np.full(len(values), np.nan, dtype=float)
    has = count > 0
    lo = np.maximum((count - 1) // 2, 0)
    hi = np.maximum(count // 2, 0)
    result[has] = (ordered[rows[has], lo[has]] + ordered[rows[has], hi[has]]) / 2
    return result


def extract_features(players, ball_xy, team, pitch=(105.3, 68.0), fps=25):
    """Create per-frame features from automatic tracking only.

    ``players`` is (frames, slots, 3): x/y in metres and speed in m/s. ``team`` contains 0 for
    home and 1 for away. Ball positions are (frames, 2), with NaNs for missing observations.
    """
    players = np.asarray(players, dtype=float)
    ball_xy = np.asarray(ball_xy, dtype=float)
    team = np.asarray(team, dtype=np.int8)
    if players.ndim != 3 or players.shape[2] < 3:
        raise ValueError("players must have shape (frames, slots, >=3)")
    n_frames, n_slots, _ = players.shape
    if ball_xy.shape != (n_frames, 2) or team.shape != (n_slots,):
        raise ValueError("ball_xy and team must align with players")
    if not np.isin(team, (0, 1)).all() or not np.any(team == 0) or not np.any(team == 1):
        raise ValueError("team must contain home=0 and away=1 slots")

    length, width = map(float, pitch)
    fps = float(fps)
    if not np.isfinite((length, width, fps)).all() or length <= 0 or width <= 0 or fps <= 0:
        raise ValueError("pitch dimensions and frame rate must be finite and positive")
    bx, by = ball_xy[:, 0], ball_xy[:, 1]
    ball_seen = np.isfinite(bx) & np.isfinite(by)
    tracked = np.isfinite(players[:, :, 0]) & np.isfinite(players[:, :, 1])
    speed = players[:, :, 2]
    valid_speed = tracked & np.isfinite(speed)
    counts = {side: tracked[:, team == side].sum(axis=1) for side in (0, 1)}

    result = {
        "ball_seen": ball_seen,
        "ball_outside_pitch": ball_seen & (
            (bx < -0.5) | (bx > length + 0.5) | (by < -0.5) | (by > width + 0.5)),
        "ball_displacement_speed_mps": np.full(n_frames, np.nan, dtype=float),
        "home_track_count": counts[0].astype(float),
        "away_track_count": counts[1].astype(float),
    }

    # Same centered, missing-aware displacement-speed definition is used for external and
    # challenge tracking. Half boundaries are supplied by the caller so smoothing never crosses
    # halftime.
    lag = 5
    if n_frames > lag * 2:
        smooth_x = pd.Series(np.where(ball_seen, bx, np.nan)).rolling(
            11, center=True, min_periods=4).median().to_numpy()
        smooth_y = pd.Series(np.where(ball_seen, by, np.nan)).rolling(
            11, center=True, min_periods=4).median().to_numpy()
        dx = smooth_x[lag * 2:] - smooth_x[:-lag * 2]
        dy = smooth_y[lag * 2:] - smooth_y[:-lag * 2]
        valid = np.isfinite(dx) & np.isfinite(dy)
        speed_out = result["ball_displacement_speed_mps"][lag:-lag]
        speed_out[valid] = np.hypot(dx[valid], dy[valid]) / (2 * lag / fps)

    rows = np.arange(n_frames)
    for side, name in ((0, "home"), (1, "away")):
        slots = np.flatnonzero(team == side)
        side_xy = players[:, slots, :2]
        distances = np.hypot(side_xy[:, :, 0] - bx[:, None],
                             side_xy[:, :, 1] - by[:, None])
        distances[~tracked[:, slots] | ~ball_seen[:, None]] = np.inf
        nearest_slot = distances.argmin(axis=1)
        nearest = distances[rows, nearest_slot]
        if distances.shape[1] > 1:
            second_nearest = np.partition(distances, 1, axis=1)[:, 1]
        else:
            second_nearest = np.full(n_frames, np.inf, dtype=float)
        nearest[~np.isfinite(nearest)] = np.nan
        second_nearest[~np.isfinite(second_nearest)] = np.nan
        near_speed = speed[:, slots][rows, nearest_slot]
        near_speed[~np.isfinite(nearest)] = np.nan
        side_speed = _row_median(speed[:, slots], valid_speed[:, slots])
        result[name + "_nearest_distance_m"] = nearest
        result[name + "_second_nearest_distance_m"] = second_nearest
        result[name + "_nearest_speed_mps"] = near_speed
        result[name + "_player_speed_median_mps"] = side_speed
        for radius in (2, 5, 10, 20):
            count = np.sum(distances <= radius, axis=1).astype(float)
            count[~ball_seen] = np.nan
            result[name + "_players_within_{}m".format(radius)] = count

    result["all_player_speed_median_mps"] = _row_median(speed, valid_speed)
    below = valid_speed & (speed < 0.5)
    speed_count = valid_speed.sum(axis=1)
    fraction = np.full(n_frames, np.nan, dtype=float)
    has_speed = speed_count > 0
    fraction[has_speed] = below.sum(axis=1)[has_speed] / speed_count[has_speed]
    result["player_speed_fraction_below_0_5_mps"] = fraction
    return result


def owner_matrix(features, pitch_length=105.0):
    """Team-symmetric geometric and movement evidence for the live owner model."""
    home = np.asarray(features["home_nearest_distance_m"], dtype=float)
    away = np.asarray(features["away_nearest_distance_m"], dtype=float)
    home_second = np.asarray(features["home_second_nearest_distance_m"], dtype=float)
    away_second = np.asarray(features["away_second_nearest_distance_m"], dtype=float)
    home_speed = np.asarray(features["home_nearest_speed_mps"], dtype=float)
    away_speed = np.asarray(features["away_nearest_speed_mps"], dtype=float)
    home_med = np.asarray(features["home_player_speed_median_mps"], dtype=float)
    away_med = np.asarray(features["away_player_speed_median_mps"], dtype=float)
    home_count = np.asarray(features["home_track_count"], dtype=float)
    away_count = np.asarray(features["away_track_count"], dtype=float)
    ball_speed = np.asarray(features["ball_displacement_speed_mps"], dtype=float)
    ball_seen = np.asarray(features["ball_seen"], dtype=float)

    margin = (away - home) / pitch_length
    nearest = np.minimum(home, away) / pitch_length
    farthest = np.maximum(home, away) / pitch_length
    second_margin = (away_second - home_second) / pitch_length
    absolute_second_margin = np.abs(second_margin)
    margin[~np.isfinite(margin)] = np.nan
    nearest[~np.isfinite(nearest)] = np.nan
    farthest[~np.isfinite(farthest)] = np.nan
    second_margin[~np.isfinite(second_margin)] = np.nan
    absolute_second_margin[~np.isfinite(absolute_second_margin)] = np.nan
    second_both_visible = np.isfinite(home_second) & np.isfinite(away_second)
    both_visible = np.isfinite(home) & np.isfinite(away)
    home_distance = np.where(np.isfinite(home), np.clip(home, 0, 100), np.nan)
    away_distance = np.where(np.isfinite(away), np.clip(away, 0, 100), np.nan)
    proximity_differences = []
    for radius in (2.0, 5.0, 10.0, 20.0):
        difference = (np.exp(-home_distance / radius)
                      - np.exp(-away_distance / radius))
        difference[~both_visible] = np.nan
        proximity_differences.append(difference)
    margin_series = pd.Series(margin)
    rolling_margin_short = margin_series.rolling(11, center=True, min_periods=3).median()
    rolling_margin_long = margin_series.rolling(31, center=True, min_periods=8).median()
    speed_difference = (away_speed - home_speed) / 10.0
    rolling_speed_difference = pd.Series(speed_difference).rolling(
        11, center=True, min_periods=3).median()
    count_differences = []
    for radius in (2, 5, 10, 20):
        home_count = np.asarray(
            features["home_players_within_{}m".format(radius)], dtype=float)
        away_count = np.asarray(
            features["away_players_within_{}m".format(radius)], dtype=float)
        difference = (away_count - home_count) / 22.0
        difference[~(np.isfinite(home_count) & np.isfinite(away_count))] = np.nan
        count_differences.append(difference)
    second_margin_product = second_margin * nearest
    second_margin_product[~second_both_visible] = np.nan
    return np.column_stack((
        margin,
        np.abs(margin),
        nearest,
        farthest,
        margin * nearest,
        margin * farthest,
        (away_speed - home_speed) / 10.0,
        (away_med - home_med) / 10.0,
        (away_count - home_count) / 11.0,
        ball_speed / 20.0,
        ball_seen,
        *proximity_differences[:3],
        rolling_margin_short.to_numpy(),
        rolling_margin_long.to_numpy(),
        rolling_speed_difference.to_numpy(),
        second_margin,
        absolute_second_margin,
        second_margin_product,
        proximity_differences[-1],
        *count_differences,
    ))


def dead_matrix(features, pitch_length=105.0):
    """Symmetric motion and stoppage evidence for the dead-ball model."""
    home = np.asarray(features["home_nearest_distance_m"], dtype=float)
    away = np.asarray(features["away_nearest_distance_m"], dtype=float)
    nearest = np.minimum(home, away) / pitch_length
    farthest = np.maximum(home, away) / pitch_length
    nearest[~np.isfinite(nearest)] = np.nan
    farthest[~np.isfinite(farthest)] = np.nan
    ball_speed = np.asarray(features["ball_displacement_speed_mps"], dtype=float) / 20.0
    ball_seen = np.asarray(features["ball_seen"], dtype=float)
    outside = np.asarray(features["ball_outside_pitch"], dtype=float)
    median_speed = np.asarray(features["all_player_speed_median_mps"], dtype=float) / 10.0
    stationary = np.asarray(features["player_speed_fraction_below_0_5_mps"], dtype=float)
    track_total = (np.asarray(features["home_track_count"], dtype=float)
                   + np.asarray(features["away_track_count"], dtype=float)) / 22.0
    rolling_player_speed = pd.Series(median_speed).rolling(
        25, center=True, min_periods=6).median().to_numpy()
    rolling_stationary = pd.Series(stationary).rolling(
        25, center=True, min_periods=6).median().to_numpy()
    return np.column_stack((
        ball_seen,
        outside,
        ball_speed,
        ball_speed ** 2,
        nearest,
        farthest,
        median_speed,
        median_speed ** 2,
        stationary,
        track_total,
        ball_seen * nearest,
        ball_seen * ball_speed,
        rolling_player_speed,
        rolling_stationary,
    ))


def _standardize(matrix, prep):
    matrix = np.asarray(matrix, dtype=float)
    median = np.asarray(prep["median"], dtype=float)
    mean = np.asarray(prep["mean"], dtype=float)
    scale = np.asarray(prep["scale"], dtype=float)
    if matrix.ndim != 2 or any(values.shape != (matrix.shape[1],)
                               for values in (median, mean, scale)):
        raise ValueError("model preprocessing dimensions do not match feature dimensions")
    if (not np.isfinite(median).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("model preprocessing contains invalid values")
    missing = ~np.isfinite(matrix)
    matrix = np.where(missing, median[None, :], matrix)
    matrix = (matrix - mean[None, :]) / scale[None, :]
    return matrix


def _sigmoid(value):
    value = np.asarray(value, dtype=float)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = 1 / (1 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1 + exp_value)
    return result


def predict_scores(features, model):
    """Return H/A/D log-probability emissions from a frozen external model."""
    if int(model.get("feature_version", -1)) != FEATURE_VERSION:
        raise ValueError("external model feature version is incompatible")
    if (tuple(model.get("owner_feature_names", ())) != OWNER_FEATURE_NAMES
            or tuple(model.get("dead_feature_names", ())) != DEAD_FEATURE_NAMES):
        raise ValueError("external model feature names/order are incompatible")
    owner_x = _standardize(owner_matrix(features), model["owner_preprocess"])
    dead_x = _standardize(dead_matrix(features), model["dead_preprocess"])
    owner_weights = np.asarray(model["owner_weights"], dtype=float)
    dead_weights = np.asarray(model["dead_weights"], dtype=float)
    if owner_weights.shape != (owner_x.shape[1],) or dead_weights.shape != (dead_x.shape[1],):
        raise ValueError("model coefficient dimensions do not match feature dimensions")
    if not np.isfinite(owner_weights).all() or not np.isfinite(dead_weights).all():
        raise ValueError("model coefficients contain non-finite values")
    owner_intercept = float(model["owner_intercept"])
    dead_intercept = float(model["dead_intercept"])
    if not np.isfinite((owner_intercept, dead_intercept)).all():
        raise ValueError("model intercepts contain non-finite values")
    owner = owner_x @ owner_weights + owner_intercept
    dead = dead_x @ dead_weights + dead_intercept
    dead_offset = float(model.get("dead_logit_offset", 0.0))
    if not np.isfinite(dead_offset):
        raise ValueError("dead-state calibration offset is non-finite")
    dead += dead_offset
    p_home = np.clip(_sigmoid(owner), 1e-7, 1 - 1e-7)
    p_dead = np.clip(_sigmoid(dead), 1e-7, 1 - 1e-7)
    log_live = np.log1p(-p_dead)
    return np.column_stack((
        log_live + np.log(p_home),
        log_live + np.log1p(-p_home),
        np.log(p_dead),
    ))


def decode_scores(scores, owner_transition_cost, dead_transition_cost):
    """Decode H/A/D emissions with costs selected by external-match validation."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 3 or not len(scores):
        raise ValueError("scores must be a non-empty (frames, 3) array")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    owner_cost = float(owner_transition_cost)
    dead_cost = float(dead_transition_cost)
    if (not np.isfinite((owner_cost, dead_cost)).all()
            or owner_cost < 0 or dead_cost < 0):
        raise ValueError("transition costs must be finite and non-negative")
    transitions = np.array(((0.0, owner_cost, dead_cost),
                            (owner_cost, 0.0, dead_cost),
                            (dead_cost, dead_cost, 0.0)), dtype=float)
    costs = -scores[0].copy()
    previous = np.zeros((len(scores), 3), dtype=np.int8)
    states = np.empty(len(scores), dtype=np.int8)
    for frame in range(1, len(scores)):
        candidates = costs[:, None] + transitions
        previous[frame] = candidates.argmin(axis=0)
        costs = -scores[frame] + candidates[previous[frame], np.arange(3)]
    states[-1] = int(costs.argmin())
    for frame in range(len(scores) - 1, 0, -1):
        states[frame - 1] = previous[frame, states[frame]]
    return states


def load_model(path=None):
    path = path or os.path.join(os.path.dirname(__file__), "external_model.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


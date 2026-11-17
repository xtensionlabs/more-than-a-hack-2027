"""Fit and match-holdout-validate the possession model on licensed external IDSSE data only.

The input must be the DFL/IDSSE position and match-information XML files for other matches.
This script never reads the challenge data, public/private challenge ground truth, or video.
The dataset is IDSSE (DFL tracking with BallPossession/BallStatus), CC BY 4.0; see the citation
metadata written to the model file and the IDSSE paper, doi:10.1038/s41597-025-04505-y.

Example:
    python starter/train_external_model.py --data-dir <external_idsse_dir> --out <temp_dir>
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from starter.external_model import (
    DEAD_FEATURE_NAMES,
    FEATURE_VERSION,
    OWNER_MIRROR_INDICES,
    OWNER_FEATURE_NAMES,
    decode_scores,
    dead_matrix,
    extract_features,
    owner_matrix,
    predict_scores,
)


MATCHES = {
    "J03WMX": ("DFL-COM-000001", "51643475", "51643514"),
    "J03WN1": ("DFL-COM-000001", "51643472", "51643517"),
    "J03WOH": ("DFL-COM-000002", "51643478", "51643520"),
    "J03WOY": ("DFL-COM-000002", "51643481", "51643523"),
    "J03WPY": ("DFL-COM-000002", "51643487", "51643526"),
    "J03WQQ": ("DFL-COM-000002", "51643484", "51643529"),
    "J03WR9": ("DFL-COM-000002", "51643490", "51643532"),
}
FPS = 25
DEAD, HOME, AWAY = 2, 0, 1
HOLDOUT_MATCH = "J03WN1"


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _paths(data_dir, match_id):
    competition, _, _ = MATCHES[match_id]
    stem = "DFL-MAT-" + match_id
    names = (
        (match_id + "-info.xml",
         "DFL_02_01_matchinformation_{}_{}.xml".format(competition, stem)),
        (match_id + "-positions.xml",
         "DFL_04_03_positions_raw_observed_{}_{}.xml".format(competition, stem)),
    )
    return tuple(next((os.path.join(data_dir, name) for name in candidates
                       if os.path.isfile(os.path.join(data_dir, name))),
                      os.path.join(data_dir, candidates[0]))
                 for candidates in names)


def _read_info(path, expected_match_id=None):
    root = ET.parse(path).getroot()
    general = next(element for element in root.iter()
                   if _tag(element) == "General")
    actual_match_id = general.get("MatchId")
    if (expected_match_id and actual_match_id
            and actual_match_id != expected_match_id):
        raise ValueError("match-information file is for {}, expected {}".format(
            actual_match_id, expected_match_id))
    home_id = general.get("HomeTeamId")
    away_id = general.get("GuestTeamId")
    environment = next((element for element in root.iter()
                        if _tag(element) == "Environment"), None)
    pitch_source = environment if environment is not None else general
    pitch = (float(pitch_source.get("PitchX", 105.0)),
             float(pitch_source.get("PitchY", 68.0)))
    player_ids = {home_id: [], away_id: []}
    if not home_id or not away_id:
        for element in root.iter():
            if _tag(element) != "Team":
                continue
            role = element.get("Role", "").lower()
            if role == "home":
                home_id = element.get("TeamId")
            elif role in ("guest", "away"):
                away_id = element.get("TeamId")
        if not home_id or not away_id:
            raise ValueError("match information is missing home/guest team ids")
        player_ids = {home_id: [], away_id: []}
    for element in root.iter():
        if _tag(element) != "Team" or element.get("TeamId") not in player_ids:
            continue
        team_id = element.get("TeamId")
        for player in element.iter():
            if _tag(player) == "Player" and player.get("PersonId"):
                player_ids[team_id].append(player.get("PersonId"))
    if not home_id or not away_id or not player_ids[home_id] or not player_ids[away_id]:
        raise ValueError("match information is missing the home/away roster")
    return home_id, away_id, player_ids, pitch


def _frame_bounds(path, expected_match_id=None):
    bounds = {}
    for _event, frame_set in ET.iterparse(path, events=("end",)):
        if _tag(frame_set) != "FrameSet":
            continue
        actual_match_id = frame_set.get("MatchId")
        if (expected_match_id and actual_match_id
                and actual_match_id != expected_match_id):
            raise ValueError("position file contains {}, expected {}".format(
                actual_match_id, expected_match_id))
        frames = frame_set.findall("Frame")
        if frames:
            section = frame_set.get("GameSection")
            first, last = int(frames[0].get("N")), int(frames[-1].get("N"))
            if section in bounds:
                bounds[section] = (min(first, bounds[section][0]),
                                   max(last, bounds[section][1]))
            else:
                bounds[section] = (first, last)
        frame_set.clear()
    return bounds


def load_match(data_dir, match_id):
    """Read tracking and machine-generated possession/dead-ball codes from one external match."""
    info_path, position_path = _paths(data_dir, match_id)
    if not os.path.isfile(info_path) or not os.path.isfile(position_path):
        raise FileNotFoundError("missing IDSSE match files for {} in {}".format(
            match_id, data_dir))
    expected_match_id = "DFL-MAT-" + match_id
    home_id, away_id, player_ids, pitch = _read_info(info_path, expected_match_id)
    bounds = _frame_bounds(position_path, expected_match_id)
    if not bounds:
        raise ValueError("position file has no FrameSet elements")

    side_by_team = {home_id: 0, away_id: 1}
    slot_by_person = {
        0: {person: slot for slot, person in enumerate(player_ids[home_id])},
        1: {person: slot for slot, person in enumerate(player_ids[away_id])},
    }
    arrays = {}
    for section, (first, last) in bounds.items():
        n = last - first + 1
        arrays[section] = {
            "first": first,
            "players": np.full((n, len(player_ids[home_id]) + len(player_ids[away_id]), 3),
                               np.nan, dtype=np.float32),
            "ball": np.full((n, 2), np.nan, dtype=np.float32),
            "ball_status": np.full(n, -1, dtype=np.int8),
            "possession": np.full(n, -1, dtype=np.int8),
        }
        arrays[section]["team"] = np.r_[
            np.zeros(len(player_ids[home_id]), dtype=np.int8),
            np.ones(len(player_ids[away_id]), dtype=np.int8)]

    for _event, frame_set in ET.iterparse(position_path, events=("end",)):
        if _tag(frame_set) != "FrameSet":
            continue
        section = frame_set.get("GameSection")
        if section not in arrays:
            frame_set.clear()
            continue
        section_data = arrays[section]
        first = section_data["first"]
        team_id = frame_set.get("TeamId", "")
        is_ball = team_id.lower() == "ball"
        side = side_by_team.get(team_id)
        person_slots = slot_by_person.get(side, {}) if side is not None else {}
        person_slot = person_slots.get(frame_set.get("PersonId"))
        if is_ball or person_slot is not None:
            target = section_data["ball"] if is_ball else section_data["players"]
            if not is_ball:
                n_home = len(player_ids[home_id])
                column = person_slot if side == 0 else n_home + person_slot
            for frame in frame_set.findall("Frame"):
                index = int(frame.get("N")) - first
                if index < 0 or index >= len(section_data["ball"]):
                    raise ValueError("frame id falls outside its GameSection bounds")
                x = frame.get("X")
                y = frame.get("Y")
                if x is not None and y is not None:
                    # DFL positions use a centre-origin pitch. Translate only; all geometry is
                    # thereby in the same 0..length, 0..width convention as the challenge ATD.
                    if is_ball:
                        target[index, 0] = float(x) + pitch[0] / 2
                        target[index, 1] = float(y) + pitch[1] / 2
                    else:
                        target[index, column, 0] = float(x) + pitch[0] / 2
                        target[index, column, 1] = float(y) + pitch[1] / 2
                if not is_ball and frame.get("S") is not None:
                    target[index, column, 2] = float(frame.get("S"))
                if is_ball:
                    status, possession = frame.get("BallStatus"), frame.get("BallPossession")
                    if status is not None:
                        section_data["ball_status"][index] = int(float(status))
                    if possession is not None:
                        section_data["possession"][index] = int(float(possession))
        frame_set.clear()

    outputs = []
    section_order = {"firstHalf": 0, "secondHalf": 1,
                     "extraTime1": 2, "extraTime2": 3, "penalties": 4}
    for section in sorted(arrays, key=lambda value: (section_order.get(value, 99), value)):
        item = arrays[section]
        features = extract_features(item["players"], item["ball"], item["team"], pitch, FPS)
        labels = np.full(len(item["ball_status"]), -1, dtype=np.int8)
        labels[item["ball_status"] == 0] = DEAD
        alive = item["ball_status"] == 1
        labels[alive & (item["possession"] == 1)] = HOME
        labels[alive & (item["possession"] == 2)] = AWAY
        outputs.append({"section": section, "features": features, "labels": labels})
    return outputs


def _cache_file(cache_dir, match_id):
    return os.path.join(cache_dir, "{}_features_v{}.npz".format(match_id, FEATURE_VERSION))


def _source_stamp(path):
    stat = os.stat(path)
    return (int(stat.st_size), int(stat.st_mtime_ns))


def _load_feature_cache(path, match_id, info_path, position_path):
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            if (int(cache["feature_version"][0]) != FEATURE_VERSION
                    or str(cache["match_id"][0]) != match_id
                    or not np.array_equal(cache["source_info"], _source_stamp(info_path))
                    or not np.array_equal(cache["source_positions"],
                                          _source_stamp(position_path))):
                return None
            segments = []
            for index in range(int(cache["segment_count"][0])):
                prefix = "segment_{}_feature_".format(index)
                features = {key[len(prefix):]: cache[key].copy()
                            for key in cache.files if key.startswith(prefix)}
                labels = cache["segment_{}_labels".format(index)].copy()
                if not features or any(len(values) != len(labels)
                                       for values in features.values()):
                    return None
                segments.append({
                    "section": str(cache["segment_{}_section".format(index)][0]),
                    "features": features,
                    "labels": labels,
                })
            return segments
    except (OSError, KeyError, ValueError, EOFError):
        return None


def _save_feature_cache(path, match_id, segments, info_path, position_path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cache = {
        "feature_version": np.asarray([FEATURE_VERSION], dtype=np.int16),
        "match_id": np.asarray([match_id]),
        "source_info": np.asarray(_source_stamp(info_path), dtype=np.int64),
        "source_positions": np.asarray(_source_stamp(position_path), dtype=np.int64),
        "segment_count": np.asarray([len(segments)], dtype=np.int16),
    }
    for index, segment in enumerate(segments):
        cache["segment_{}_section".format(index)] = np.asarray([segment["section"]])
        cache["segment_{}_labels".format(index)] = segment["labels"]
        for key, values in segment["features"].items():
            cache["segment_{}_feature_{}".format(index, key)] = values
    np.savez_compressed(path, **cache)


def _fit_preprocess(matrix):
    matrix = np.asarray(matrix, dtype=float)
    median = np.nanmedian(matrix, axis=0)
    median[~np.isfinite(median)] = 0.0
    clean = np.where(np.isfinite(matrix), matrix, median[None, :])
    mean = clean.mean(axis=0)
    scale = clean.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {"median": median, "mean": mean, "scale": scale}, (clean - mean) / scale


def _fit_logistic(matrix, target, penalty):
    target = np.asarray(target, dtype=float)
    prep, x = _fit_preprocess(matrix)
    classes, counts = np.unique(target, return_counts=True)
    if len(classes) < 2:
        raise ValueError("external training fold contains only one class")
    class_weight = {label: len(target) / (len(classes) * count)
                    for label, count in zip(classes, counts)}
    sample_weight = np.asarray([class_weight[label] for label in target], dtype=float)
    weight_sum = sample_weight.sum()

    def objective(parameters):
        coefficients, intercept = parameters[:-1], parameters[-1]
        logits = x @ coefficients + intercept
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        loss = np.dot(sample_weight, np.logaddexp(0.0, logits) - target * logits) / weight_sum
        loss += 0.5 * penalty * np.dot(coefficients, coefficients)
        error = (probability - target) * sample_weight
        gradient = np.r_[x.T @ error / weight_sum + penalty * coefficients,
                         error.sum() / weight_sum]
        return loss, gradient

    initial = np.zeros(x.shape[1] + 1, dtype=float)
    result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                      options={"maxiter": 250, "ftol": 1e-9})
    if not result.success and result.status != 1:
        raise RuntimeError("external logistic fit failed: {}".format(result.message))
    if not np.isfinite(result.x).all():
        raise RuntimeError("external logistic fit produced non-finite coefficients")
    return {"preprocess": {k: v.tolist() for k, v in prep.items()},
            "weights": result.x[:-1].tolist(), "intercept": float(result.x[-1])}


def fit_model(match_segments, penalty):
    owner_x, owner_y, dead_x, dead_y = [], [], [], []
    for segment in match_segments:
        labels = segment["labels"]
        known = labels >= 0
        live = known & np.isin(labels, (HOME, AWAY))
        owner_x.append(owner_matrix(segment["features"])[live])
        owner_y.append((labels[live] == HOME).astype(np.int8))
        dead_x.append(dead_matrix(segment["features"])[known])
        dead_y.append((labels[known] == DEAD).astype(np.int8))
    owner_x = np.concatenate(owner_x)
    owner_y = np.concatenate(owner_y)
    # Mirroring flips only team-signed features. Distances, absolute margins, ball speed and
    # visibility retain their values, so the model cannot learn a home-side preference.
    mirrored = owner_x.copy()
    mirrored[:, OWNER_MIRROR_INDICES] *= -1
    owner_x = np.concatenate((owner_x, mirrored), axis=0)
    owner_y = np.concatenate((owner_y, 1 - owner_y), axis=0)
    dead_x = np.concatenate(dead_x)
    dead_y = np.concatenate(dead_y)
    owner = _fit_logistic(owner_x, owner_y, penalty)
    dead = _fit_logistic(dead_x, dead_y, penalty)
    return {
        "feature_version": FEATURE_VERSION,
        "owner_feature_names": list(OWNER_FEATURE_NAMES),
        "dead_feature_names": list(DEAD_FEATURE_NAMES),
        "owner_preprocess": owner["preprocess"],
        "owner_weights": owner["weights"],
        "owner_intercept": owner["intercept"],
        "dead_preprocess": dead["preprocess"],
        "dead_weights": dead["weights"],
        "dead_intercept": dead["intercept"],
        "training_penalty": float(penalty),
    }


def _viterbi(emissions, owner_penalty, dead_penalty):
    return decode_scores(emissions, owner_penalty, dead_penalty)


def _spells(states):
    starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
    stops = np.r_[starts[1:], len(states)]
    return [(int(start), int(stop), int(states[start]))
            for start, stop in zip(starts, stops) if states[start] != DEAD]


def _known_windows(mask):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    stops = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
    return zip(starts.tolist(), stops.tolist())


def _macro_f1(predicted, reference):
    values = []
    for label in (HOME, AWAY, DEAD):
        tp = np.sum((predicted == label) & (reference == label))
        fp = np.sum((predicted == label) & (reference != label))
        fn = np.sum((predicted != label) & (reference == label))
        values.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(values))


def _evaluate_with_emissions(segments, emissions, owner_penalty, dead_penalty):
    predictions, references = [], []
    overlaps = []
    n_truth_spells = n_guess_spells = 0
    time_offset = 0
    for segment, segment_emissions in zip(segments, emissions):
        labels = segment["labels"]
        valid = labels >= 0
        for start, stop in _known_windows(valid):
            state = _viterbi(segment_emissions[start:stop], owner_penalty, dead_penalty)
            truth = labels[start:stop]
            predictions.append(state)
            references.append(truth)
            truth_spells = _spells(truth)
            guess_spells = _spells(state)
            for i, (gt_start, gt_stop, label) in enumerate(truth_spells):
                for j, (pred_start, pred_stop, other) in enumerate(guess_spells):
                    if label == other:
                        a, b, c, d = (gt_start + time_offset, gt_stop + time_offset,
                                      pred_start + time_offset, pred_stop + time_offset)
                        intersection = max(0, min(b, d) - max(a, c))
                        union = (b - a) + (d - c) - intersection
                        if union:
                            overlaps.append((intersection / union,
                                             n_truth_spells + i,
                                             n_guess_spells + j))
            n_truth_spells += len(truth_spells)
            n_guess_spells += len(guess_spells)
            time_offset += stop - start + 1
    if not references:
        raise ValueError("validation data has no labeled frames")
    pred = np.concatenate(predictions)
    ref = np.concatenate(references)
    overlaps.sort(key=lambda item: (-item[0], item[1], item[2]))
    per_iou = {}
    f1_values = []
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        used_truth = np.zeros(n_truth_spells, dtype=bool)
        used_guess = set()
        true_positives = 0
        for iou, truth_index, guess_index in overlaps:
            if iou < threshold:
                break
            if used_truth[truth_index] or guess_index in used_guess:
                continue
            used_truth[truth_index] = True
            used_guess.add(guess_index)
            true_positives += 1
        precision = true_positives / max(n_guess_spells, 1)
        recall = true_positives / max(n_truth_spells, 1)
        f1 = (2 * precision * recall / max(precision + recall, 1e-12))
        per_iou[str(threshold)] = float(f1)
        f1_values.append(f1)
    return {"macro_f1": _macro_f1(pred, ref),
            "spell_f1_mean": float(np.mean(f1_values)),
            "spell_f1_at_iou_0_5": per_iou["0.5"],
            "spell_f1_per_iou": per_iou}


def evaluate(model, segments, owner_penalty, dead_penalty):
    emissions = [predict_scores(segment["features"], model) for segment in segments]
    return _evaluate_with_emissions(segments, emissions, owner_penalty, dead_penalty)


def _validate_by_match(matches):
    # The first external-only sweep selected the lightest L2 penalty from its coarse grid and a
    # neutral dead-state offset. Recheck the neighboring L2 scales with the expanded temporal
    # range while holding the well-supported offset and owner transition cost fixed.
    penalties = (1e-5, 1e-4, 1e-3)
    dead_offsets = (0.0,)
    owner_costs = (12.0,)
    dead_costs = (12.0, 24.0, 48.0, 96.0, 192.0)
    fold_reports = []
    if len(matches) < 2:
        raise ValueError("at least two complete external matches are required")
    for validation_id in sorted(matches):
        print("tuning external fold; hold out {}".format(validation_id), flush=True)
        best = None
        train = [segment for match_id, segments in matches.items()
                 if match_id != validation_id for segment in segments]
        valid = matches[validation_id]
        for penalty in penalties:
            model = fit_model(train, penalty)
            for dead_offset in dead_offsets:
                calibrated_model = {**model, "dead_logit_offset": dead_offset}
                emissions = [predict_scores(segment["features"], calibrated_model)
                             for segment in valid]
                for owner_cost in owner_costs:
                    for dead_cost in dead_costs:
                        result = _evaluate_with_emissions(
                            valid, emissions, owner_cost, dead_cost)
                        candidate = (result["spell_f1_mean"], result["macro_f1"],
                                     -abs(dead_offset), -penalty, -owner_cost, -dead_cost)
                        if best is None or candidate > best[0]:
                            best = (candidate, {
                                "training_matches": sorted(
                                    match_id for match_id in matches
                                    if match_id != validation_id),
                                "validation_match": validation_id,
                                "logistic_l2": penalty,
                                "dead_logit_offset": dead_offset,
                                "owner_transition_cost": owner_cost,
                                "dead_transition_cost": dead_cost,
                                **result,
                            })
        fold_reports.append(best[1])
    return fold_reports


def _validate_holdout(matches, holdout_id, selected):
    train = [segment for match_id, segments in matches.items()
             if match_id != holdout_id for segment in segments]
    valid = matches[holdout_id]
    model = fit_model(train, selected["logistic_l2"])
    model["dead_logit_offset"] = selected["dead_logit_offset"]
    emissions = [predict_scores(segment["features"], model) for segment in valid]
    result = _evaluate_with_emissions(
        valid, emissions, selected["owner_transition_cost"],
        selected["dead_transition_cost"])
    return {
        "training_matches": sorted(match_id for match_id in matches if match_id != holdout_id),
        "validation_match": holdout_id,
        **selected,
        **result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True,
                        help="directory containing the external IDSSE XML files")
    parser.add_argument("--out", required=True,
                        help="directory outside this repository for weights/reports")
    parser.add_argument("--cache-dir",
                        help="optional directory for reusable external-only feature arrays")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    matches = {}
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    for match_id in MATCHES:
        info_path, position_path = _paths(args.data_dir, match_id)
        if os.path.isfile(info_path) and os.path.isfile(position_path):
            cached = (_load_feature_cache(
                _cache_file(args.cache_dir, match_id), match_id, info_path, position_path)
                if args.cache_dir else None)
            if cached is not None:
                print("loading cached external features {}".format(match_id), flush=True)
                matches[match_id] = cached
            else:
                print("loading external match {}".format(match_id), flush=True)
                matches[match_id] = load_match(args.data_dir, match_id)
                if args.cache_dir:
                    _save_feature_cache(
                        _cache_file(args.cache_dir, match_id), match_id,
                        matches[match_id], info_path, position_path)
    missing_matches = sorted(set(MATCHES) - set(matches))
    if missing_matches:
        raise FileNotFoundError("missing complete external IDSSE matches: {}".format(
            ", ".join(missing_matches)))
    for match_id, segments in sorted(matches.items()):
        labels = np.concatenate([segment["labels"] for segment in segments])
        known = labels >= 0
        counts = {"home": int(np.sum(labels == HOME)),
                  "away": int(np.sum(labels == AWAY)),
                  "dead": int(np.sum(labels == DEAD))}
        print("{}: {} sections, {:,} frames, {:,} labeled ({:.1%}), {}".format(
            match_id, len(segments), len(labels), int(known.sum()),
            float(known.mean()), counts))
    tuning_matches = {match_id: segments for match_id, segments in matches.items()
                      if match_id != HOLDOUT_MATCH}
    folds = _validate_by_match(tuning_matches)
    selected_penalty = float(np.median([fold["logistic_l2"] for fold in folds]))
    selected_dead_offset = float(np.median([fold["dead_logit_offset"] for fold in folds]))
    selected_owner_cost = float(np.median([fold["owner_transition_cost"] for fold in folds]))
    selected_dead_cost = float(np.median([fold["dead_transition_cost"] for fold in folds]))
    selected = {"logistic_l2": selected_penalty,
                "dead_logit_offset": selected_dead_offset,
                "owner_transition_cost": selected_owner_cost,
                "dead_transition_cost": selected_dead_cost}
    print("checking untouched external holdout {}".format(HOLDOUT_MATCH), flush=True)
    holdout_report = _validate_holdout(matches, HOLDOUT_MATCH, selected)
    all_segments = [segment for segments in matches.values() for segment in segments]
    print("fitting final model on all external matches", flush=True)
    final_model = fit_model(all_segments, selected_penalty)
    final_model.update({
        "source": "IDSSE, DFL / Sportec Solutions, CC BY 4.0",
        "source_doi": "10.6084/m9.figshare.28196177",
        "source_paper_doi": "10.1038/s41597-025-04505-y",
        "license": "CC BY 4.0",
        "external_label_source": "DFL BallStatus and BallPossession codes",
        "selection_metric": "mean possession-spell F1 at IoU 0.3, 0.4, 0.5, 0.6, 0.7",
        "training_matches": sorted(matches),
        "tuning_matches": sorted(tuning_matches),
        "holdout_match": HOLDOUT_MATCH,
        "owner_transition_cost": selected_owner_cost,
        "dead_transition_cost": selected_dead_cost,
        "dead_logit_offset": selected_dead_offset,
        "fps": FPS,
        "pitch": [105.0, 68.0],
        "validation": holdout_report,
        "tuning_folds": folds,
        "training_label_counts": {
            label: int(sum(np.sum(segment["labels"] == state)
                           for segments in matches.values() for segment in segments))
            for state, label in ((HOME, "home"), (AWAY, "away"), (DEAD, "dead"))
        },
    })
    with open(os.path.join(args.out, "external_model.json"), "w", encoding="utf-8") as handle:
        json.dump(final_model, handle, indent=2, sort_keys=True)
        handle.write("\n")
    report = {"source": "IDSSE, DFL / Sportec Solutions",
              "source_doi": "10.6084/m9.figshare.28196177",
              "source_paper_doi": "10.1038/s41597-025-04505-y",
              "license": "CC BY 4.0",
              "holdout": holdout_report,
              "tuning_folds": folds,
              "selected": selected}
    with open(os.path.join(args.out, "external_validation.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("weights:", os.path.join(args.out, "external_model.json"))


if __name__ == "__main__":
    main()

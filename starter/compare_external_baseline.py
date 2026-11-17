"""Compare the external-only model with nearest-player assignment on IDSSE matches only.

This is an external diagnostic. It never reads the challenge bundle, its video, or its labels.
The baseline assigns the closer team's nearest track when the ball is observed and D otherwise.

Example:
    python starter/compare_external_baseline.py --data-dir <idsse_dir> \
        --cache-dir <external_feature_cache> --out <external_comparison.json>
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from starter.external_model import predict_scores
from starter.train_external_model import (
    AWAY,
    DEAD,
    HOME,
    MATCHES,
    _cache_file,
    _evaluate_with_emissions,
    _load_feature_cache,
    _paths,
    _save_feature_cache,
    fit_model,
    load_match,
)


def _load_matches(data_dir, cache_dir):
    matches = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    for match_id in MATCHES:
        info_path, position_path = _paths(data_dir, match_id)
        if not os.path.isfile(info_path) or not os.path.isfile(position_path):
            raise FileNotFoundError("missing external IDSSE files for {}".format(match_id))
        segments = (_load_feature_cache(
            _cache_file(cache_dir, match_id), match_id, info_path, position_path)
            if cache_dir else None)
        if segments is None:
            print("loading external match {}".format(match_id), flush=True)
            segments = load_match(data_dir, match_id)
            if cache_dir:
                _save_feature_cache(
                    _cache_file(cache_dir, match_id), match_id, segments,
                    info_path, position_path)
        else:
            print("loading cached external features {}".format(match_id), flush=True)
        matches[match_id] = segments
    return matches


def _nearest_player_emissions(segments):
    emissions = []
    for segment in segments:
        features = segment["features"]
        home = np.asarray(features["home_nearest_distance_m"], dtype=float)
        away = np.asarray(features["away_nearest_distance_m"], dtype=float)
        seen = np.asarray(features["ball_seen"], dtype=bool)
        home_seen, away_seen = np.isfinite(home), np.isfinite(away)
        states = np.full(len(seen), DEAD, dtype=np.int8)
        states[seen & home_seen & ~away_seen] = HOME
        states[seen & away_seen & ~home_seen] = AWAY
        both = seen & home_seen & away_seen
        states[both] = np.where(home[both] <= away[both], HOME, AWAY)
        scores = np.full((len(states), 3), -1000.0)
        scores[np.arange(len(states)), states] = 0.0
        emissions.append(scores)
    return emissions


def _score_fold(matches, validation_match, training_match_ids, settings):
    training = [segment for match_id in training_match_ids
                for segment in matches[match_id]]
    validation = matches[validation_match]
    fitted = fit_model(training, float(settings["logistic_l2"]))
    fitted["dead_logit_offset"] = float(settings["dead_logit_offset"])
    model_emissions = [predict_scores(segment["features"], fitted)
                       for segment in validation]
    model_score = _evaluate_with_emissions(
        validation, model_emissions,
        float(settings["owner_transition_cost"]),
        float(settings["dead_transition_cost"]))
    nearest_score = _evaluate_with_emissions(
        validation, _nearest_player_emissions(validation), 0.0, 0.0)
    return {"validation_match": validation_match,
            "model": model_score,
            "nearest_player": nearest_score}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True,
                        help="directory containing the seven external IDSSE XML matches")
    parser.add_argument("--cache-dir",
                        help="optional external-only feature cache directory")
    parser.add_argument("--model", default=os.path.join("starter", "external_model.json"))
    parser.add_argument("--out", required=True,
                        help="where to write the external baseline comparison JSON")
    args = parser.parse_args()

    with open(args.model, "r", encoding="utf-8") as handle:
        model = json.load(handle)
    if (model.get("source_doi") != "10.6084/m9.figshare.28196177"
            or model.get("license") != "CC BY 4.0"
            or set(model.get("training_matches", [])) != set(MATCHES)):
        raise ValueError("model is missing the approved external-only provenance")

    matches = _load_matches(args.data_dir, args.cache_dir)
    rows = []
    for fold in model["tuning_folds"]:
        rows.append(_score_fold(
            matches, fold["validation_match"], fold["training_matches"], fold))
    holdout = model["validation"]
    selected = {
        "logistic_l2": model["training_penalty"],
        "dead_logit_offset": model["dead_logit_offset"],
        "owner_transition_cost": model["owner_transition_cost"],
        "dead_transition_cost": model["dead_transition_cost"],
    }
    holdout_row = _score_fold(
        matches, holdout["validation_match"], holdout["training_matches"],
        selected)

    report = {
        "comparison": "external IDSSE only; no challenge match data",
        "baseline": "nearest team track to visible ball; D if ball or all player tracks are missing",
        "source_doi": model["source_doi"],
        "license": model["license"],
        "selection_metric": model["selection_metric"],
        "tuning_fold_average": {
            name: {
                metric: float(np.mean([row[name][metric] for row in rows]))
                for metric in ("spell_f1_mean", "spell_f1_at_iou_0_5", "macro_f1")
            }
            for name in ("model", "nearest_player")
        },
        "tuning_folds": rows,
        "development_holdout": holdout_row,
        "holdout_note": (
            "Excluded from fitting and parameter selection; inspected during iterative development, "
            "so this is not a pristine final test estimate."
        ),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report["tuning_fold_average"], indent=2, sort_keys=True))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

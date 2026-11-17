"""Create a full-match submission with a model fit and tuned on external IDSSE matches only.

    python run_full.py [--model starter/external_model.json] [--out submission/Xtenders.csv]

The production path reads automatic tracking only. It does not load challenge video, SmartTagging,
possession labels, or any evaluation ground truth.
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "starter"))
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from external_model import (  # noqa: E402
    decode_scores,
    extract_features,
    load_model,
    predict_scores,
)
from rawatd import load_raw_atd, n_slots_from_meta  # noqa: E402


def _team_of_each_track(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2 or any(not team_id for team_id in team_ids):
        raise ValueError("tracking metadata must declare exactly two teams")
    home_id = team_ids[0]
    slot_teams = [player.get("teamId") for player in root.iter("Player")]
    if not slot_teams or any(team_id not in team_ids for team_id in slot_teams):
        raise ValueError("every tracking slot must map to one of the two teams")
    return np.asarray([0 if team_id == home_id else 1 for team_id in slot_teams],
                      dtype=np.int8)


def _to_letters(states):
    states = np.asarray(states, dtype=np.int8)
    if not np.isin(states, (0, 1, 2)).all():
        raise ValueError("decoded states must be H=0, A=1, or D=2")
    return np.asarray(("H", "A", "D"), dtype="<U1")[states]


def _write_labels(n_total, blocks, decoded):
    labels = np.full(n_total, "X", dtype="<U1")
    if len(blocks) != len(decoded):
        raise ValueError("each video block must have one decoded label sequence")
    for (half, start, stop), sequence in zip(blocks, decoded):
        sequence = np.asarray(sequence, dtype="<U1")
        if (start < 0 or stop < start or stop >= n_total
                or len(sequence) != stop - start + 1):
            raise ValueError("decoded labels do not align with half {}".format(half))
        labels[start:stop + 1] = sequence
    return labels


def _validate_labels(labels, n_total, blocks):
    labels = np.asarray(labels)
    if len(labels) != n_total or not np.isin(labels, ("H", "A", "D", "X")).all():
        raise ValueError("submission labels have invalid length or label values")
    for half, start, stop in blocks:
        if np.any(labels[start:stop + 1] == "X"):
            raise ValueError("X appears inside graded half {}".format(half))
    values, counts = np.unique(labels, return_counts=True)
    print("OK", dict(zip(values, counts)))


def _print_summary(labels, blocks):
    fps = float(GAMES[GAME]["fps"])
    for half, start, stop in blocks:
        segment = labels[start:stop + 1]
        edges = np.r_[0, np.flatnonzero(segment[1:] != segment[:-1]) + 1,
                      len(segment)]
        lengths = np.asarray([
            end - begin for begin, end in zip(edges[:-1], edges[1:])
            if segment[begin] in ("H", "A")
        ], dtype=float) / fps
        shares = {label: round(float(np.mean(segment == label)), 3)
                  for label in ("H", "A", "D")}
        median = float(np.median(lengths)) if len(lengths) else 0.0
        short_share = float(np.mean(lengths < 2.0)) if len(lengths) else 0.0
        print("half {}: {} possessions={} median={:.1f}s share<2s={:.0%}".format(
            half, shares, len(lengths), median, short_share))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="submission/Xtenders.csv")
    parser.add_argument("--model", default=os.path.join("starter", "external_model.json"))
    args = parser.parse_args()

    model = load_model(args.model)
    training_matches = model.get("training_matches", [])
    allowed_external_matches = {
        "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"
    }
    if (not training_matches or len(training_matches) != len(set(training_matches))
            or set(training_matches) != allowed_external_matches
            or GAME in training_matches):
        raise ValueError("model must be trained on all seven approved external matches only")
    if (model.get("source_doi") != "10.6084/m9.figshare.28196177"
            or model.get("source_paper_doi") != "10.1038/s41597-025-04505-y"
            or model.get("license") != "CC BY 4.0"
            or model.get("selection_metric") !=
            "mean possession-spell F1 at IoU 0.3, 0.4, 0.5, 0.6, 0.7"):
        raise ValueError("model does not cite the approved external training source")
    if (model.get("holdout_match") != "J03WN1"
            or model.get("validation", {}).get("validation_match") != "J03WN1"
            or set(model.get("tuning_matches", [])) != allowed_external_matches - {"J03WN1"}
            or not model.get("tuning_folds")
            or not model.get("training_label_counts")):
        raise ValueError("model is missing external-only validation provenance")

    meta, raw = require(["atd_meta", "atd_raw"])
    expected_frames = int(GAMES[GAME]["n_video_frames"])
    ids, players, ball = load_raw_atd(
        raw, n_slots_from_meta(meta), expected_frames=expected_frames)
    n_frames = len(ids)
    if (n_frames != expected_frames or ids[0] != 1
            or not np.array_equal(ids, np.arange(1, n_frames + 1))):
        raise ValueError("tracking frame ids are missing, duplicated, or out of order")
    team = _team_of_each_track(meta)
    if len(team) != players.shape[1] or not np.isin(team, (0, 1)).all():
        raise ValueError("tracking team metadata does not align with player slots")
    if not np.any(team == 0) or not np.any(team == 1):
        raise ValueError("tracking metadata must include both teams")
    if ball.shape[0] != n_frames or players.shape[0] != n_frames:
        raise ValueError("tracking arrays do not align with frame ids")

    pitch = tuple(GAMES[GAME]["pitch"])
    players_m = np.asarray(players, dtype=np.float32)
    players_m[:, :, 0] *= np.float32(pitch[0])
    players_m[:, :, 1] *= np.float32(pitch[1])
    ball_xy = ball[:, :2]
    ball_xy[:, 0] *= np.float32(pitch[0])
    ball_xy[:, 1] *= np.float32(pitch[1])

    blocks = half_video_blocks(GAME)
    decoded = []
    transition_owner = float(model["owner_transition_cost"])
    transition_dead = float(model["dead_transition_cost"])
    for half, start, stop in blocks:
        if start < 0 or stop >= n_frames or stop < start:
            raise ValueError("half {} falls outside the tracking timeline".format(half))
        half_players = players_m[start:stop + 1]
        half_ball = ball_xy[start:stop + 1]
        features = extract_features(half_players, half_ball, team, pitch,
                                    fps=float(model.get("fps", 25)))
        emissions = predict_scores(features, model)
        states = decode_scores(emissions, transition_owner, transition_dead)
        decoded.append(_to_letters(states))
        print("half {}: {} frames, ball observed {:.1%}".format(
            half, len(states), float(np.mean(features["ball_seen"]))))

    labels = _write_labels(n_frames, blocks, decoded)
    _validate_labels(labels, n_frames, blocks)
    _print_summary(labels, blocks)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame({"frame": np.arange(n_frames), "label": labels}).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

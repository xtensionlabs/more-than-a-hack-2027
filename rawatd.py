"""Fast reader for the raw Metrica ATD text file.

A line is ``<frame_id>:<x,y,speed;...>:<ball_x,ball_y,ball_z>`` with NaN for missing
observations. The reader streams directly into NumPy arrays to keep memory use bounded.
"""
import xml.etree.ElementTree as ET

import numpy as np


def n_slots_from_meta(meta_path):
    """Number of player track slots declared in the ATD metadata."""
    return sum(1 for _ in ET.parse(meta_path).getroot().iter("Player"))


def load_raw_atd(raw_path, n_slots, expected_frames=None):
    """frame_id (N,), players (N, n_slots, 3: x, y, speed) and ball (N, 3: x, y, z).

    Coordinates are still normalised 0-1. frame_id counts from 1; video frame = frame_id - 1.
    When ``expected_frames`` is supplied, allocate once and stream each record into its final
    slot. This avoids retaining hundreds of thousands of Python objects for a full match.
    """
    if expected_frames is not None:
        n_frames = int(expected_frames)
        if n_frames <= 0 or n_slots <= 0:
            raise ValueError("expected frame count and player slot count must be positive")
        ids = np.empty(n_frames, dtype=np.int64)
        players = np.empty((n_frames, n_slots, 3), dtype=np.float32)
        ball = np.empty((n_frames, 3), dtype=np.float32)
        count = 0
        with open(raw_path, encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= n_frames:
                    raise ValueError("ATD file has more than {:,} frames".format(n_frames))
                parts = line.rstrip("\r\n").split(":")
                if len(parts) != 3:
                    raise ValueError("malformed ATD record at row {}".format(index + 1))
                frame, group, ball_group = parts
                player_values = np.fromstring(group.replace(";", ","), sep=",")
                ball_values = np.fromstring(ball_group, sep=",")
                if player_values.size != n_slots * 3 or ball_values.size != 3:
                    raise ValueError("ATD row {} has an unexpected number of coordinates".format(
                        index + 1))
                ids[index] = int(frame)
                players[index] = player_values.reshape(n_slots, 3)
                ball[index] = ball_values
                count += 1
        if count != n_frames:
            raise ValueError("ATD file has {} frames; expected {}".format(count, n_frames))
        return ids, players, ball

    ids, players, ball = [], [], []
    with open(raw_path) as handle:
        for line in handle:
            frame, group, ball_group = line.rstrip("\n").split(":")
            ids.append(int(frame))
            players.append(np.array(group.replace(";", ",").split(","), dtype=float))
            ball.append(np.array(ball_group.split(","), dtype=float))
    return (np.array(ids), np.array(players).reshape(len(ids), n_slots, 3), np.array(ball))


def mean_player_speed(speed):
    """Mean of the tracker's own speed field (m/s) over the players tracked in each frame.

    NaN where no player has a speed. No warnings and no invented values.
    """
    finite = np.isfinite(speed)
    count = finite.sum(axis=1)
    total = np.where(finite, speed, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)

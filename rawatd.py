"""Fast reader for the raw Metrica ATD text file (checked against kloppy on real data).

A line is  <frame_id>:<x,y,speed;x,y,speed;...>:<ball_x,ball_y,ball_z>  with NaN for a missing
observation. kloppy needs minutes and gigabytes for a full match; this reads one in seconds.
On the delivered broadcast file it returns exactly the numbers kloppy returns (first 2,000
frames, every slot's x and the ball compared), and the same 50.5% ball visibility DATA.md quotes.
"""
import xml.etree.ElementTree as ET

import numpy as np


def n_slots_from_meta(meta_path):
    """Number of track slots the metadata declares (46 tactical, 51 broadcast per DATA.md)."""
    return sum(1 for _ in ET.parse(meta_path).getroot().iter("Player"))


def load_raw_atd(raw_path, n_slots):
    """frame_id (N,), players (N, n_slots, 3: x, y, speed) and ball (N, 3: x, y, z).

    Coordinates are still normalised 0-1. frame_id counts from 1; video frame = frame_id - 1.
    """
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

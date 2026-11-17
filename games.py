"""Challenge match constants and paths used by the production runner."""

import os


GAME = "BAR-ATH"
DATA_DIR = os.path.abspath(os.environ.get(
    "MTAH_DATA_DIR", os.path.join(os.path.dirname(__file__), "data")))

GAMES = {
    GAME: {
        "n_video_frames": 199319,
        "fps": 25,
        "pitch": (105.3, 68.0),
        "offsets": {1: 1702057, 2: 1702058},
        "halves": {1: (1713242, 1792708), 2: (1812866, 1892255)},
    }
}


def half_video_blocks(game=GAME, adjust=0):
    """Return ``(half, first_video_frame, last_video_frame)`` for graded windows."""
    if game not in GAMES:
        raise KeyError("no such game {!r}; available: {}".format(game, sorted(GAMES)))
    entry = GAMES[game]
    return [
        (half, first_id - entry["offsets"][half] - adjust,
         last_id - entry["offsets"][half] - adjust)
        for half, (first_id, last_id) in sorted(entry["halves"].items())
    ]


def require(names, game=GAME, data_dir=None):
    """Return paths for the tactical ATD files or report which files are missing."""
    if game not in GAMES:
        raise KeyError("no such game {!r}; available: {}".format(game, sorted(GAMES)))
    names = list(names)
    suffixes = {"atd_meta": "_atd.xml", "atd_raw": "_atd.txt"}
    unknown = [name for name in names if name not in suffixes]
    if unknown:
        raise KeyError("unsupported tactical assets: {}".format(", ".join(unknown)))
    root = os.path.abspath(data_dir or DATA_DIR)
    stem = game + "_tactical"
    paths = [os.path.join(root, "tactical", stem + suffixes[name]) for name in names]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "required tactical tracking files were not found:\n  {}\n"
            "Set MTAH_DATA_DIR to the organizers' bundle root.".format(
                "\n  ".join(missing)))
    return paths

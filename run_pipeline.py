
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from kloppy import metrica, sportscode

import pipeline
from games import asset_paths, half_video_blocks
from pipeline import *

N, MATCH = 199_319, "BAR-ATH"
PATHS = asset_paths()

# Fill AFTER checking tag_agreement(); empty = Smart Tagging unused.
# Example once verified: {"SOME CODE": +1, "OTHER CODE": -1}
SIGN_BY_CODE = {}

# ---------- 1. load tracking: one row per VIDEO frame ----------
ds = metrica.load_tracking_epts(meta_data=PATHS["atd_meta"], raw_data=PATHS["atd_raw"])
df = ds.to_df()
df = df.set_index(df["frame_id"] - 1).reindex(range(N))  # row i == video frame i
slots = sorted(
    {c[:-2] for c in df.columns if c.endswith("_x") and not c.startswith("ball")}
)
xy = np.stack([df[[f"{s}_x", f"{s}_y"]].to_numpy() * [L, W] for s in slots], axis=1)
spd = np.stack([df[f"{s}_s"].to_numpy() for s in slots], axis=1)
bx, by = df["ball_x"].to_numpy() * L, df["ball_y"].to_numpy() * W

# ---------- 2. which slots are home / away ----------
home_ids = {str(p.player_id) for p in ds.metadata.teams[0].players}
home = np.array([i for i, s in enumerate(slots) if s in home_ids])
away = np.array([i for i, s in enumerate(slots) if s not in home_ids])
print(len(slots), "slots;", len(home), "home /", len(away), "away")
print("rows:", len(df), "| ball missing on", f"{np.isnan(bx).mean():.1%}", "of rows")
assert len(df) == N, f"expected {N} rows, got {len(df)}"
assert (
    0 < len(home) < len(slots)
), "team mapping failed - compare slots[:5] with the player ids"

# ---------- 3. clean, tag, decode each half ----------
bx, by, observed = clean_ball(bx, by)
xs = smooth_players(xy)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
    sm = np.nanmean(spd, axis=1)

ev = tag_frames(sportscode.load(PATHS["smart_tagging"]).to_df())
print("Team values in the tag file:", sorted(ev["team_l"].unique()))
# Once you know the names, map them (0 = home, 1 = away), for example:
# pipeline.TEAM_TO_STATE.update({"fc barcelona": 0, "athletic club": 1})
import difflib

meta_names = {0: ds.metadata.teams[0].name, 1: ds.metadata.teams[1].name}
for t in ev["team_l"].unique():
    if t in ("n/a", "nan", "none"):
        continue  # tags with no team (e.g. throw-ins) are skipped
    best = max(
        meta_names,
        key=lambda k: difflib.SequenceMatcher(None, t, str(meta_names[k]).lower()).ratio(),
    )
    pipeline.TEAM_TO_STATE[t] = best
print("Tag team mapping:", pipeline.TEAM_TO_STATE, "| metadata names:", meta_names)
tb = tag_bonus_array(ev, N, SIGN_BY_CODE, w=0.3)

blocks = half_video_blocks(MATCH)
decoded = []
for half, a, b in blocks:
    sl = slice(a, b + 1)
    decoded.append(decode_frames(bx[sl], by[sl], sm[sl], xs[sl], home, away, tb[sl]))

# ---------- 4. write + check ----------
labels = write_labels(N, blocks, decoded)
validate(labels, N, blocks)
summary(labels, blocks)
Path("submission").mkdir(exist_ok=True)
# ASSUMPTION: one-column CSV. Check starter/03_first_submission.ipynb and
# SUBMISSION.md and copy the exact column names and layout they use.
Path("submission").mkdir(exist_ok=True)
out = pd.DataFrame({"frame": np.arange(N), "label": labels})
out.to_csv("submission/possession.csv", index=False)
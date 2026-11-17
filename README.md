# Xtenders — More Than a Hack 2027

Xtenders is a three-member team. This repository contains our Initial Challenge possession entry, its frozen model, external-only training and benchmark scripts, and the submission format validator.

The production runner uses automatic tactical tracking and team metadata. It does not read challenge video, Smart Tagging, or possession ground truth. The model was trained and tuned using licensed external match data only. The organizers' latest rules clarification takes precedence; see the technical report for the data boundary and validation limits.

## Generate and validate the submission

Use Python and install the pinned dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-solution.txt
```

Set `MTAH_DATA_DIR` to the root of the organizers' data bundle. The tactical ATD files must be under its `tactical` directory.

```powershell
$env:MTAH_DATA_DIR = 'D:\path\to\Challenge-Datasets'
python run_full.py --out submission/Xtenders.csv
python evaluation/validate_submission.py submission/Xtenders.csv
```

The runner validates tracking continuity and team alignment, predicts each graded half independently, and writes `X` outside the graded windows. The output is deterministic for identical inputs and model weights. The validator checks row count, columns, frame order, and label values; it does not score predictions. Challenge data and generated submissions are ignored by Git.

## Model provenance and reproduction

The frozen weights and provenance are in `starter/external_model.json`. The seven training matches and their framewise DFL `BallStatus`/`BallPossession` fields come from the IDSSE dataset, licensed CC BY 4.0: [dataset record](https://doi.org/10.6084/m9.figshare.28196177), described in [Bassek et al. (2025)](https://www.nature.com/articles/s41597-025-04505-y). The source paper does not identify individual annotators for those fields; the technical report records that limitation. For the DFL code mapping, see the [Floodlight parser](https://github.com/floodlight-sports/floodlight/blob/main/floodlight/io/dfl.py).

To refit and regenerate the external validation report, download the seven licensed match-information and position XML files, then run:

```powershell
python starter/train_external_model.py `
  --data-dir 'D:\path\to\idsse-matches' `
  --out "$env:TEMP\mtah2027-model-build" `
  --cache-dir "$env:TEMP\mtah2027-feature-cache"
```

The training script reads external matches only. To reproduce the model-versus-nearest-player comparison on the same external data:

```powershell
python starter/compare_external_baseline.py `
  --data-dir 'D:\path\to\idsse-matches' `
  --cache-dir "$env:TEMP\mtah2027-feature-cache" `
  --out "$env:TEMP\mtah2027-external-comparison.json"
```

See the team-named [technical report](Xtenders.pdf) for the method, results, compliance boundary, and limitations. `Xtenders.md` is the editable source for that PDF. External results show improvement over a simple nearest-player reference; they do not establish a ranking against other teams.

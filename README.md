# ITU Challenge – Phase 2

Crop type and phenology stage classification from Sentinel-2 imagery for the ITU AI/ML in 5G Challenge (Track 1).

Given a set of `(longitude, latitude, date)` points and the Sentinel-2 L2A tiles that cover them, the system predicts:

- **Crop type**: `background`, `corn`, `rice`, or `soybean`
- **Phenology stage**: `Dormancy`, `Greenup`, `MidGreenup`, `Peak`, `Maturity`, `MidSenescence`, `Senescence`

## What's in this repo

| File | Purpose |
|---|---|
| [extract_features.py](extract_features.py) | Reads Sentinel-2 TIFFs, samples bands at each point, computes temporal features (per-band stats and vegetation indices around the target date). Shared by train and inference. |
| [train.py](train.py) | Trains two LightGBM classifiers (crop + phenology) with `GroupKFold` CV on `point_id`. Writes model bundles to `models/`. |
| [inference.py](inference.py) | Container entrypoint. Reads `/input`, builds features, loads `models/*.joblib`, writes `/output/result.json`. |
| [Dockerfile](Dockerfile) | Python 3.11 + GDAL base image; installs deps and bakes the models in. |
| [run.sh](run.sh) | Container CMD — runs `inference.py`. |
| [requirements.txt](requirements.txt) | numpy, pandas, rasterio, scikit-learn, lightgbm, joblib. |
| [points_train_label.csv](points_train_label.csv) | Training labels (point, date, crop, phenophase). |
| `region_train_*/` | Sentinel-2 TIFFs used for training. |
| `models/` | Trained `crop_model.joblib` and `pheno_model.joblib` (loaded at inference). |

## Training

Requires the `region_train_*/` folders and `points_train_label.csv` in the repo root.

```bash
pip install -r requirements.txt
python train.py
```

Feature extraction is cached to `features_cache.pkl` on first run. Output: `models/crop_model.joblib` and `models/pheno_model.joblib`, each bundled with its class list and feature column order. Mean macro-F1 per fold is printed for each model.

## Inference (Docker, challenge format)

Build the image (models are copied in):

```bash
docker build -t itu-phase2 .
```

Run with the challenge input/output contract:

```bash
docker run --rm \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  itu-phase2
```

The container:
1. Finds the points CSV in `/input` (prefers a name containing `point` and not `label`).
2. Finds the subfolder containing Sentinel-2 `.tiff` / `.tif` files.
3. Builds features, loads the models, predicts crop + phenology.
4. Writes `/output/result.json` with keys `"{Longitude}_{Latitude}_{phenophase_date}"` mapping to `[crop, phenophase]`.

If the models are missing, inference falls back to predicting `["rice", "Maturity"]` for every row so the submission stays valid.

### Local inference without Docker

```bash
INPUT_DIR=./test_input_sample \
OUTPUT_DIR=./out \
MODELS_DIR=./models \
python inference.py
```

## Input format

- **CSV** with columns `Longitude`, `Latitude`, `phenophase_date` (plus any extras — ignored).
- **TIFFs** named like `region03_2018-07-13-00-00_2018-07-13-23-59_Sentinel-2_L2A_B04_(Raw).tiff`, one file per band per date. Bands used: B01–B09, B8A, B11, B12.

## Output format

`result.json`:

```json
{
  "116.1234_39.5678_2018-07-13": ["corn", "Peak"],
  ...
}
```

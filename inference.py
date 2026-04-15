"""Container entrypoint. Reads /input, writes /output/result.json."""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from extract_features import build_feature_table

INPUT_DIR = os.environ.get("INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
MODELS_DIR = os.environ.get("MODELS_DIR", "/workspace/models")

CROPS = ["background", "corn", "rice", "soybean"]
PHENOS = ["Dormancy", "Greenup", "MidGreenup", "Peak", "Maturity", "MidSenescence", "Senescence"]


def _find_points_csv(folder: str) -> str:
    csvs = [n for n in os.listdir(folder) if n.lower().endswith(".csv")]
    if not csvs:
        raise FileNotFoundError(f"no csv in {folder}")
    # prefer csvs that look like input (contain 'point', not 'label')
    def score(n: str) -> tuple:
        low = n.lower()
        return (("point" in low), ("label" not in low), -len(n))
    csvs.sort(key=score, reverse=True)
    return os.path.join(folder, csvs[0])


def _find_tiff_folder(folder: str) -> str:
    # prefer any subdir containing .tiff files
    for entry in os.listdir(folder):
        p = os.path.join(folder, entry)
        if os.path.isdir(p) and any(f.endswith((".tiff", ".tif")) for f in os.listdir(p)):
            return p
    # fall back to input folder itself
    return folder


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    points_csv = _find_points_csv(INPUT_DIR)
    tiff_dir = _find_tiff_folder(INPUT_DIR)
    print(f"points: {points_csv}", flush=True)
    print(f"tiffs:  {tiff_dir}", flush=True)

    df = pd.read_csv(points_csv)
    feats = build_feature_table(df, tiff_dir)

    crop_model_path = os.path.join(MODELS_DIR, "crop_model.joblib")
    pheno_model_path = os.path.join(MODELS_DIR, "pheno_model.joblib")
    have_models = os.path.exists(crop_model_path) and os.path.exists(pheno_model_path)

    if have_models:
        crop_bundle = joblib.load(crop_model_path)
        pheno_bundle = joblib.load(pheno_model_path)
        X = feats.reindex(columns=crop_bundle["features"]).fillna(-9999).values
        crop_pred = crop_bundle["model"].predict(X)
        crop_labels = [crop_bundle["classes"][int(i)] for i in crop_pred]

        Xp = feats.reindex(columns=pheno_bundle["features"]).fillna(-9999).values
        pheno_pred = pheno_bundle["model"].predict(Xp)
        pheno_labels = [pheno_bundle["classes"][int(i)] for i in pheno_pred]
    else:
        # fallback: predict rice/Maturity for every row so submission is valid
        print("WARN: models not found, using fallback predictions", flush=True)
        crop_labels = ["rice"] * len(df)
        pheno_labels = ["Maturity"] * len(df)

    result: dict[str, list[str]] = {}
    for i, row in df.iterrows():
        key = f"{row['Longitude']}_{row['Latitude']}_{row['phenophase_date']}"
        result[key] = [crop_labels[i], pheno_labels[i]]

    out_path = os.path.join(OUTPUT_DIR, "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"wrote {len(result)} predictions to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

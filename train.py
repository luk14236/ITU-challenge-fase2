"""Train crop + phenology classifiers from local training data.

Expected layout (relative to this file):
  points_train_label.csv
  region_train_4/*.tiff   (and optionally region_train_1..3)

Outputs joblib bundles to models/.
"""
from __future__ import annotations

import os
import sys
from glob import glob

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import lightgbm as lgb

from extract_features import build_feature_table

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
CACHE = os.path.join(ROOT, "features_cache.pkl")


def _collect_tiff_dirs() -> list[str]:
    dirs = []
    for name in os.listdir(ROOT):
        p = os.path.join(ROOT, name)
        if os.path.isdir(p) and name.startswith("region_train") and glob(os.path.join(p, "*.tiff")):
            dirs.append(p)
    return dirs


def build_features(labels: pd.DataFrame) -> pd.DataFrame:
    """Build features by merging tiffs from every available region_train_* folder into one index."""
    # to keep it simple and reuse the single-folder API, build per folder and overlay.
    # a point only matches one region, so combining frames with .combine_first works.
    tiff_dirs = _collect_tiff_dirs()
    if not tiff_dirs:
        raise RuntimeError("no region_train_* folders found")
    print(f"using tiff folders: {tiff_dirs}", flush=True)
    # Monkey-patch: merge all dirs into a synthetic directory view by symlink? Simpler: iterate folders and use first that yields features.
    frames = []
    for d in tiff_dirs:
        print(f"[features] folder {d}", flush=True)
        f = build_feature_table(labels, d)
        frames.append(f)
    # take the first non-null row across folders
    combined = frames[0]
    for nxt in frames[1:]:
        combined = combined.combine_first(nxt)
    return combined


def train_classifier(X: np.ndarray, y: np.ndarray, groups: np.ndarray, classes: list[str]) -> tuple[lgb.LGBMClassifier, float]:
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    f1s = []
    best_model = None
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        m = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multiclass",
            num_class=len(classes),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(40, verbose=False)])
        pred = m.predict(X[va])
        f = f1_score(y[va], pred, average="macro")
        f1s.append(f)
        print(f"  fold {fold}: macro F1 = {f:.4f}", flush=True)
        best_model = m
    return best_model, float(np.mean(f1s))


def main() -> int:
    os.makedirs(MODELS_DIR, exist_ok=True)
    labels = pd.read_csv(os.path.join(ROOT, "points_train_label.csv"))
    print(f"loaded {len(labels)} labeled rows ({labels['point_id'].nunique()} points)", flush=True)

    if os.path.exists(CACHE):
        print(f"using cached features {CACHE}", flush=True)
        feats = pd.read_pickle(CACHE)
    else:
        feats = build_features(labels)
        feats.to_pickle(CACHE)
        print(f"cached features -> {CACHE}", flush=True)

    # drop rows where all features are NaN (point not in any loaded region)
    valid = feats.notna().any(axis=1)
    print(f"valid rows: {valid.sum()}/{len(feats)}", flush=True)
    feats = feats[valid].reset_index(drop=True)
    lab = labels[valid.values].reset_index(drop=True)

    feat_cols = [c for c in feats.columns]
    X = feats[feat_cols].fillna(-9999).values
    groups = lab["point_id"].values

    # crop model
    crop_classes = sorted(lab["crop_type"].dropna().unique().tolist())
    y_crop = np.array([crop_classes.index(c) for c in lab["crop_type"]])
    print("\n=== training crop classifier ===", flush=True)
    crop_model, crop_f1 = train_classifier(X, y_crop, groups, crop_classes)
    joblib.dump({"model": crop_model, "classes": crop_classes, "features": feat_cols},
                os.path.join(MODELS_DIR, "crop_model.joblib"))
    print(f"crop F1 (mean): {crop_f1:.4f}", flush=True)

    # phenology model (trained on all rows, all crops)
    pheno_classes = sorted(lab["phenophase_name"].dropna().unique().tolist())
    y_ph = np.array([pheno_classes.index(p) for p in lab["phenophase_name"]])
    print("\n=== training phenology classifier ===", flush=True)
    ph_model, ph_f1 = train_classifier(X, y_ph, groups, pheno_classes)
    joblib.dump({"model": ph_model, "classes": pheno_classes, "features": feat_cols},
                os.path.join(MODELS_DIR, "pheno_model.joblib"))
    print(f"pheno F1 (mean): {ph_f1:.4f}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Feature extraction from Sentinel-2 TIFFs for a set of (lon, lat, date) points.

Shared by training (local) and inference (container).
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

# filename pattern:  region03_2018-07-13-00-00_2018-07-13-23-59_Sentinel-2_L2A_B04_(Raw).tiff
# also:              region00_2018-10-11-00_00_2018-10-11-23_59_Sentinel-2_L2A_B04_(Raw).tiff
_RE = re.compile(
    r"(region\d+)_(\d{4}-\d{2}-\d{2})[-_]\d{2}[-_]\d{2}_.*?_(B[0-9A]+)_"
)


def index_tiffs(folder: str) -> dict:
    """Returns {region: {date: {band: path}}}."""
    idx: dict = defaultdict(lambda: defaultdict(dict))
    for path in glob(os.path.join(folder, "*.tiff")) + glob(os.path.join(folder, "*.tif")):
        name = os.path.basename(path)
        m = _RE.search(name)
        if not m:
            continue
        region, date, band = m.group(1), m.group(2), m.group(3)
        idx[region][date][band] = path
    return idx


def _parse_date(s: str) -> datetime:
    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"bad date: {s}")


def pick_closest_dates(available: list[str], target: str, k: int = 3) -> list[str]:
    """Pick k dates closest to target from available (sorted by abs diff)."""
    t = _parse_date(target)
    return sorted(available, key=lambda d: abs((_parse_date(d) - t).days))[:k]


def read_pixel(path: str, lon: float, lat: float) -> float | None:
    """Sample single pixel at (lon, lat). Returns None on errors or out-of-bounds."""
    try:
        with rasterio.open(path) as src:
            try:
                r, c = rowcol(src.transform, lon, lat)
            except Exception:
                return None
            if r < 0 or c < 0 or r >= src.height or c >= src.width:
                return None
            val = src.read(1, window=((r, r + 1), (c, c + 1)))
            if val.size == 0:
                return None
            v = float(val[0, 0])
            nodata = src.nodata
            if nodata is not None and v == nodata:
                return None
            return v
    except Exception:
        return None


_BOUNDS_CACHE: dict = {}


def _region_bounds(idx: dict) -> dict:
    key = id(idx)
    if key in _BOUNDS_CACHE:
        return _BOUNDS_CACHE[key]
    out = {}
    for region, dates in idx.items():
        if not dates:
            continue
        first_date = next(iter(dates))
        path = next(iter(dates[first_date].values()))
        with rasterio.open(path) as src:
            out[region] = src.bounds
    _BOUNDS_CACHE[key] = out
    return out


def point_region(idx: dict, lon: float, lat: float) -> str | None:
    for region, b in _region_bounds(idx).items():
        if b.left <= lon <= b.right and b.bottom <= lat <= b.top:
            return region
    return None


def _ndvi(nir: float | None, red: float | None) -> float | None:
    if nir is None or red is None:
        return None
    s = nir + red
    if s == 0:
        return None
    return (nir - red) / s


def extract_row(idx: dict, lon: float, lat: float, target_date: str, n_dates: int = 3) -> dict:
    """Extract features for one (lon, lat, date) observation.

    Returns dict of features (NaN for missing). One column per band per temporal slot
    (t0 = closest date, t1 = second-closest, ...), plus NDVI per slot and summary stats.
    """
    feats: dict = {}
    region = point_region(idx, lon, lat)
    if region is None:
        return feats

    dates_avail = sorted(idx[region].keys())
    if not dates_avail:
        return feats

    chosen = pick_closest_dates(dates_avail, target_date, k=n_dates)
    # keep in chronological order so slot semantics are stable
    chosen = sorted(chosen, key=lambda d: _parse_date(d))

    ndvi_series: list[float] = []
    for i, d in enumerate(chosen):
        band_map = idx[region][d]
        band_vals: dict[str, float | None] = {}
        for b in BANDS:
            p = band_map.get(b)
            band_vals[b] = read_pixel(p, lon, lat) if p else None
            feats[f"t{i}_{b}"] = band_vals[b] if band_vals[b] is not None else np.nan
        ndvi = _ndvi(band_vals.get("B08"), band_vals.get("B04"))
        feats[f"t{i}_NDVI"] = ndvi if ndvi is not None else np.nan
        if ndvi is not None:
            ndvi_series.append(ndvi)
        # days offset from target
        feats[f"t{i}_dayoff"] = (_parse_date(d) - _parse_date(target_date)).days

    if ndvi_series:
        feats["ndvi_mean"] = float(np.mean(ndvi_series))
        feats["ndvi_max"] = float(np.max(ndvi_series))
        feats["ndvi_min"] = float(np.min(ndvi_series))
        feats["ndvi_range"] = feats["ndvi_max"] - feats["ndvi_min"]

    # month of target date as cyclic feature
    m = _parse_date(target_date).month
    feats["month_sin"] = float(np.sin(2 * np.pi * m / 12))
    feats["month_cos"] = float(np.cos(2 * np.pi * m / 12))
    return feats


def build_feature_table(points_df: pd.DataFrame, tiff_folder: str, n_dates: int = 3) -> pd.DataFrame:
    """points_df must have columns: Longitude, Latitude, phenophase_date."""
    idx = index_tiffs(tiff_folder)
    rows = []
    total = len(points_df)
    for i, r in enumerate(points_df.itertuples(index=False)):
        f = extract_row(idx, float(r.Longitude), float(r.Latitude), str(r.phenophase_date), n_dates=n_dates)
        rows.append(f)
        if (i + 1) % 200 == 0:
            print(f"  [features] {i+1}/{total}", flush=True)
    feat = pd.DataFrame(rows)
    return feat

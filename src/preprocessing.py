"""
Multi-ROI preprocessing used for the frozen R-DeepDEM Protocol-A
benchmark.

Scientific lineage
------------------
Phase-5 multi-ROI preparation:
  * four ROI definitions;
  * SRTM / USGS 3DEP / NAIP export on the ROI UTM grid;
  * ROI-wise SRTM vertical bias correction;
  * terrain and optical feature construction;
  * 256 x 256 patches with stride 128;
  * minimum valid ratio 0.95;
  * x-blocked Protocol-A split with buffer columns.

Notebook-specific Drive orchestration and display code are intentionally
excluded from this public module.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import rasterio

from scipy.ndimage import sobel, laplace

try:
    import ee
except ImportError:
    ee = None


PATCH_SIZE = 256
STRIDE = 128
MIN_VALID_RATIO = 0.95


def _require_ee():
    if ee is None:
        raise ImportError(
            "earthengine-api is required for the "
            "Earth Engine availability/export helpers."
        )

DEFAULT_ROIS = [
    {
        "roi_id": "houston_buffalo_bayou",
        "city": "Houston",
        "state": "TX",
        "bbox": [-95.4200, 29.7485, -95.4070, 29.7585],
        "utm_epsg": 32615,
        "role": "existing_phase1_phase2_phase3_roi"
    },

    {
        "roi_id": "san_antonio_river",
        "city": "San Antonio",
        "state": "TX",
        "bbox": [-98.5050, 29.4140, -98.4920, 29.4240],
        "utm_epsg": 32614,
        "role": "new_phase5_roi"
    },

    {
        "roi_id": "austin_waller_creek",
        "city": "Austin",
        "state": "TX",
        "bbox": [-97.7400, 30.2600, -97.7270, 30.2700],
        "utm_epsg": 32614,
        "role": "new_phase5_roi"
    },

    {
        "roi_id": "dallas_trinity_river",
        "city": "Dallas",
        "state": "TX",
        "bbox": [-96.8250, 32.7650, -96.8120, 32.7750],
        "utm_epsg": 32614,
        "role": "new_phase5_roi"
    }
]

def roi_geom_from_bbox(bbox):
    _require_ee()
    return ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)


def check_roi_availability(roi):
    _require_ee()
    roi_id = roi["roi_id"]
    geom = roi_geom_from_bbox(roi["bbox"])

    print(f"Checking ROI: {roi_id}")

    naip_collection = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterBounds(geom)
        .filterDate("2018-01-01", "2024-12-31")
    )

    dep3_collection = (
        ee.ImageCollection("USGS/3DEP/1m")
        .filterBounds(geom)
    )

    naip_count = naip_collection.size().getInfo()
    dep3_count = dep3_collection.size().getInfo()

    latest_naip_date = None
    if naip_count > 0:
        latest_naip = naip_collection.sort("system:time_start", False).first()
        latest_naip_date = ee.Date(latest_naip.get("system:time_start")).format("YYYY-MM-dd").getInfo()

    usable = naip_count > 0 and dep3_count > 0

    return {
        "roi_id": roi_id,
        "city": roi["city"],
        "state": roi["state"],
        "naip_count": naip_count,
        "latest_naip_date": latest_naip_date,
        "dep3_count": dep3_count,
        "usable": usable,
        "bbox": roi["bbox"],
        "utm_epsg": roi["utm_epsg"]
    }

def bbox_to_ee_geom(bbox_string_or_list):
    _require_ee()
    if isinstance(bbox_string_or_list, str):
        bbox = json.loads(bbox_string_or_list.replace("'", '"'))
    else:
        bbox = bbox_string_or_list
    return ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)


def export_roi_stack_mosaic_to_drive(row):
    _require_ee()
    roi_id = row["roi_id"]
    geom = bbox_to_ee_geom(row["bbox"])
    epsg = int(row["utm_epsg"])
    crs = f"EPSG:{epsg}"

    latest_date = str(row["latest_naip_date"])
    naip_year = latest_date[:4]

    print(f"\nPreparing mosaic export for: {roi_id}")
    print("CRS:", crs)
    print("NAIP year:", naip_year)

    # SRTM
    srtm = (
        ee.Image("USGS/SRTMGL1_003")
        .select("elevation")
        .rename("srtm")
        .resample("bilinear")
    )

    # 3DEP
    dep3 = (
        ee.ImageCollection("USGS/3DEP/1m")
        .filterBounds(geom)
        .mosaic()
        .select("elevation")
        .rename("dep3")
    )

    # NAIP yearly mosaic, not single tile
    naip_collection = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterBounds(geom)
        .filterDate(f"{naip_year}-01-01", f"{int(naip_year)+1}-01-01")
        .select(["R", "G", "B", "N"])
    )

    naip = (
        naip_collection
        .mosaic()
        .rename(["naip_R", "naip_G", "naip_B", "naip_NIR"])
    )

    stack = (
        srtm
        .addBands(dep3)
        .addBands(naip)
        .toFloat()
        .clip(geom)
    )

    task = ee.batch.Export.image.toDrive(
        image=stack,
        description=f"phase5_{roi_id}_stack_mosaic",
        folder="GDEMSR_phase5_gee_exports",
        fileNamePrefix=f"phase5_{roi_id}_stack_mosaic",
        region=geom,
        scale=1,
        crs=crs,
        maxPixels=1e10
    )

    task.start()

    print("Started mosaic export task:", task.id)

    return {
        "roi_id": roi_id,
        "task_id": task.id,
        "description": f"phase5_{roi_id}_stack_mosaic",
        "drive_folder": "GDEMSR_phase5_gee_exports",
        "expected_filename_prefix": f"phase5_{roi_id}_stack_mosaic",
        "naip_year": naip_year
    }

def safe_index(num, den, eps=1e-6):
    return num / (den + eps)


def create_b4b_channels(stack_tif):
    with rasterio.open(stack_tif) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile

    # Band order:
    # 1 srtm, 2 dep3, 3 R, 4 G, 5 B, 6 NIR
    srtm = arr[0]
    dep3 = arr[1]
    R = arr[2]
    G = arr[3]
    B = arr[4]
    NIR = arr[5]

    valid = (
        np.isfinite(srtm) &
        np.isfinite(dep3) &
        np.isfinite(R) &
        np.isfinite(G) &
        np.isfinite(B) &
        np.isfinite(NIR)
    )

    # ROI-wise SRTM bias correction
    bias = np.nanmean(dep3[valid] - srtm[valid])
    srtm_bc = srtm + bias

    dzdx = sobel(srtm_bc, axis=1) / 8.0
    dzdy = sobel(srtm_bc, axis=0) / 8.0
    lap = laplace(srtm_bc)

    ndwi_like = safe_index(G - NIR, G + NIR)
    ndvi_like = safe_index(NIR - R, NIR + R)

    x_10ch = np.stack([
        srtm_bc,
        dzdx,
        dzdy,
        lap,
        R,
        G,
        B,
        NIR,
        ndwi_like,
        ndvi_like
    ], axis=0).astype(np.float32)

    residual = (dep3 - srtm_bc).astype(np.float32)

    return x_10ch, residual, dep3.astype(np.float32), srtm_bc.astype(np.float32), valid, bias, profile


def get_roi_id_from_filename(tif_path):
    return tif_path.stem.replace("phase5_", "").replace("_stack", "")


def generate_patches_for_roi(tif_path, out_root):
    roi_id = get_roi_id_from_filename(tif_path)
    roi_patch_dir = out_root / roi_id
    roi_patch_dir.mkdir(parents=True, exist_ok=True)

    x_10ch, residual, dep3, srtm_bc, valid, bias, profile = create_b4b_channels(tif_path)

    H, W = residual.shape
    records = []
    patch_id = 0

    print("\n" + "="*80)
    print("Generating patches for:", roi_id)
    print("Input shape:", H, "x", W)
    print("Overall valid ratio:", round(float(valid.mean()), 4))
    print("Bias correction dep3 - srtm:", round(float(bias), 4), "m")

    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x0 in range(0, W - PATCH_SIZE + 1, STRIDE):
            valid_patch = valid[y:y+PATCH_SIZE, x0:x0+PATCH_SIZE]
            valid_ratio = float(valid_patch.mean())

            if valid_ratio < MIN_VALID_RATIO:
                continue

            x_patch = x_10ch[:, y:y+PATCH_SIZE, x0:x0+PATCH_SIZE]
            residual_patch = residual[y:y+PATCH_SIZE, x0:x0+PATCH_SIZE]
            dep3_patch = dep3[y:y+PATCH_SIZE, x0:x0+PATCH_SIZE]
            srtm_patch = srtm_bc[y:y+PATCH_SIZE, x0:x0+PATCH_SIZE]

            patch_name = f"{roi_id}_p{patch_id:04d}_y{y}_x{x0}.npz"
            patch_path = roi_patch_dir / patch_name

            np.savez_compressed(
                patch_path,
                x=x_patch,
                residual=residual_patch,
                dep3=dep3_patch,
                srtm_bc=srtm_patch,
                valid=valid_patch.astype(np.uint8),
                roi_id=roi_id,
                y=y,
                x0=x0,
                bias=bias
            )

            records.append({
                "roi_id": roi_id,
                "patch_id": patch_id,
                "patch_name": patch_name,
                "patch_file": str(patch_path),
                "y": y,
                "x": x0,
                "valid_ratio": valid_ratio,
                "bias": float(bias),
                "residual_mean": float(np.nanmean(residual_patch)),
                "residual_std": float(np.nanstd(residual_patch)),
                "srtm_bc_mean": float(np.nanmean(srtm_patch)),
                "dep3_mean": float(np.nanmean(dep3_patch))
            })

            patch_id += 1

    print("Accepted patches:", len(records))
    return records

def assign_spatial_x_split(group):
    """
    Spatial x-stripe split.
    Uses x-coordinate columns to reduce leakage from overlapping patches.

    Split logic:
    - left/middle part for train
    - separated x column as buffer
    - next x column as val
    - separated x column as buffer
    - rightmost x columns as test
    """
    g = group.copy()
    xs = sorted(g["x"].unique())
    n = len(xs)

    if n < 5:
        # fallback, rarely needed
        train_x = xs[:max(1, n-2)]
        val_x = xs[max(1, n-2):max(2, n-1)]
        test_x = xs[max(2, n-1):]
        buffer_x = []
    else:
        # about 60% train, then buffer, val, buffer, test
        train_end = max(3, int(round(0.60 * n)))
        train_end = min(train_end, n - 4)

        train_x = xs[:train_end]
        buffer1_x = [xs[train_end]]
        val_x = [xs[train_end + 1]]
        buffer2_x = [xs[train_end + 2]]
        test_x = xs[train_end + 3:]
        buffer_x = buffer1_x + buffer2_x

    g["split"] = "unused_buffer"
    g.loc[g["x"].isin(train_x), "split"] = "train"
    g.loc[g["x"].isin(val_x), "split"] = "val"
    g.loc[g["x"].isin(test_x), "split"] = "test"

    g["split_design"] = "spatial_x_blocked_with_buffer"

    return g


def build_roi_manifest():
    """Return the four frozen study-area definitions."""
    return pd.DataFrame(
        DEFAULT_ROIS
    ).copy()


def check_all_roi_availability(rois=None):
    """
    Run the Phase-5 availability check for all supplied ROIs.
    """
    _require_ee()

    records = []

    for roi in (
        DEFAULT_ROIS
        if rois is None
        else rois
    ):
        try:
            records.append(
                check_roi_availability(
                    roi
                )
            )
        except Exception as exc:
            records.append({
                "roi_id": roi["roi_id"],
                "city": roi["city"],
                "state": roi["state"],
                "naip_count": None,
                "latest_naip_date": None,
                "dep3_count": None,
                "usable": False,
                "bbox": roi["bbox"],
                "utm_epsg": roi["utm_epsg"],
                "error": str(exc),
            })

    return pd.DataFrame(
        records
    )


def make_protocol_a_split(
    inventory_df,
):
    """
    Apply the frozen per-ROI spatial x-blocked split.
    """
    return (
        inventory_df
        .groupby(
            "roi_id",
            group_keys=False,
        )
        .apply(
            assign_spatial_x_split
        )
        .reset_index(
            drop=True
        )
    )


def compute_normalization_stats(
    split_df,
    split_name="train",
):
    """
    Compute the train-only 10-channel and residual normalization.

    This is a function-level refactor of the executed Phase-5
    B4bPatchDataset.compute_norm_stats implementation.

    ``split_df`` must contain a resolved ``patch_file`` column.
    """
    frame = (
        split_df[
            split_df["split"]
            == split_name
        ]
        .reset_index(drop=True)
    )

    if frame.empty:
        raise ValueError(
            f"No samples found for split={split_name!r}"
        )

    if "patch_file" not in frame.columns:
        raise KeyError(
            "split_df must contain a resolved "
            "'patch_file' column."
        )

    x_sum = np.zeros(
        10,
        dtype=np.float64,
    )

    x_sumsq = np.zeros(
        10,
        dtype=np.float64,
    )

    x_count = np.zeros(
        10,
        dtype=np.float64,
    )

    y_sum = 0.0
    y_sumsq = 0.0
    y_count = 0.0

    for path in frame[
        "patch_file"
    ].tolist():

        with np.load(path) as data:

            x = data["x"].astype(
                np.float32
            )

            y = data[
                "residual"
            ].astype(
                np.float32
            )

        for channel in range(10):

            values = x[channel]

            finite = np.isfinite(
                values
            )

            x_sum[channel] += (
                np.nansum(
                    values[finite]
                )
            )

            x_sumsq[channel] += (
                np.nansum(
                    values[finite] ** 2
                )
            )

            x_count[channel] += (
                finite.sum()
            )

        finite_y = np.isfinite(y)

        y_sum += np.nansum(
            y[finite_y]
        )

        y_sumsq += np.nansum(
            y[finite_y] ** 2
        )

        y_count += finite_y.sum()

    x_mean = (
        x_sum
        / np.maximum(
            x_count,
            1,
        )
    )

    x_var = (
        x_sumsq
        / np.maximum(
            x_count,
            1,
        )
    ) - (
        x_mean ** 2
    )

    x_std = np.sqrt(
        np.maximum(
            x_var,
            1e-8,
        )
    )

    y_mean = (
        y_sum
        / max(
            y_count,
            1,
        )
    )

    y_var = (
        y_sumsq
        / max(
            y_count,
            1,
        )
    ) - (
        y_mean ** 2
    )

    y_std = math.sqrt(
        max(
            y_var,
            1e-8,
        )
    )

    return {
        "x_mean":
            x_mean.tolist(),

        "x_std":
            x_std.tolist(),

        "y_mean":
            float(y_mean),

        "y_std":
            float(y_std),
    }

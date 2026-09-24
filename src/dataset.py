"""
Dataset construction for the frozen Protocol-A benchmark.

The E2c dataset class is retained from the Phase-7 checkpoint
reproduction source. The only public-release adaptation is
``attach_patch_files``: the private absolute ``patch_file`` column was
removed from the Zenodo split and replaced with ``patch_relpath``.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


def attach_patch_files(
    split_df,
    patch_root,
):
    """
    Resolve public relative patch identifiers to a local NPZ root.

    This changes filesystem addressing only; patch identity, ROI,
    coordinates and split assignments are unchanged.
    """
    frame = split_df.copy()

    root = Path(
        patch_root
    )

    if "patch_relpath" in frame.columns:

        frame["patch_file"] = (
            frame[
                "patch_relpath"
            ]
            .map(
                lambda value:
                str(
                    root
                    / str(value)
                )
            )
        )

    elif "patch_file" not in frame.columns:

        raise KeyError(
            "Expected 'patch_relpath' "
            "or 'patch_file' column."
        )

    return frame


def load_normalization_stats(
    path,
):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )

class ProtocolADualScaleDataset(Dataset):
    def __init__(self, dataframe, split_name, norm_stats):
        self.df = (
            dataframe[dataframe["split"] == split_name]
            .reset_index(drop=True)
            .copy()
        )
        self.split_name = split_name
        self.x_mean = np.asarray(norm_stats["x_mean"], dtype=np.float32)[:, None, None]
        self.x_std = np.asarray(norm_stats["x_std"], dtype=np.float32)[:, None, None]
        self.y_mean = np.float32(norm_stats["y_mean"])
        self.y_std = np.float32(norm_stats["y_std"])

        if self.x_mean.shape[0] != 10 or self.x_std.shape[0] != 10:
            raise AssertionError(
                f"Expected 10 original globally normalized channels, "
                f"found means={self.x_mean.shape[0]}, stds={self.x_std.shape[0]}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        patch_path = Path(row["patch_file"])

        if not patch_path.exists():
            raise FileNotFoundError(patch_path)

        with np.load(patch_path, allow_pickle=True) as data:
            required_keys = {"x", "residual", "dep3", "srtm_bc", "valid", "roi_id"}
            missing_keys = required_keys.difference(data.files)
            if missing_keys:
                raise KeyError(
                    f"{patch_path} is missing NPZ keys: {sorted(missing_keys)}"
                )

            x_raw = data["x"].astype(np.float32)
            residual = data["residual"].astype(np.float32)
            dep3 = data["dep3"].astype(np.float32)
            srtm_bc = data["srtm_bc"].astype(np.float32)
            stored_valid = data["valid"].astype(bool)
            roi_id = str(data["roi_id"])

        if x_raw.shape != (10, 256, 256):
            raise AssertionError(
                f"Unexpected x shape for {patch_path}: {x_raw.shape}; "
                "expected (10, 256, 256)"
            )
        if dep3.shape != (256, 256) or srtm_bc.shape != (256, 256):
            raise AssertionError(f"Unexpected DEM shape in {patch_path}")
        if roi_id != str(row["roi_id"]):
            raise AssertionError(
                f"ROI mismatch for {patch_path}: NPZ={roi_id}, CSV={row['roi_id']}"
            )

        # Exact recovered validity rule from Phase-6.
        finite_x_all = np.isfinite(x_raw).all(axis=0)
        valid = stored_valid & finite_x_all

        # Globally normalized original 10 channels.
        x_global = (x_raw - self.x_mean) / np.maximum(self.x_std, 1e-6)

        # Patch-local normalization only for SRTM channel x_raw[0].
        local_values = x_raw[0][valid]
        if local_values.size == 0:
            local_mean = float(self.x_mean[0, 0, 0])
            local_std = float(self.x_std[0, 0, 0])
        else:
            local_mean = float(np.mean(local_values))
            local_std = float(np.std(local_values))

        if not np.isfinite(local_std) or local_std < 1e-6:
            local_std = 1.0

        local_srtm = (x_raw[0] - local_mean) / local_std

        # Exact 11-channel E2c ordering:
        # global SRTM, local SRTM, then globally normalized channels 1..9.
        x_dual = np.concatenate(
            [x_global[0:1], local_srtm[None], x_global[1:10]],
            axis=0,
        )

        y_norm = (residual - self.y_mean) / max(float(self.y_std), 1e-6)

        x_dual = np.nan_to_num(
            x_dual, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        y_norm = np.nan_to_num(
            y_norm, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        dep3 = np.nan_to_num(
            dep3, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        srtm_bc = np.nan_to_num(
            srtm_bc, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

        return {
            "x": torch.from_numpy(x_dual),
            "y_norm": torch.from_numpy(y_norm[None]),
            "dep3": torch.from_numpy(dep3[None]),
            "srtm_bc": torch.from_numpy(srtm_bc[None]),
            "valid": torch.from_numpy(valid.astype(np.float32)[None]),
            "roi_id": roi_id,
            "patch_file": str(patch_path),
            "row_index": int(index),
            "local_srtm_mean": np.float32(local_mean),
            "local_srtm_std": np.float32(local_std),
        }

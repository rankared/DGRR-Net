#!/usr/bin/env python3
"""
Evaluate the released E2c checkpoint on a locally reconstructed
Protocol-A split.

Reports pooled valid-pixel elevation MAE, RMSE and bias.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import DataLoader

from src.dataset import (
    ProtocolADualScaleDataset,
    attach_patch_files,
    load_normalization_stats,
)

from src.inference import (
    load_checkpoint,
)

from src.model import (
    E2cDualScaleResidualUNet,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--support-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--patch-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=[
            "train",
            "val",
            "test",
        ],
    )

    parser.add_argument(
        "--batch-size",
        default=2,
        type=int,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    split = pd.read_csv(
        args.support_dir
        / "data"
        / "protocol_a_split.csv"
    )

    split = attach_patch_files(
        split,
        args.patch_root,
    )

    norm_stats = (
        load_normalization_stats(
            args.support_dir
            / "data"
            / "normalization_stats.json"
        )
    )

    dataset = (
        ProtocolADualScaleDataset(
            split,
            args.split,
            norm_stats,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        args.device
    )

    model = (
        E2cDualScaleResidualUNet(
            input_channels=11,
            base_channels=32,
            dropout=0.05,
        )
        .to(device)
    )

    checkpoint = load_checkpoint(
        model,
        args.support_dir
        / "checkpoints"
        / "rdeepdem_e2c_best.pt",
        device=device,
    )

    model.eval()

    y_mean = float(
        norm_stats["y_mean"]
    )

    y_std = float(
        norm_stats["y_std"]
    )

    abs_sum = 0.0
    sq_sum = 0.0
    error_sum = 0.0
    valid_pixels = 0

    with torch.no_grad():

        for batch in loader:

            x = batch["x"].to(
                device
            )

            dep3 = batch[
                "dep3"
            ].to(
                device
            )

            srtm_bc = batch[
                "srtm_bc"
            ].to(
                device
            )

            valid = (
                batch["valid"]
                .to(device)
                > 0.5
            )

            prediction_norm = model(x)

            prediction_residual = (
                prediction_norm
                * y_std
                + y_mean
            )

            prediction_elevation = (
                srtm_bc
                + prediction_residual
            )

            error = (
                prediction_elevation
                - dep3
            )

            values = error[
                valid
            ].double()

            abs_sum += float(
                values.abs().sum()
            )

            sq_sum += float(
                (values ** 2).sum()
            )

            error_sum += float(
                values.sum()
            )

            valid_pixels += int(
                values.numel()
            )

    if valid_pixels == 0:
        raise RuntimeError(
            "No valid evaluation pixels."
        )

    metrics = {
        "model":
            "E2c dual-scale residual U-Net",

        "split":
            args.split,

        "patches":
            len(dataset),

        "valid_pixels":
            valid_pixels,

        "mae_m":
            abs_sum
            / valid_pixels,

        "rmse_m":
            (
                sq_sum
                / valid_pixels
            ) ** 0.5,

        "bias_m":
            error_sum
            / valid_pixels,

        "checkpoint_epoch":
            int(
                checkpoint["epoch"]
            ),
    }

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    if args.output_json is not None:

        args.output_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output_json.write_text(
            json.dumps(
                metrics,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

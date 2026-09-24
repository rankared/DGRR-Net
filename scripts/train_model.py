#!/usr/bin/env python3
"""
Train the proposed E2c dual-scale residual U-Net using the frozen
Protocol-A dataset contract.

Checkpoint selection:
    lowest pooled validation elevation MAE

This script expects the patch NPZ files themselves to be available
locally. They are not redistributed in the supporting-data deposit.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from src.dataset import (
    ProtocolADualScaleDataset,
    attach_patch_files,
    load_normalization_stats,
)

from src.model import (
    E2cDualScaleResidualUNet,
)

from src.train import (
    build_optimizer_and_scheduler,
    train_one_epoch,
)


def evaluate_elevation(
    model,
    loader,
    norm_stats,
    device,
):
    model.eval()

    abs_sum = 0.0
    sq_sum = 0.0
    err_sum = 0.0
    count = 0.0

    y_mean = float(
        norm_stats["y_mean"]
    )

    y_std = float(
        norm_stats["y_std"]
    )

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

            valid = batch[
                "valid"
            ].to(
                device
            ) > 0.5

            prediction_norm = model(
                x
            )

            residual_m = (
                prediction_norm
                * y_std
                + y_mean
            )

            elevation_m = (
                srtm_bc
                + residual_m
            )

            error = (
                elevation_m
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

            err_sum += float(
                values.sum()
            )

            count += float(
                values.numel()
            )

    if count == 0:
        raise RuntimeError(
            "Validation set has no valid pixels."
        )

    return {
        "mae":
            abs_sum / count,

        "rmse":
            (sq_sum / count) ** 0.5,

        "bias":
            err_sum / count,

        "valid_pixels":
            int(count),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--support-dir",
        required=True,
        type=Path,
        help=(
            "R-DeepDEM_supporting_data directory."
        ),
    )

    parser.add_argument(
        "--patch-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "e2c_proposed.json"
        ),
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = parser.parse_args()

    repo_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    config_path = (
        args.config
        if args.config.is_absolute()
        else repo_root
        / args.config
    )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as handle:
        config = json.load(
            handle
        )

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

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    device = torch.device(
        args.device
    )

    train_dataset = (
        ProtocolADualScaleDataset(
            split,
            "train",
            norm_stats,
        )
    )

    val_dataset = (
        ProtocolADualScaleDataset(
            split,
            "val",
            norm_stats,
        )
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    batch_size = int(
        config["batch_size"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    model = (
        E2cDualScaleResidualUNet(
            input_channels=int(
                config["input_channels"]
            ),
            base_channels=int(
                config["base_channels"]
            ),
            dropout=float(
                config["dropout"]
            ),
        )
        .to(device)
    )

    optimizer, scheduler = (
        build_optimizer_and_scheduler(
            model,
            epochs=int(
                config["epochs"]
            ),
            learning_rate=float(
                config["learning_rate"]
            ),
            weight_decay=float(
                config["weight_decay"]
            ),
            eta_min=float(
                config["eta_min"]
            ),
        )
    )

    amp_enabled = (
        bool(
            config.get(
                "amp",
                True,
            )
        )
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_mae = float("inf")
    best_epoch = None

    history = []

    for epoch_index in range(
        int(config["epochs"])
    ):

        epoch = (
            epoch_index + 1
        )

        train_metrics = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                lambda_slope=float(
                    config[
                        "lambda_slope"
                    ]
                ),
                gradient_clip_norm=float(
                    config[
                        "gradient_clip_norm"
                    ]
                ),
                amp_enabled=amp_enabled,
            )
        )

        val_metrics = (
            evaluate_elevation(
                model,
                val_loader,
                norm_stats,
                device,
            )
        )

        row = {
            "epoch": epoch,
            **{
                f"train_{key}":
                    value
                for key, value
                in train_metrics.items()
            },
            **{
                f"val_{key}":
                    value
                for key, value
                in val_metrics.items()
            },
            "lr":
                optimizer
                .param_groups[0]["lr"],
        }

        history.append(
            row
        )

        pd.DataFrame(
            history
        ).to_csv(
            args.output_dir
            / "training_history.csv",
            index=False,
        )

        if (
            val_metrics["mae"]
            < best_mae
        ):

            best_mae = float(
                val_metrics["mae"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_name":
                        config["model"],
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "scheduler_state_dict":
                        scheduler.state_dict(),
                    "scaler_state_dict":
                        scaler.state_dict(),
                    "best_val_mae":
                        best_mae,
                    "val_metrics":
                        val_metrics,
                },
                args.output_dir
                / "best_model.pt",
            )

        scheduler.step()

        print(
            f"epoch={epoch:03d} "
            f"train_loss="
            f"{train_metrics['total_loss']:.6f} "
            f"val_mae="
            f"{val_metrics['mae']:.6f} "
            f"best_epoch="
            f"{best_epoch}"
        )

    print(
        "\nBest epoch:",
        best_epoch,
    )

    print(
        "Best validation MAE:",
        best_mae,
    )


if __name__ == "__main__":
    main()

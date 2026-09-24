#!/usr/bin/env python3
"""
Verify the integrity and checkpoint compatibility of the public
DGRR-Net release.

Without --patch-root:
    checkpoint hash
    split counts
    architecture parameter count
    strict checkpoint loading
    synthetic 11-channel forward pass

With --patch-root:
    additionally verifies the real public Protocol-A dataset path.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import torch

from src.dataset import (
    ProtocolADualScaleDataset,
    attach_patch_files,
    load_normalization_stats,
)

from src.inference import load_checkpoint

from src.model import (
    E2cDualScaleResidualUNet,
)


EXPECTED_CHECKPOINT_SHA = (
    "e85cf834cbe50bb356a4021980f9fe7f"
    "642b3966af44471e42a2cc57acc985a8"
)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--support-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--patch-root",
        default=None,
        type=Path,
    )

    args = parser.parse_args()

    checkpoint_path = (
        args.support_dir
        / "checkpoints"
        / "rdeepdem_e2c_best.pt"
    )

    observed_sha = sha256(
        checkpoint_path
    )

    if observed_sha != EXPECTED_CHECKPOINT_SHA:
        raise AssertionError(
            "Checkpoint SHA-256 mismatch."
        )


    split = pd.read_csv(
        args.support_dir
        / "data"
        / "protocol_a_split.csv"
    )

    observed_counts = (
        split["split"]
        .value_counts()
        .to_dict()
    )

    expected_counts = {
        "train": 107,
        "val": 27,
        "test": 25,
        "unused_buffer": 54,
    }

    if observed_counts != expected_counts:
        raise AssertionError(
            f"Unexpected split counts: "
            f"{observed_counts}"
        )


    model = E2cDualScaleResidualUNet(
        input_channels=11,
        base_channels=32,
        dropout=0.05,
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    if parameter_count != 3_807_489:
        raise AssertionError(
            "E2c parameter-count mismatch."
        )


    checkpoint = load_checkpoint(
        model,
        checkpoint_path,
        device="cpu",
    )

    checkpoint_epoch = int(
        checkpoint["epoch"]
    )

    if checkpoint_epoch != 63:
        raise AssertionError(
            "Unexpected checkpoint epoch."
        )


    model.eval()


    if args.patch_root is None:

        x = torch.zeros(
            1,
            11,
            256,
            256,
            dtype=torch.float32,
        )

        with torch.no_grad():
            prediction = model(x)

        test_mode = "synthetic"


    else:

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
                "test",
                norm_stats,
            )
        )

        if len(dataset) != 25:
            raise AssertionError(
                "Expected 25 test patches."
            )

        sample = dataset[0]

        if tuple(
            sample["x"].shape
        ) != (
            11,
            256,
            256,
        ):
            raise AssertionError(
                "Unexpected real-patch input shape."
            )

        with torch.no_grad():

            prediction = model(
                sample["x"]
                .unsqueeze(0)
            )

        test_mode = "real_test_patch"


    if tuple(
        prediction.shape
    ) != (
        1,
        1,
        256,
        256,
    ):
        raise AssertionError(
            "Unexpected prediction shape."
        )


    if not torch.isfinite(
        prediction
    ).all():
        raise AssertionError(
            "Non-finite prediction."
        )


    print(
        "Checkpoint SHA-256      : PASS"
    )

    print(
        "Protocol-A split counts : PASS"
    )

    print(
        "E2c parameters          : "
        "3,807,489 PASS"
    )

    print(
        "Checkpoint epoch        : "
        "63 PASS"
    )

    print(
        "Forward smoke test      : "
        f"{test_mode} PASS"
    )


if __name__ == "__main__":
    main()

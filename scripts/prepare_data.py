#!/usr/bin/env python3
"""
Generate Phase-5-compatible patch NPZ files and a Protocol-A split
from locally available aligned six-band ROI stack GeoTIFFs.

Expected stack band order:
    1 SRTM
    2 USGS 3DEP
    3 NAIP R
    4 NAIP G
    5 NAIP B
    6 NAIP NIR

Expected file naming:
    phase5_<roi_id>_stack.tif

The public preprocessing module also contains the Earth Engine
availability/export helpers used to construct these aligned stacks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing import (
    generate_patches_for_roi,
    make_protocol_a_split,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stack-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing "
            "phase5_<roi_id>_stack.tif files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for patch NPZ files.",
    )

    parser.add_argument(
        "--inventory-csv",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--split-csv",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stack_files = sorted(
        args.stack_dir.glob(
            "phase5_*_stack.tif"
        )
    )

    if not stack_files:
        raise FileNotFoundError(
            "No phase5_*_stack.tif files found "
            f"in {args.stack_dir}"
        )

    records = []

    for stack_path in stack_files:
        records.extend(
            generate_patches_for_roi(
                tif_path=stack_path,
                out_root=args.output_dir,
            )
        )

    inventory = pd.DataFrame(
        records
    )

    if inventory.empty:
        raise RuntimeError(
            "No patches passed the frozen "
            "validity criterion."
        )

    args.inventory_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory.to_csv(
        args.inventory_csv,
        index=False,
    )

    split = make_protocol_a_split(
        inventory
    )

    args.split_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    split.to_csv(
        args.split_csv,
        index=False,
    )

    print(
        "Accepted patches:",
        len(inventory),
    )

    print(
        "\nSplit counts:"
    )

    print(
        split["split"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()

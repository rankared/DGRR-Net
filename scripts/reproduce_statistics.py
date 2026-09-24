#!/usr/bin/env python3
"""
Recompute the 27 planned E2c paired comparisons from the released
per-patch metric table.

Protocol:
    25 paired Protocol-A test patches
    E2c vs SRTM, B4b and E2a
    two-sided Wilcoxon signed-rank tests
    Holm family-wise correction across all 27 planned comparisons
    10,000 ROI-stratified paired bootstrap resamples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats

from src.statistics import (
    holm_adjust,
    matched_rank_biserial,
    paired_standardized_mean_difference,
    paired_bootstrap_improvement_ci,
    percent_improvement,
)


SEED = 42
BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05

COMPARATORS = [
    "srtm",
    "b4b",
    "e2a",
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--support-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    wide = pd.read_csv(
        args.support_dir
        / "results"
        / "per_patch_metrics.csv"
    )

    expected = pd.read_csv(
        args.support_dir
        / "results"
        / "paired_tests.csv"
    )

    # Preserve the exact metric order used by the frozen
    # Phase-7F-G planned-comparison table.
    metric_order = list(
        dict.fromkeys(
            expected["metric"].tolist()
        )
    )

    direction_map = (
        expected[
            [
                "metric",
                "direction",
            ]
        ]
        .drop_duplicates()
        .set_index(
            "metric"
        )["direction"]
        .to_dict()
    )

    rows = []

    for metric_position, metric in enumerate(
        metric_order
    ):

        direction = (
            direction_map[
                metric
            ]
        )

        for comparator_position, comparator in enumerate(
            COMPARATORS
        ):

            a = (
                wide[
                    wide["model_id"]
                    == "e2c"
                ][
                    [
                        "patch_key",
                        "roi_id",
                        metric,
                    ]
                ]
                .rename(
                    columns={
                        metric:
                            "e2c_value"
                    }
                )
            )

            b = (
                wide[
                    wide["model_id"]
                    == comparator
                ][
                    [
                        "patch_key",
                        metric,
                    ]
                ]
                .rename(
                    columns={
                        metric:
                            "comparator_value"
                    }
                )
            )

            paired = a.merge(
                b,
                on="patch_key",
                how="inner",
                validate="one_to_one",
            ).dropna()

            e2c_values = (
                paired["e2c_value"]
                .to_numpy(
                    dtype=np.float64
                )
            )

            comparator_values = (
                paired[
                    "comparator_value"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            if direction == "lower":

                improvements = (
                    comparator_values
                    - e2c_values
                )

            else:

                improvements = (
                    e2c_values
                    - comparator_values
                )

            if np.allclose(
                improvements,
                0.0,
            ):

                wilcoxon_statistic = 0.0
                wilcoxon_p_raw = 1.0

            else:

                result = stats.wilcoxon(
                    improvements,
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    method="auto",
                )

                wilcoxon_statistic = float(
                    result.statistic
                )

                wilcoxon_p_raw = float(
                    result.pvalue
                )

            (
                improvement_mean,
                ci_lower,
                ci_upper,
            ) = (
                paired_bootstrap_improvement_ci(
                    model_values=e2c_values,
                    comparator_values=
                        comparator_values,
                    roi_values=
                        paired[
                            "roi_id"
                        ].to_numpy(),
                    direction=direction,
                    resamples=
                        BOOTSTRAP_RESAMPLES,
                    confidence_level=
                        CONFIDENCE_LEVEL,
                    seed=(
                        SEED
                        + 200_000
                        + 1000
                        * metric_position
                        + 10
                        * comparator_position
                    ),
                )
            )

            e2c_mean = float(
                np.mean(
                    e2c_values
                )
            )

            comparator_mean = float(
                np.mean(
                    comparator_values
                )
            )

            rows.append({
                "metric":
                    metric,

                "direction":
                    direction,

                "model":
                    "e2c",

                "comparator":
                    comparator,

                "paired_patches":
                    len(paired),

                "e2c_mean":
                    e2c_mean,

                "comparator_mean":
                    comparator_mean,

                "mean_improvement_favouring_e2c":
                    improvement_mean,

                "improvement_ci_lower":
                    ci_lower,

                "improvement_ci_upper":
                    ci_upper,

                "percent_improvement_favouring_e2c":
                    percent_improvement(
                        baseline=
                            comparator_mean,
                        model=
                            e2c_mean,
                        direction=
                            direction,
                    ),

                "wilcoxon_statistic":
                    wilcoxon_statistic,

                "wilcoxon_p_raw":
                    wilcoxon_p_raw,

                "paired_rank_biserial":
                    matched_rank_biserial(
                        improvements
                    ),

                "paired_standardized_mean_difference_dz":
                    paired_standardized_mean_difference(
                        improvements
                    ),

                "bootstrap_resamples":
                    BOOTSTRAP_RESAMPLES,
            })

    reproduced = pd.DataFrame(
        rows
    )

    reproduced[
        "wilcoxon_p_holm"
    ] = holm_adjust(
        reproduced[
            "wilcoxon_p_raw"
        ].to_numpy(
            dtype=np.float64
        )
    )

    reproduced[
        "wilcoxon_significant_holm"
    ] = (
        reproduced[
            "wilcoxon_p_holm"
        ]
        < ALPHA
    )

    reproduced[
        "bootstrap_ci_excludes_zero"
    ] = (
        (
            reproduced[
                "improvement_ci_lower"
            ] > 0
        )
        |
        (
            reproduced[
                "improvement_ci_upper"
            ] < 0
        )
    )

    key = [
        "metric",
        "comparator",
    ]

    expected_check = (
        expected
        .sort_values(key)
        .reset_index(drop=True)
    )

    reproduced_check = (
        reproduced
        .sort_values(key)
        .reset_index(drop=True)
    )

    if len(expected_check) != 27:
        raise AssertionError(
            "Expected 27 frozen planned comparisons."
        )

    if len(reproduced_check) != 27:
        raise AssertionError(
            "Reproduction did not produce "
            "27 planned comparisons."
        )

    numeric_columns = [
        "paired_patches",
        "e2c_mean",
        "comparator_mean",
        "mean_improvement_favouring_e2c",
        "improvement_ci_lower",
        "improvement_ci_upper",
        "percent_improvement_favouring_e2c",
        "wilcoxon_statistic",
        "wilcoxon_p_raw",
        "paired_rank_biserial",
        "paired_standardized_mean_difference_dz",
        "bootstrap_resamples",
        "wilcoxon_p_holm",
    ]

    failures = []

    for column in numeric_columns:

        observed = (
            reproduced_check[
                column
            ].to_numpy(
                dtype=np.float64
            )
        )

        target = (
            expected_check[
                column
            ].to_numpy(
                dtype=np.float64
            )
        )

        if not np.allclose(
            observed,
            target,
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        ):

            max_difference = float(
                np.nanmax(
                    np.abs(
                        observed
                        - target
                    )
                )
            )

            failures.append(
                (
                    column,
                    max_difference,
                )
            )

    for column in [
        "wilcoxon_significant_holm",
        "bootstrap_ci_excludes_zero",
    ]:

        if not np.array_equal(
            reproduced_check[
                column
            ].astype(bool),
            expected_check[
                column
            ].astype(bool),
        ):
            failures.append(
                (
                    column,
                    "boolean mismatch",
                )
            )

    if failures:
        raise AssertionError(
            "Statistical reproduction mismatch:\n"
            + "\n".join(
                str(item)
                for item
                in failures
            )
        )

    if args.output_csv is not None:

        args.output_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        reproduced.to_csv(
            args.output_csv,
            index=False,
        )

    print(
        "Planned comparisons:",
        len(reproduced),
    )

    print(
        "Holm-adjusted paired tests: PASS"
    )

    print(
        "Bootstrap confidence intervals: PASS"
    )

    print(
        "Released paired_tests.csv reproduced: PASS"
    )


if __name__ == "__main__":
    main()

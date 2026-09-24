"""
Paired statistical-analysis primitives used for the manuscript benchmark.

Includes:
  * Holm family-wise correction;
  * paired rank-biserial effect size;
  * standardized paired mean difference;
  * ROI-stratified bootstrap confidence intervals;
  * paired bootstrap improvement intervals.
"""

from __future__ import annotations

import re

from typing import (
    List,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd

from scipy import stats

def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p_values_array = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(
        p_values_array.shape,
        np.nan,
        dtype=np.float64,
    )

    finite_indices = np.flatnonzero(
        np.isfinite(p_values_array)
    )

    if finite_indices.size == 0:
        return adjusted

    ordered_indices = finite_indices[
        np.argsort(
            p_values_array[finite_indices]
        )
    ]

    number_of_tests = int(
        ordered_indices.size
    )

    running_maximum = 0.0

    for order_position, original_index in enumerate(
        ordered_indices
    ):
        multiplier = (
            number_of_tests
            - order_position
        )

        candidate = min(
            1.0,
            multiplier
            * p_values_array[original_index],
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted[original_index] = (
            running_maximum
        )

    return adjusted


def matched_rank_biserial(differences: np.ndarray) -> float:
    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    differences = differences[
        np.isfinite(differences)
        & (differences != 0)
    ]

    if differences.size == 0:
        return 0.0

    ranks = stats.rankdata(
        np.abs(differences),
        method="average",
    )

    positive_rank_sum = float(
        ranks[differences > 0].sum()
    )

    negative_rank_sum = float(
        ranks[differences < 0].sum()
    )

    total_rank_sum = float(
        ranks.sum()
    )

    if total_rank_sum <= 0:
        return 0.0

    return float(
        (
            positive_rank_sum
            - negative_rank_sum
        )
        / total_rank_sum
    )


def paired_standardized_mean_difference(
    differences: np.ndarray,
) -> float:
    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if differences.size < 2:
        return np.nan

    standard_deviation = float(
        np.std(
            differences,
            ddof=1,
        )
    )

    if standard_deviation <= 0:
        return 0.0

    return float(
        np.mean(differences)
        / standard_deviation
    )


def percent_improvement(
    baseline: float,
    model: float,
    direction: str,
) -> float:
    if (
        not np.isfinite(baseline)
        or not np.isfinite(model)
        or baseline == 0
    ):
        return np.nan

    if direction == "lower":
        return float(
            100.0
            * (baseline - model)
            / abs(baseline)
        )

    return float(
        100.0
        * (model - baseline)
        / abs(baseline)
    )


def stratified_bootstrap_indices(
    roi_values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    roi_values = np.asarray(roi_values)

    sampled_indices: List[np.ndarray] = []

    for roi_id in sorted(
        pd.unique(roi_values).tolist()
    ):
        indices = np.flatnonzero(
            roi_values == roi_id
        )

        sampled_indices.append(
            rng.choice(
                indices,
                size=indices.size,
                replace=True,
            )
        )

    return np.concatenate(
        sampled_indices
    )


def stratified_bootstrap_mean_ci(
    values: np.ndarray,
    roi_values: np.ndarray,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> Tuple[float, float, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    roi_values = np.asarray(
        roi_values
    )

    finite = (
        np.isfinite(values)
        & pd.notna(roi_values)
    )

    values = values[finite]
    roi_values = roi_values[finite]

    if values.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(
        resamples,
        dtype=np.float64,
    )

    for bootstrap_index in range(resamples):
        sampled = stratified_bootstrap_indices(
            roi_values,
            rng,
        )

        bootstrap_means[bootstrap_index] = float(
            np.mean(
                values[sampled]
            )
        )

    tail_probability = (
        1.0
        - confidence_level
    ) / 2.0

    lower, upper = np.quantile(
        bootstrap_means,
        [
            tail_probability,
            1.0 - tail_probability,
        ],
    )

    return (
        float(np.mean(values)),
        float(lower),
        float(upper),
    )


def paired_bootstrap_improvement_ci(
    model_values: np.ndarray,
    comparator_values: np.ndarray,
    roi_values: np.ndarray,
    direction: str,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> Tuple[float, float, float]:
    model_values = np.asarray(
        model_values,
        dtype=np.float64,
    )

    comparator_values = np.asarray(
        comparator_values,
        dtype=np.float64,
    )

    roi_values = np.asarray(
        roi_values
    )

    finite = (
        np.isfinite(model_values)
        & np.isfinite(comparator_values)
        & pd.notna(roi_values)
    )

    model_values = model_values[finite]
    comparator_values = comparator_values[finite]
    roi_values = roi_values[finite]

    if direction == "lower":
        improvements = (
            comparator_values
            - model_values
        )
    else:
        improvements = (
            model_values
            - comparator_values
        )

    if improvements.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(
        resamples,
        dtype=np.float64,
    )

    for bootstrap_index in range(resamples):
        sampled = stratified_bootstrap_indices(
            roi_values,
            rng,
        )

        bootstrap_means[bootstrap_index] = float(
            np.mean(
                improvements[sampled]
            )
        )

    tail_probability = (
        1.0
        - confidence_level
    ) / 2.0

    lower, upper = np.quantile(
        bootstrap_means,
        [
            tail_probability,
            1.0 - tail_probability,
        ],
    )

    return (
        float(np.mean(improvements)),
        float(lower),
        float(upper),
    )


def infer_roi_from_patch_key(patch_key: str) -> str:
    match = re.match(
        r"(.+?)_p\d+_y-?\d+_x-?\d+$",
        str(patch_key),
    )

    return (
        match.group(1)
        if match
        else "unknown_roi"
    )


def raw_ci_to_percent(
    row: pd.Series,
    raw_column: str,
) -> float:
    denominator = abs(
        float(row["comparator_mean"])
    )

    if denominator <= 0:
        return np.nan

    return float(
        100.0
        * float(row[raw_column])
        / denominator
    )


def format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def paired_metric_frame(
    patch_metric_wide,
    metric_name,
    model_a,
    model_b,
):
    """
    Align two models by patch key for paired analysis.

    This is the public function-level form of the Phase-7F-G pairing
    operation; the original notebook used a global dataframe.
    """
    model_a_frame = (
        patch_metric_wide[
            patch_metric_wide[
                "model_id"
            ] == model_a
        ][
            [
                "patch_key",
                "roi_id",
                metric_name,
            ]
        ]
        .rename(
            columns={
                metric_name:
                    "model_a_value",

                "roi_id":
                    "roi_id_a",
            }
        )
    )

    model_b_frame = (
        patch_metric_wide[
            patch_metric_wide[
                "model_id"
            ] == model_b
        ][
            [
                "patch_key",
                "roi_id",
                metric_name,
            ]
        ]
        .rename(
            columns={
                metric_name:
                    "model_b_value",

                "roi_id":
                    "roi_id_b",
            }
        )
    )

    merged = model_a_frame.merge(
        model_b_frame,
        on="patch_key",
        how="inner",
        validate="one_to_one",
    )

    merged["roi_id"] = (
        merged["roi_id_a"]
        .fillna(
            merged["roi_id_b"]
        )
    )

    return merged.dropna(
        subset=[
            "model_a_value",
            "model_b_value",
            "roi_id",
        ]
    )

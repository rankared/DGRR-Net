"""
Patch-internal hydrological-consistency metrics used in Phase 7D.

Frozen primary protocol:
  * pixel size: 1 m;
  * Priority-Flood depression conditioning;
  * deterministic D8 flow routing;
  * sink threshold: 0.10 m;
  * stream threshold: 1,000 m^2 contributing area;
  * stream tolerance: 2 pixels;
  * evaluation buffer: 8 pixels.

These are patch-internal metrics. Flow accumulation is truncated at patch
boundaries and is not full-catchment hydrology.
"""

from __future__ import annotations

import heapq
import math

from collections import deque
from typing import Dict, Mapping, Tuple

import numpy as np

from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label,
)


PIXEL_SIZE_M = 1.0
HYDRO_EVAL_BUFFER_PIXELS = 8

SINK_DEPTH_THRESHOLDS_M = (
    0.05,
    0.10,
    0.25,
    0.50,
)

PRIMARY_SINK_THRESHOLD_M = 0.10

STREAM_AREA_THRESHOLDS_M2 = (
    500.0,
    1000.0,
    2000.0,
)

PRIMARY_STREAM_THRESHOLD_M2 = 1000.0

STREAM_TOLERANCE_PIXELS = 2.0

D8_NEIGHBOURS: Tuple[
    Tuple[
        int,
        int,
        float,
        int,
        float,
    ],
    ...,
] = (
    (-1,  0, 1.0,            0,   0.0),
    (-1,  1, math.sqrt(2.0), 1,  45.0),
    ( 0,  1, 1.0,            2,  90.0),
    ( 1,  1, math.sqrt(2.0), 3, 135.0),
    ( 1,  0, 1.0,            4, 180.0),
    ( 1, -1, math.sqrt(2.0), 5, 225.0),
    ( 0, -1, 1.0,            6, 270.0),
    (-1, -1, math.sqrt(2.0), 7, 315.0),
)

CONNECTIVITY_8 = np.ones(
    (3, 3),
    dtype=np.uint8,
)

def safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def circular_absolute_error_deg(predicted_deg: np.ndarray, reference_deg: np.ndarray) -> np.ndarray:
    delta = np.abs(predicted_deg - reference_deg)
    return np.minimum(delta, 360.0 - delta)


def error_summary(error: np.ndarray, prefix: str) -> Dict[str, float | int]:
    values = np.asarray(error, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mae": float("nan"),
            f"{prefix}_rmse": float("nan"),
            f"{prefix}_bias": float("nan"),
            f"{prefix}_median_absolute_error": float("nan"),
            f"{prefix}_p90_absolute_error": float("nan"),
            f"{prefix}_count": 0,
        }
    absolute = np.abs(values)
    return {
        f"{prefix}_mae": float(np.mean(absolute)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(values * values))),
        f"{prefix}_bias": float(np.mean(values)),
        f"{prefix}_median_absolute_error": float(np.median(absolute)),
        f"{prefix}_p90_absolute_error": float(np.percentile(absolute, 90)),
        f"{prefix}_count": int(values.size),
    }


def correlation_summary(predicted: np.ndarray, reference: np.ndarray) -> float:
    x = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(reference, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def outlet_seed_mask(valid_mask: np.ndarray) -> np.ndarray:
    """Return valid cells touching the raster edge or an invalid cell."""
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("valid_mask must be 2D")

    seeds = np.zeros_like(valid)
    seeds[0, :] = valid[0, :]
    seeds[-1, :] = valid[-1, :]
    seeds[:, 0] = valid[:, 0]
    seeds[:, -1] = valid[:, -1]

    adjacent_to_invalid = binary_dilation(~valid, structure=CONNECTIVITY_8) & valid
    seeds |= adjacent_to_invalid
    return seeds


def priority_flood_fill(
    dem: np.ndarray,
    valid_mask: np.ndarray,
    *,
    epsilon: bool,
) -> np.ndarray:
    """
    Fill depressions with an 8-connected Priority-Flood.

    When epsilon=True, cells that would otherwise be flat are raised with
    np.nextafter so every flooded cell has a strictly descending path towards
    an outlet. Calculations use float64 to preserve these tiny increments.
    """
    elevation = np.asarray(dem, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if elevation.shape != valid.shape:
        raise ValueError("DEM and valid mask shapes differ")
    if not np.all(np.isfinite(elevation[valid])):
        raise ValueError("DEM contains non-finite valid elevations")

    rows, cols = elevation.shape
    filled = elevation.copy()
    visited = np.zeros_like(valid)
    seeds = outlet_seed_mask(valid)
    heap: List[Tuple[float, int, int]] = []

    seed_rows, seed_cols = np.nonzero(seeds)
    for row, col in zip(seed_rows.tolist(), seed_cols.tolist()):
        visited[row, col] = True
        heapq.heappush(heap, (float(filled[row, col]), row, col))

    if not heap and np.any(valid):
        raise RuntimeError("No Priority-Flood outlet seeds were found")

    while heap:
        current_elevation, row, col = heapq.heappop(heap)
        for dr, dc, _distance, _code, _angle in D8_NEIGHBOURS:
            nr = row + dr
            nc = col + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if not valid[nr, nc] or visited[nr, nc]:
                continue

            visited[nr, nc] = True
            raw = float(elevation[nr, nc])
            if epsilon and raw <= current_elevation:
                conditioned = float(np.nextafter(current_elevation, np.inf))
            else:
                conditioned = max(raw, current_elevation)

            filled[nr, nc] = conditioned
            heapq.heappush(heap, (conditioned, nr, nc))

    if not np.array_equal(visited, valid):
        missing = int(np.sum(valid & ~visited))
        raise RuntimeError(f"Priority-Flood did not visit {missing} valid cells")

    filled[~valid] = np.nan
    return filled


def d8_receivers(
    conditioned_dem: np.ndarray,
    valid_mask: np.ndarray,
    pixel_size_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each valid cell to its steepest strictly downslope D8 neighbour."""
    dem = np.asarray(conditioned_dem, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    rows, cols = dem.shape
    total = rows * cols
    receivers = np.full(total, -1, dtype=np.int64)
    direction_codes = np.full((rows, cols), -1, dtype=np.int8)

    valid_rows, valid_cols = np.nonzero(valid)
    for row, col in zip(valid_rows.tolist(), valid_cols.tolist()):
        source_value = float(dem[row, col])
        best_gradient = 0.0
        best_receiver = -1
        best_code = -1

        for dr, dc, distance_cells, code, _angle in D8_NEIGHBOURS:
            nr = row + dr
            nc = col + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if not valid[nr, nc]:
                continue
            drop = source_value - float(dem[nr, nc])
            gradient = drop / (distance_cells * pixel_size_m)
            if gradient > best_gradient:
                best_gradient = gradient
                best_receiver = nr * cols + nc
                best_code = code

        source_index = row * cols + col
        receivers[source_index] = best_receiver
        direction_codes[row, col] = best_code

    return receivers, direction_codes


def flow_accumulation(
    receivers: np.ndarray,
    valid_mask: np.ndarray,
    pixel_area_m2: float,
) -> Tuple[np.ndarray, int]:
    """Compute D8 contributing area with a topological source-to-outlet pass."""
    valid = np.asarray(valid_mask, dtype=bool)
    total = valid.size
    receiver = np.asarray(receivers, dtype=np.int64)
    if receiver.shape != (total,):
        raise ValueError("Receiver array has an unexpected shape")

    valid_flat = valid.ravel()
    indegree = np.zeros(total, dtype=np.int32)
    accumulation_cells = np.zeros(total, dtype=np.float64)
    accumulation_cells[valid_flat] = 1.0

    valid_indices = np.flatnonzero(valid_flat)
    for source in valid_indices.tolist():
        destination = int(receiver[source])
        if destination >= 0:
            if not valid_flat[destination]:
                raise RuntimeError("A D8 receiver points to an invalid cell")
            indegree[destination] += 1

    queue: deque[int] = deque(
        int(index) for index in valid_indices[indegree[valid_indices] == 0].tolist()
    )
    processed = 0

    while queue:
        source = queue.popleft()
        processed += 1
        destination = int(receiver[source])
        if destination >= 0:
            accumulation_cells[destination] += accumulation_cells[source]
            indegree[destination] -= 1
            if indegree[destination] == 0:
                queue.append(destination)

    expected = int(valid_flat.sum())
    if processed != expected:
        raise RuntimeError(
            f"D8 graph contains a cycle or disconnected dependency: processed={processed}, expected={expected}"
        )

    accumulation_area = accumulation_cells.reshape(valid.shape) * pixel_area_m2
    accumulation_area[~valid] = np.nan
    outlets = int(np.sum(valid_flat & (receiver < 0)))
    return accumulation_area, outlets


def hydrology_evaluation_mask(valid_mask: np.ndarray, buffer_pixels: int) -> np.ndarray:
    """Common interior support used for all hydrological metrics."""
    valid = np.asarray(valid_mask, dtype=bool)
    support = binary_erosion(valid, structure=CONNECTIVITY_8, iterations=1, border_value=0)
    if buffer_pixels > 0:
        support[:buffer_pixels, :] = False
        support[-buffer_pixels:, :] = False
        support[:, :buffer_pixels] = False
        support[:, -buffer_pixels:] = False
    return support


def compute_hydrology_fields(dem: np.ndarray, valid_mask: np.ndarray) -> Dict[str, np.ndarray | int]:
    filled = priority_flood_fill(dem, valid_mask, epsilon=False)
    conditioned = priority_flood_fill(dem, valid_mask, epsilon=True)
    fill_depth = filled - np.asarray(dem, dtype=np.float64)
    fill_depth[~valid_mask] = np.nan
    fill_depth[np.abs(fill_depth) < 1e-12] = 0.0

    receivers, direction_codes = d8_receivers(conditioned, valid_mask, PIXEL_SIZE_M)
    accumulation_area, outlet_count = flow_accumulation(
        receivers,
        valid_mask,
        PIXEL_SIZE_M * PIXEL_SIZE_M,
    )
    log10_accumulation = np.log10(1.0 + accumulation_area)
    log10_accumulation[~valid_mask] = np.nan

    return {
        "filled_dem": filled,
        "conditioned_dem": conditioned,
        "fill_depth": fill_depth,
        "direction_codes": direction_codes,
        "flow_accumulation_area_m2": accumulation_area,
        "log10_flow_accumulation": log10_accumulation,
        "outlet_count": outlet_count,
    }


def binary_counts(predicted: np.ndarray, reference: np.ndarray) -> Dict[str, int]:
    pred = np.asarray(predicted, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    return {
        "tp": int(np.sum(pred & ref)),
        "fp": int(np.sum(pred & ~ref)),
        "fn": int(np.sum(~pred & ref)),
        "tn": int(np.sum(~pred & ~ref)),
    }


def metrics_from_counts(counts: Mapping[str, int], prefix: str) -> Dict[str, float | int]:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
    iou = safe_divide(tp, tp + fp + fn)
    return {
        f"{prefix}_tp": tp,
        f"{prefix}_fp": fp,
        f"{prefix}_fn": fn,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_iou": iou,
    }


def tolerance_stream_counts(
    predicted_stream: np.ndarray,
    reference_stream: np.ndarray,
    tolerance_pixels: float,
) -> Dict[str, int]:
    pred = np.asarray(predicted_stream, dtype=bool)
    ref = np.asarray(reference_stream, dtype=bool)
    pred_count = int(pred.sum())
    ref_count = int(ref.sum())

    if pred_count > 0:
        distance_to_pred = distance_transform_edt(~pred)
        matched_reference = int(np.sum(ref & (distance_to_pred <= tolerance_pixels)))
    else:
        matched_reference = 0

    if ref_count > 0:
        distance_to_ref = distance_transform_edt(~ref)
        matched_prediction = int(np.sum(pred & (distance_to_ref <= tolerance_pixels)))
    else:
        matched_prediction = 0

    return {
        "matched_reference": matched_reference,
        "reference_count": ref_count,
        "matched_prediction": matched_prediction,
        "prediction_count": pred_count,
    }


def tolerance_metrics_from_counts(counts: Mapping[str, int], prefix: str) -> Dict[str, float | int]:
    matched_ref = int(counts["matched_reference"])
    ref_count = int(counts["reference_count"])
    matched_pred = int(counts["matched_prediction"])
    pred_count = int(counts["prediction_count"])
    recall = safe_divide(matched_ref, ref_count)
    precision = safe_divide(matched_pred, pred_count)
    if math.isfinite(precision) and math.isfinite(recall) and precision + recall > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = float("nan")
    return {
        f"{prefix}_matched_reference": matched_ref,
        f"{prefix}_reference_count": ref_count,
        f"{prefix}_matched_prediction": matched_pred,
        f"{prefix}_prediction_count": pred_count,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
    }


def object_detection_counts(predicted_sink: np.ndarray, reference_sink: np.ndarray) -> Dict[str, int]:
    pred_labels, pred_objects = label(predicted_sink, structure=CONNECTIVITY_8)
    ref_labels, ref_objects = label(reference_sink, structure=CONNECTIVITY_8)

    detected_ref = 0
    for object_id in range(1, ref_objects + 1):
        if np.any(pred_labels[ref_labels == object_id] > 0):
            detected_ref += 1

    matched_pred = 0
    for object_id in range(1, pred_objects + 1):
        if np.any(ref_labels[pred_labels == object_id] > 0):
            matched_pred += 1

    return {
        "matched_reference_objects": int(detected_ref),
        "reference_objects": int(ref_objects),
        "matched_prediction_objects": int(matched_pred),
        "prediction_objects": int(pred_objects),
    }


def object_metrics_from_counts(counts: Mapping[str, int], prefix: str) -> Dict[str, float | int]:
    matched_ref = int(counts["matched_reference_objects"])
    ref_objects = int(counts["reference_objects"])
    matched_pred = int(counts["matched_prediction_objects"])
    pred_objects = int(counts["prediction_objects"])
    recall = safe_divide(matched_ref, ref_objects)
    precision = safe_divide(matched_pred, pred_objects)
    if math.isfinite(precision) and math.isfinite(recall) and precision + recall > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = float("nan")
    return {
        f"{prefix}_matched_reference_objects": matched_ref,
        f"{prefix}_reference_objects": ref_objects,
        f"{prefix}_matched_prediction_objects": matched_pred,
        f"{prefix}_prediction_objects": pred_objects,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
    }


def direction_metrics(
    predicted_codes: np.ndarray,
    reference_codes: np.ndarray,
    support: np.ndarray,
) -> Dict[str, float | int | np.ndarray]:
    common = support & (predicted_codes >= 0) & (reference_codes >= 0)
    pred_codes = predicted_codes[common].astype(np.int16)
    ref_codes = reference_codes[common].astype(np.int16)
    if pred_codes.size == 0:
        return {
            "agreement_count": 0,
            "comparison_count": 0,
            "angle_errors": np.asarray([], dtype=np.float64),
        }
    agreement = int(np.sum(pred_codes == ref_codes))
    angles = np.asarray([item[4] for item in D8_NEIGHBOURS], dtype=np.float64)
    angle_error = circular_absolute_error_deg(angles[pred_codes], angles[ref_codes])
    return {
        "agreement_count": agreement,
        "comparison_count": int(pred_codes.size),
        "angle_errors": angle_error,
    }

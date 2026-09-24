"""
Elevation and physical terrain metrics used in Phase 7.

The legacy gradient metric and the physical 1 m terrain derivatives are
kept separate, matching the manuscript evaluation workflow.
"""

from __future__ import annotations

import numpy as np

from scipy.ndimage import binary_erosion

def neighbour_gradient_values(error: np.ndarray, valid: np.ndarray):
    """
    Legacy Phase-6 'slope' metric components.
    This measures first differences of the elevation error in m/pixel.
    It is NOT physical slope angle in degrees.
    """
    valid_x = valid[:, 1:] & valid[:, :-1]
    valid_y = valid[1:, :] & valid[:-1, :]

    dx_error = error[:, 1:] - error[:, :-1]
    dy_error = error[1:, :] - error[:-1, :]

    values_x = dx_error[valid_x].astype(np.float64, copy=False)
    values_y = dy_error[valid_y].astype(np.float64, copy=False)
    return values_x, values_y


def scalar_error_metrics(error_values: np.ndarray):
    error_values = np.asarray(error_values, dtype=np.float64)
    abs_error = np.abs(error_values)
    return {
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error_values)))),
        "bias": float(error_values.mean()),
        "median_absolute_error": float(np.median(abs_error)),
        "p90_absolute_error": float(np.percentile(abs_error, 90)),
        "p95_absolute_error": float(np.percentile(abs_error, 95)),
        "valid_pixels": int(error_values.size),
    }


def gradient_metrics(dx_values: np.ndarray, dy_values: np.ndarray):
    values = np.concatenate([dx_values, dy_values])
    return {
        "legacy_gradient_mae": float(np.mean(np.abs(values))),
        "legacy_gradient_rmse": float(np.sqrt(np.mean(np.square(values)))),
        "legacy_gradient_error_std": float(np.std(values, ddof=0)),
        "valid_slope_pairs": int(values.size),
    }


def complete_metrics(error: np.ndarray, valid: np.ndarray):
    valid = valid.astype(bool)
    error_values = error[valid]
    dx_values, dy_values = neighbour_gradient_values(error, valid)
    return {
        **scalar_error_metrics(error_values),
        **gradient_metrics(dx_values, dy_values),
    }

def terrain_fields(dem: np.ndarray, valid_mask: np.ndarray, pixel_size_m: float):
    """
    Compute physical terrain fields on a common one-pixel-eroded 3x3 support.

    Coordinate convention:
      x increases eastward;
      y increases northward;
      aspect is downslope azimuth clockwise from north in [0, 360).

    Returns arrays with original DEM shape and NaN outside derivative support.
    """
    dem = np.asarray(dem, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(dem)

    if dem.ndim != 2 or valid.shape != dem.shape:
        raise ValueError('DEM and valid mask must be matching 2D arrays.')
    if dem.shape[0] < 3 or dem.shape[1] < 3:
        raise ValueError('DEM must be at least 3x3.')

    support = binary_erosion(
        valid,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
        border_value=0,
    )

    center = dem[1:-1, 1:-1]
    east = dem[1:-1, 2:]
    west = dem[1:-1, :-2]
    north = dem[:-2, 1:-1]
    south = dem[2:, 1:-1]
    northeast = dem[:-2, 2:]
    northwest = dem[:-2, :-2]
    southeast = dem[2:, 2:]
    southwest = dem[2:, :-2]

    h = float(pixel_size_m)
    h2 = h * h

    # First derivatives in eastward and northward coordinates.
    dzdx = (east - west) / (2.0 * h)
    dzdy_north = (north - south) / (2.0 * h)

    # Second derivatives.
    d2zdx2 = (east - 2.0 * center + west) / h2
    d2zdy2 = (north - 2.0 * center + south) / h2
    d2zdxdy = (northeast - northwest - southeast + southwest) / (4.0 * h2)

    gradient_magnitude = np.hypot(dzdx, dzdy_north)
    slope_deg = np.degrees(np.arctan(gradient_magnitude))

    # Downslope direction = negative gradient; azimuth clockwise from north.
    downslope_east = -dzdx
    downslope_north = -dzdy_north
    aspect_deg = (np.degrees(np.arctan2(downslope_east, downslope_north)) + 360.0) % 360.0

    laplacian = d2zdx2 + d2zdy2

    # Mean curvature of graph z=f(x,y), units 1/m.
    denominator = 2.0 * np.power(1.0 + dzdx**2 + dzdy_north**2, 1.5)
    mean_curvature = (
        (1.0 + dzdy_north**2) * d2zdx2
        - 2.0 * dzdx * dzdy_north * d2zdxdy
        + (1.0 + dzdx**2) * d2zdy2
    ) / np.maximum(denominator, 1e-12)

    fields = {}
    inner_support = support[1:-1, 1:-1]
    for name, inner_values in {
        'dzdx': dzdx,
        'dzdy_north': dzdy_north,
        'gradient_magnitude': gradient_magnitude,
        'slope_deg': slope_deg,
        'aspect_deg': aspect_deg,
        'laplacian': laplacian,
        'mean_curvature': mean_curvature,
    }.items():
        full = np.full(dem.shape, np.nan, dtype=np.float32)
        destination = full[1:-1, 1:-1]
        destination[inner_support] = inner_values[inner_support].astype(np.float32)
        fields[name] = full

    fields['support_mask'] = support
    return fields


def circular_absolute_error_deg(predicted_deg: np.ndarray, reference_deg: np.ndarray):
    delta = (predicted_deg - reference_deg + 180.0) % 360.0 - 180.0
    return np.abs(delta)


def pearson_correlation(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size != x.size:
        return np.nan
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std <= 1e-12 or y_std <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def error_summary(error_values: np.ndarray, prefix: str):
    values = np.asarray(error_values, dtype=np.float64)
    if values.size == 0:
        return {
            f'{prefix}_mae': np.nan,
            f'{prefix}_rmse': np.nan,
            f'{prefix}_bias': np.nan,
            f'{prefix}_error_std': np.nan,
            f'{prefix}_median_absolute_error': np.nan,
            f'{prefix}_p90_absolute_error': np.nan,
            f'{prefix}_count': 0,
        }
    absolute = np.abs(values)
    return {
        f'{prefix}_mae': float(np.mean(absolute)),
        f'{prefix}_rmse': float(np.sqrt(np.mean(values**2))),
        f'{prefix}_bias': float(np.mean(values)),
        f'{prefix}_error_std': float(np.std(values, ddof=0)),
        f'{prefix}_median_absolute_error': float(np.median(absolute)),
        f'{prefix}_p90_absolute_error': float(np.percentile(absolute, 90)),
        f'{prefix}_count': int(values.size),
    }


def circular_error_summary(absolute_error_values: np.ndarray):
    values = np.asarray(absolute_error_values, dtype=np.float64)
    if values.size == 0:
        return {
            'aspect_circular_mae_deg': np.nan,
            'aspect_circular_rmse_deg': np.nan,
            'aspect_circular_median_error_deg': np.nan,
            'aspect_circular_p90_error_deg': np.nan,
            'aspect_count': 0,
        }
    return {
        'aspect_circular_mae_deg': float(np.mean(values)),
        'aspect_circular_rmse_deg': float(np.sqrt(np.mean(values**2))),
        'aspect_circular_median_error_deg': float(np.median(values)),
        'aspect_circular_p90_error_deg': float(np.percentile(values, 90)),
        'aspect_count': int(values.size),
    }

"""Explicit, approximate metric anchoring of an unchanged relative teacher.

This fits two *model estimates* on one RGB grid. It does not recover measured
camera calibration, validate absolute accuracy, or change either input array.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class PseudoCalibrationError(ValueError):
    """The supplied model planes cannot support a stable metric estimate."""


def _inverse_relative(values: np.ndarray, units: str) -> np.ndarray:
    return values.astype(np.float64) if units == "relative_inverse_depth" else 1.0 / values.astype(np.float64)


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # Normalize only the small fitting copy, never either scientific plane.
    center = float(np.median(x))
    spread = float(np.percentile(x, 95) - np.percentile(x, 5))
    if spread <= max(abs(center), 1.0) * 1e-7:
        raise PseudoCalibrationError("Relative teacher is constant or has insufficient depth variation")
    u = (x - center) / spread
    order = np.argsort(u)
    bins = np.array_split(order, 64)
    bx = np.array([np.median(u[index]) for index in bins])
    by = np.array([np.median(y[index]) for index in bins])
    ii, jj = np.triu_indices(len(bx), 1)
    separation = bx[jj] - bx[ii]
    useful = separation > .1
    slope = float(np.median((by[jj[useful]] - by[ii[useful]]) / separation[useful]))
    intercept = float(np.median(y - slope * u))
    design = np.column_stack((u, np.ones_like(u)))
    denominator = np.maximum(y, np.median(y) * .1)
    # Relative-error Huber IRLS avoids a few near-camera pixels dominating scale.
    for _ in range(16):
        residual = (slope * u + intercept - y) / denominator
        deviation = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        cutoff = max(.01, 1.345 * float(deviation))
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-12)) / denominator**2
        weighted = design * np.sqrt(weights[:, None])
        parameters = np.linalg.lstsq(weighted, y * np.sqrt(weights), rcond=None)[0]
        slope, intercept = float(parameters[0]), float(parameters[1])
    scale = slope / spread
    offset = intercept - scale * center
    if not np.isfinite([scale, offset]).all() or scale <= 0:
        raise PseudoCalibrationError("Relative and metric teachers do not support a positive finite scale")
    return scale, offset


def anchor_relative_depth(relative: np.ndarray, units: str, metric_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit ``1/Z_est = scale * relative_inverse + offset`` on the same RGB grid.

    DA2 supplies relative_inverse directly; DA3's relative_depth is reciprocated
    in a separate fitting copy. The caller must verify identical input RGB hashes
    and coordinates: equal array dimensions alone cannot prove registration.
    Rejected fits raise PseudoCalibrationError; no fallback is selected here.
    """
    relative = np.asarray(relative)
    metric_depth = np.asarray(metric_depth)
    if units not in {"relative_inverse_depth", "relative_depth"}:
        raise PseudoCalibrationError("A relative-depth or relative-inverse-depth teacher is required")
    if relative.ndim != 2 or relative.shape != metric_depth.shape or metric_depth.ndim != 2:
        raise PseudoCalibrationError("Both teacher planes must have the exact same HxW RGB grid")
    if relative.dtype.kind != "f" or metric_depth.dtype.kind != "f":
        raise PseudoCalibrationError("Teacher planes must contain floating-point scientific values")
    height, width = relative.shape
    if min(height, width) < 32:
        raise PseudoCalibrationError("Teacher grids are too small for spatially held-out validation")
    joint = np.isfinite(relative) & (relative > 0) & np.isfinite(metric_depth) & (metric_depth > 0)
    rng = np.random.default_rng(1949)
    training_x, training_y, holdouts = [], [], []
    for row in range(8):
        y0, y1 = row * height // 8, (row + 1) * height // 8
        for column in range(8):
            x0, x1 = column * width // 8, (column + 1) * width // 8
            local = np.flatnonzero(joint[y0:y1, x0:x1])
            if len(local) < 32:
                continue
            selected = rng.choice(local, min(512, len(local)), replace=False)
            yy, xx = np.unravel_index(selected, (y1 - y0, x1 - x0))
            xx, yy = xx + x0, yy + y0
            x = _inverse_relative(relative[yy, xx], units)
            y = 1.0 / metric_depth[yy, xx].astype(np.float64)
            halfway = len(x) // 2
            training_x.append(x[:halfway])
            training_y.append(y[:halfway])
            holdouts.append((row, column, x[halfway:], y[halfway:]))
    if len(holdouts) < 32 or sum(len(x) for x in training_x) < 1024:
        raise PseudoCalibrationError("Insufficient positive overlap across the 64 validation tiles")
    train_x, train_y = np.concatenate(training_x), np.concatenate(training_y)
    if np.percentile(train_y, 95) - np.percentile(train_y, 5) <= np.median(train_y) * .02:
        raise PseudoCalibrationError("Metric anchor has insufficient depth variation for a stable scale")
    scale, offset = _fit_affine(train_x, train_y)
    held_x = np.concatenate([entry[2] for entry in holdouts])
    held_y = np.concatenate([entry[3] for entry in holdouts])
    ranks_x = np.argsort(np.argsort(held_x)).astype(np.float64)
    ranks_y = np.argsort(np.argsort(held_y)).astype(np.float64)
    rank_correlation = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
    if rank_correlation < .5:
        raise PseudoCalibrationError("Teachers have weak held-out geometric agreement (rank correlation below 0.5)")
    tiles, errors, agreeing_tiles = [], [], 0
    for row, column, x, y in holdouts:
        estimated_inverse = scale * x + offset
        error = np.full(y.shape, np.inf)
        positive = np.isfinite(estimated_inverse) & (estimated_inverse > 0)
        error[positive] = np.abs(y[positive] / estimated_inverse[positive] - 1.0)
        errors.append(error)
        tile_median = float(np.median(error))
        tile_p90 = float(np.percentile(error, 90, method="higher"))
        agreeing_tiles += int(tile_median <= .4)
        tiles.append({"row": row, "column": column, "heldout_count": len(error),
            "median_relative_depth_error": tile_median if np.isfinite(tile_median) else None,
            "p90_relative_depth_error": tile_p90 if np.isfinite(tile_p90) else None,
            "nonpositive_prediction_count": int((~positive).sum()),
            "null_error_means_unbounded": not np.isfinite([tile_median, tile_p90]).all()})
    all_errors = np.concatenate(errors)
    median = float(np.median(all_errors))
    p90 = float(np.percentile(all_errors, 90, method="higher"))
    if median > .35 or p90 > 1.0 or agreeing_tiles < 32:
        raise PseudoCalibrationError(f"Metric anchoring rejected: held-out relative depth errors median={median:.3f}, p90={p90:.3f}; agreeing tiles={agreeing_tiles}/64")
    anchored = np.full(relative.shape, np.nan, dtype=np.float32)
    valid = np.zeros(relative.shape, dtype=bool)
    for y0 in range(0, height, 128):
        block = relative[y0:y0 + 128]
        block_valid = np.isfinite(block) & (block > 0)
        inverse = np.full(block.shape, np.nan, dtype=np.float64)
        inverse[block_valid] = scale * _inverse_relative(block[block_valid], units) + offset
        block_valid &= np.isfinite(inverse) & (inverse > 0)
        with np.errstate(over="ignore", divide="ignore"):
            output = np.full(block.shape, np.nan, dtype=np.float32)
            output[block_valid] = 1.0 / inverse[block_valid]
        block_valid &= np.isfinite(output) & (output > 0)
        output[~block_valid] = np.nan
        anchored[y0:y0 + 128], valid[y0:y0 + 128] = output, block_valid
    input_valid_count = int((np.isfinite(relative) & (relative > 0)).sum())
    if valid.sum() < .95 * input_valid_count:
        raise PseudoCalibrationError("Fitted inverse depth is nonpositive or nonfinite for too much of the relative teacher")
    metadata = {
        "schema": "ipde.model_metric_anchor.v1", "accepted": True,
        "label_kind": "model_anchored_pseudo_depth", "units": "meters", "input_units": units,
        "scale_inverse_meters_per_relative_inverse_unit": scale, "offset_inverse_meters": offset,
        "fit_formula": "1/Z_est_meters = scale * relative_inverse + offset",
        "relative_inverse_formula": "relative" if units == "relative_inverse_depth" else "1/relative",
        "anchor_is_measured": False, "camera_calibration_recovered": False,
        "input_rgb_grid_requirement": "Caller must verify both input RGB hashes and coordinates are identical",
        "raw_inputs_modified": False, "output_dtype": "float32",
        "fit": {"method": "spatially balanced robust affine inverse-depth fit with Huber IRLS", "seed": 1949,
            "training_sample_count": len(train_x), "relative_inverse_fit_min": float(train_x.min()),
            "relative_inverse_fit_max": float(train_x.max())},
        "validation": {"grid": [8, 8], "heldout_sample_count": len(held_x), "rank_correlation": rank_correlation,
            "median_relative_depth_error": median, "p90_relative_depth_error": p90,
            "agreeing_tile_count": agreeing_tiles, "spatial_tiles": tiles,
            "error_definition": "abs(anchored_depth - metric_model_depth) / metric_model_depth",
            "agreement_is_absolute_accuracy": False},
        "valid_pixel_count": int(valid.sum()),
        "limitation": "Scale is borrowed from another model estimate. Agreement neither measures true depth error nor proves stereo correctness.",
    }
    return anchored, valid, metadata

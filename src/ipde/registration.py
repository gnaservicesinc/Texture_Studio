"""Evidence-checked display/stereo registration for derived depth products.

The display image has no exported pinhole calibration in the examined Apple
files. A same-camera RGB homography is therefore an empirical approximation,
not a recovered camera calibration. Unsupported regions stay NaN.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


class RegistrationError(ValueError):
    """The available evidence cannot register a derived depth plane safely."""


def _asset(discovery: Any, name: str) -> Any:
    assets = [asset for asset in discovery.assets if asset.semantic_name == name]
    if len(assets) != 1:
        raise RegistrationError(f"registration requires exactly one {name} image")
    return assets[0]


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def _fit_matches(display_points: np.ndarray, stereo_points: np.ndarray,
                 display_shape: tuple[int, int], stereo_shape: tuple[int, int]) -> dict[str, Any]:
    """Fit with 75% of matches; validate the untouched 25% spatially."""
    import cv2

    record: dict[str, Any] = {"accepted": False, "reason": "insufficient feature correspondences",
                              "feature_match_count": len(display_points)}
    display_points = np.asarray(display_points, dtype=np.float64).reshape(-1, 2)
    stereo_points = np.asarray(stereo_points, dtype=np.float64).reshape(-1, 2)
    if display_points.shape != stereo_points.shape or not np.isfinite(display_points).all() or not np.isfinite(stereo_points).all():
        record["reason"] = "feature correspondences have different shapes or nonfinite coordinates"
        return record
    # SIFT can emit multiple orientations for one keypoint. They are a single
    # geometric observation, and must not leak from the fit into validation or
    # inflate a cell's independent correspondence count.
    if len(display_points):
        _, unique = np.unique(display_points, axis=0, return_index=True)
        unique.sort()
        display_points, stereo_points = display_points[unique], stereo_points[unique]
        _, unique = np.unique(stereo_points, axis=0, return_index=True)
        unique.sort()
        display_points, stereo_points = display_points[unique], stereo_points[unique]
    record["unique_feature_match_count"] = len(display_points)
    if len(display_points) < 64:
        return record
    indices = np.random.default_rng(42).permutation(len(display_points))
    split = int(len(indices) * .75)
    train, heldout = indices[:split], indices[split:]
    cv2.setRNGSeed(42)
    matrix, inliers = cv2.findHomography(display_points[train], stereo_points[train],
                                       cv2.USAC_MAGSAC, 2., maxIters=10000, confidence=.999)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        record["reason"] = "feature fit is singular or nonfinite"
        return record
    errors = np.linalg.norm(_project(display_points[heldout], matrix) - stereo_points[heldout], axis=1)
    height, width = stereo_shape
    cells, support = [], np.zeros((4, 4), dtype=bool)
    for y in range(4):
        for x in range(4):
            points = stereo_points[heldout]
            selected = ((points[:, 0] >= x * width / 4) & (points[:, 0] < (x + 1) * width / 4)
                        & (points[:, 1] >= y * height / 4) & (points[:, 1] < (y + 1) * height / 4))
            count = int(selected.sum())
            median = float(np.median(errors[selected])) if count else None
            p90 = float(np.percentile(errors[selected], 90)) if count else None
            support[y, x] = count >= 4 and median <= 1.25 and p90 <= 3.
            cells.append({"x": x, "y": y, "heldout_count": count,
                          "median_error_pixels": median, "p90_error_pixels": p90,
                          "supported": bool(support[y, x])})
    median, p90 = np.percentile(errors, [50, 90])
    hull = cv2.convexHull(stereo_points[train][inliers[:, 0].astype(bool)].astype(np.float32))
    coverage = float(cv2.contourArea(hull) / (height * width))
    corners = np.array([[0., 0.], [display_shape[1] - 1., 0.],
                        [0., display_shape[0] - 1.], [display_shape[1] - 1., display_shape[0] - 1.]])
    denominator = np.column_stack((corners, np.ones(4))) @ matrix[2]
    projective_change = float(np.ptp(denominator) / abs(np.mean(denominator)))
    record.update({"display_to_stereo_homography": matrix.tolist(),
                   "stereo_to_display_homography": np.linalg.inv(matrix).tolist(),
                   "heldout_count": len(heldout), "heldout_median_error_pixels": float(median),
                   "heldout_p90_error_pixels": float(p90),
                   "heldout_fraction_within_2_pixels": float(np.mean(errors < 2.)),
                   "inlier_hull_fraction": coverage, "stereo_inlier_hull": hull.reshape(-1, 2).tolist(),
                   "spatial_validation_cells": cells,
                   "supported_cells": support.tolist(), "projective_denominator_change": projective_change,
                   "validation_thresholds": {"minimum_matches": 64, "maximum_heldout_median_pixels": 1.,
                       "maximum_heldout_p90_pixels": 3., "minimum_hull_fraction": .5,
                       "minimum_supported_cells": 8, "maximum_projective_denominator_change": .01,
                       "cell_minimum_heldout": 4, "cell_maximum_median_pixels": 1.25,
                       "cell_maximum_p90_pixels": 3.}})
    record["accepted"] = bool(median <= 1. and p90 <= 3. and coverage >= .5
                               and support.sum() >= 8 and projective_change <= .01)
    if record["accepted"]:
        record["reason"] = "empirical same-camera registration accepted only inside validated spatial cells and feature hull"
    else:
        failures = []
        if median > 1.:
            failures.append(f"held-out median {median:.2f} px exceeds 1.00 px")
        if p90 > 3.:
            failures.append(f"held-out p90 {p90:.2f} px exceeds 3.00 px")
        if coverage < .5:
            failures.append(f"feature hull covers {coverage:.1%}, below 50%")
        if support.sum() < 8:
            failures.append(f"only {int(support.sum())}/16 spatial cells are supported, below 8")
        if projective_change > .01:
            failures.append(f"projective variation {projective_change:.1%} exceeds 1%")
        record["reason"] = "Display registration rejected: " + "; ".join(failures)
    return record


def estimate_display_registration(discovery: Any) -> dict[str, Any]:
    """Estimate DISPLAY <-> its metadata-designated same-side stereo camera."""
    import cv2

    spatial = discovery.spatial_photo or {}
    side = str(spatial.get("monoscopic_image_location", "")).lower()
    record: dict[str, Any] = {"accepted": False, "source_sha256": discovery.source_sha256,
        "reference_role": side, "reason": "monoscopic camera location is unknown",
        "method": "mutual SIFT ratio matches at unique positions; same-side MAGSAC homography; held-out 4x4 spatial validation",
        "raw_assets_modified": False, "calibration_recovered": False,
        "coordinate_policy": "native decoded pixel centers; EXIF orientation is not separately applied",
        "precision_scope": "approximate RGB registration; camera-axis depth transport assumes nearly identical optical axes",
        "depth_resampling": "bilinear only with finite positive contributors and <=2% footprint depth spread; unsupported is NaN",
        "right_reprojection": "nearest pixel forward projection with closest-depth z-buffer; holes are retained"}
    if side not in {"left", "right"}:
        return record
    display, reference = _asset(discovery, "display"), _asset(discovery, f"spatial_{side}")
    left, right = _asset(discovery, "spatial_left"), _asset(discovery, "spatial_right")
    d, s = np.asarray(display.array), np.asarray(reference.array)
    record.update({"display_shape": list(d.shape[:2]), "stereo_shape": list(s.shape[:2]),
                   "display_image_index": display.parent_image_index,
                   "reference_image_index": reference.parent_image_index,
                   "display_array_sha256": hashlib.sha256(d.tobytes()).hexdigest(),
                   "reference_array_sha256": hashlib.sha256(s.tobytes()).hexdigest()})
    if d.dtype != np.uint8 or s.dtype != np.uint8 or d.ndim != 3 or s.ndim != 3:
        record["reason"] = "feature registration currently requires separately decoded uint8 RGB views"
        return record
    # Match at a common feature-detection scale; the fitted coordinates still
    # refer to each original grid, including the half-pixel resize offset.
    detector_width = min(2688, s.shape[1])
    detector_height = round(s.shape[0] * detector_width / s.shape[1])
    detector = cv2.SIFT_create(nfeatures=12000)
    points, descriptors = [], []
    for image in (d, s):
        resized = cv2.resize(image[..., :3], (detector_width, detector_height), interpolation=cv2.INTER_AREA)
        keypoints, descriptor = detector.detectAndCompute(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY), None)
        scale_x, scale_y = detector_width / image.shape[1], detector_height / image.shape[0]
        points.append(np.array([[(k.pt[0] + .5) / scale_x - .5,
                                 (k.pt[1] + .5) / scale_y - .5] for k in keypoints]))
        descriptors.append(descriptor)
    if any(desc is None or len(desc) < 2 for desc in descriptors):
        record["reason"] = "no usable feature descriptors"
        return record
    matcher = cv2.BFMatcher()
    forward = matcher.knnMatch(descriptors[0], descriptors[1], k=2)
    reverse = matcher.knnMatch(descriptors[1], descriptors[0], k=2)
    mutual = {(m.queryIdx, m.trainIdx) for pair in reverse if len(pair) == 2
              for m, n in [pair] if m.distance < .65 * n.distance}
    matches = [m for pair in forward if len(pair) == 2 for m, n in [pair]
               if m.distance < .65 * n.distance and (m.trainIdx, m.queryIdx) in mutual]
    record.update(_fit_matches(np.array([points[0][m.queryIdx] for m in matches]),
                               np.array([points[1][m.trainIdx] for m in matches]), d.shape[:2], s.shape[:2]))
    if record["accepted"] and side == "right":
        from .spatial import register_stereo_rows
        _, _, alignment = register_stereo_rows(left.array, right.array)
        record["stereo_row_registration"] = alignment
    return record


def _check_record(discovery: Any, registration: dict[str, Any]) -> None:
    if not registration.get("accepted"):
        raise RegistrationError(str(registration.get("reason", "registration was not accepted")))
    if registration.get("source_sha256") != discovery.source_sha256:
        raise RegistrationError("registration belongs to a different source file")


def _spatial_support(x: np.ndarray, y: np.ndarray, registration: dict[str, Any]) -> np.ndarray:
    height, width = registration["stereo_shape"]
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x <= width - 1) & (y <= height - 1)
    # Every contributing stereo pixel must have spatial evidence. In particular,
    # an interpolation footprint must not borrow a value across a rejected cell.
    support = np.asarray(registration["supported_cells"], dtype=bool)
    safe_x, safe_y = np.where(valid, x, 0), np.where(valid, y, 0)
    for xx in (np.floor(safe_x), np.ceil(safe_x)):
        for yy in (np.floor(safe_y), np.ceil(safe_y)):
            cell_x = np.clip(np.floor(xx * 4 / width).astype(np.intp), 0, 3)
            cell_y = np.clip(np.floor(yy * 4 / height).astype(np.intp), 0, 3)
            valid &= support[cell_y, cell_x]
            # Accepted cells can include a corner outside the fitted feature
            # hull. Do not extrapolate a depth label into that unobserved area.
            if "stereo_inlier_hull" in registration:
                hull = np.asarray(registration["stereo_inlier_hull"], dtype=np.float64)
                orientation = np.sum(hull[:, 0] * np.roll(hull[:, 1], -1)
                                     - hull[:, 1] * np.roll(hull[:, 0], -1))
                inside = np.ones(x.shape, dtype=bool)
                for start, end in zip(hull, np.roll(hull, -1, axis=0)):
                    cross = (end[0] - start[0]) * (yy - start[1]) - (end[1] - start[1]) * (xx - start[0])
                    inside &= cross >= -1e-7 if orientation >= 0 else cross <= 1e-7
                valid &= inside
    return valid


def _sample_depth(depth: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Conservative interpolation excludes NaN and depth discontinuities."""
    height, width = depth.shape
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x <= width - 1) & (y <= height - 1)
    xx, yy = np.where(valid, x, 0), np.where(valid, y, 0)
    x0, y0 = np.floor(xx).astype(np.intp), np.floor(yy).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = xx - x0, yy - y0
    weights = np.stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy))
    values = np.stack((depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1]))
    active = weights > 0
    valid &= np.all(~active | (np.isfinite(values) & (values > 0)), axis=0)
    minimum = np.min(np.where(active, values, np.inf), axis=0)
    maximum = np.max(np.where(active, values, -np.inf), axis=0)
    valid &= maximum - minimum <= .02 * minimum
    sampled = np.sum(np.where(active & np.isfinite(values), values, 0) * weights, axis=0)
    return np.where(valid, sampled, np.nan).astype(np.float32), valid


def register_display_depth(depth_display: np.ndarray, discovery: Any,
                           registration: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Transport a display teacher plane to its same-side stereo grid."""
    _check_record(discovery, registration)
    depth = np.asarray(depth_display)
    if depth.shape != tuple(registration["display_shape"]):
        raise RegistrationError("teacher depth does not match the display reference dimensions")
    height, width = registration["stereo_shape"]
    output = np.full((height, width), np.nan, dtype=np.float32)
    supported = np.zeros((height, width), dtype=bool)
    inverse = np.asarray(registration["stereo_to_display_homography"], dtype=np.float64)
    for start in range(0, height, 128):
        y, x = np.mgrid[start:min(start + 128, height), :width]
        mapped = _project(np.column_stack((x.ravel(), y.ravel())), inverse).reshape(*x.shape, 2)
        values, valid = _sample_depth(depth, mapped[..., 0], mapped[..., 1])
        valid &= _spatial_support(x, y, registration)
        output[start:start + len(x)] = np.where(valid, values, np.nan)
        supported[start:start + len(x)] = valid
    return output, supported


def _left_to_right_depth(depth: np.ndarray, spatial: dict[str, Any],
                         row_registration: dict[str, Any]) -> np.ndarray:
    """Forward-project left Z to native right pixels; retain occlusion holes."""
    height, width = depth.shape
    y, x = np.indices(depth.shape, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    flow = spatial["principal_point_delta_x_pixels"] - (spatial["focal_length_pixels_for_depth"]
                                                        * spatial["baseline_meters"] / np.where(valid, depth, 1.))
    target_x = x + flow
    target_y = y
    if row_registration.get("applied"):
        affine = np.asarray(row_registration["right_to_aligned_affine"])
        target_y = (y - affine[1, 0] * target_x - affine[1, 2]) / affine[1, 1]
    valid &= (target_x >= 0) & (target_x <= width - 1) & (target_y >= 0) & (target_y <= height - 1)
    ix = np.rint(np.where(valid, target_x, 0)).astype(np.intp)
    iy = np.rint(np.where(valid, target_y, 0)).astype(np.intp)
    result = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(result, (iy * width + ix)[valid], depth[valid])
    return np.where(np.isfinite(result), result, np.nan).reshape(height, width)


def project_left_depth_to_display(left_depth: np.ndarray, discovery: Any,
                                  registration: dict[str, Any]) -> np.ndarray:
    """Create an explicitly resampled display-grid RAFT metric depth estimate."""
    _check_record(discovery, registration)
    depth = np.asarray(left_depth, dtype=np.float32)
    if depth.shape != tuple(registration["stereo_shape"]):
        raise RegistrationError("left depth does not match the calibrated stereo dimensions")
    if registration["reference_role"] == "right":
        spatial = discovery.spatial_photo or {}
        if not spatial.get("rectified_stereo_ready"):
            raise RegistrationError("cross-camera projection requires validated rectified stereo calibration")
        depth = _left_to_right_depth(depth, spatial, registration.get("stereo_row_registration", {}))
    height, width = registration["display_shape"]
    output = np.full((height, width), np.nan, dtype=np.float32)
    matrix = np.asarray(registration["display_to_stereo_homography"], dtype=np.float64)
    for start in range(0, height, 128):
        y, x = np.mgrid[start:min(start + 128, height), :width]
        mapped = _project(np.column_stack((x.ravel(), y.ravel())), matrix).reshape(*x.shape, 2)
        values, valid = _sample_depth(depth, mapped[..., 0], mapped[..., 1])
        valid &= _spatial_support(mapped[..., 0], mapped[..., 1], registration)
        output[start:start + len(x)] = np.where(valid, values, np.nan)
    return output

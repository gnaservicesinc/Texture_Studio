from types import SimpleNamespace

import numpy as np
import pytest

from ipde.registration import (
    RegistrationError, _fit_matches, _left_to_right_depth, _sample_depth,
    estimate_display_registration, project_left_depth_to_display, register_display_depth,
)


def record(shape=(8, 8), side="left"):
    return {"accepted": True, "source_sha256": "fixture", "reference_role": side,
            "display_shape": list(shape), "stereo_shape": list(shape),
            "display_to_stereo_homography": np.eye(3).tolist(),
            "stereo_to_display_homography": np.eye(3).tolist(),
            "supported_cells": np.ones((4, 4), dtype=bool).tolist()}


def test_identity_depth_transport_preserves_values_and_nan():
    discovery = SimpleNamespace(source_sha256="fixture")
    depth = np.arange(64, dtype=np.float32).reshape(8, 8) + 1
    depth[3, 4] = np.nan
    original = depth.copy()
    result, valid = register_display_depth(depth, discovery, record())
    np.testing.assert_array_equal(result, depth)
    np.testing.assert_array_equal(valid, np.isfinite(depth))
    np.testing.assert_array_equal(depth, original)
    np.testing.assert_array_equal(project_left_depth_to_display(depth, discovery, record()), depth)


def test_interpolation_never_blends_foreground_and_background_or_invalid_pixels():
    depth = np.array([[1., 5.], [1., 5.]], dtype=np.float32)
    result, valid = _sample_depth(depth, np.array([.5, 0., 1.]), np.array([.5, 0., 1.]))
    assert np.isnan(result[0]) and not valid[0]
    np.testing.assert_array_equal(result[1:], [1., 5.])
    depth[0, 0] = np.nan
    result, valid = _sample_depth(depth, np.array([.5, 1.]), np.array([0., 0.]))
    assert np.isnan(result[0]) and not valid[0]
    assert result[1] == 5. and valid[1]  # zero-weight NaN does not invalidate an exact sample


def test_supported_cells_exclude_unvalidated_footprint():
    r = record()
    r["supported_cells"][1][1] = False
    discovery = SimpleNamespace(source_sha256="fixture")
    depth = np.ones((8, 8), dtype=np.float32)
    result, mask = register_display_depth(depth, discovery, r)
    assert np.isnan(result[2:4, 2:4]).all()
    assert not mask[2:4, 2:4].any()
    assert np.isfinite(result[0, 0])
    # Display sampling straddles a valid and rejected cell even though the
    # metric depth values agree; it must remain unsupported.
    r["display_to_stereo_homography"][0][2] = .5
    projected = project_left_depth_to_display(depth, discovery, r)
    assert np.isnan(projected[2, 1])


def test_cross_camera_projection_has_correct_sign_and_nearest_surface_wins():
    depth = np.full((2, 8), np.nan, dtype=np.float32)
    depth[0, 2], depth[0, 3] = 2., 1.
    spatial = {"principal_point_delta_x_pixels": 0., "focal_length_pixels_for_depth": 2.,
               "baseline_meters": 1.}
    result = _left_to_right_depth(depth, spatial, {})
    assert result[0, 1] == 1.
    assert np.count_nonzero(np.isfinite(result)) == 1
    # A nonzero principal-point delta participates in the correspondence.
    spatial["principal_point_delta_x_pixels"] = 1.
    assert _left_to_right_depth(depth, spatial, {})[0, 2] == 1.


def test_right_reprojection_undoes_vertical_alignment():
    depth = np.full((4, 8), np.nan, dtype=np.float32)
    depth[2, 4] = 1.
    spatial = {"principal_point_delta_x_pixels": 0., "focal_length_pixels_for_depth": 1.,
               "baseline_meters": 1.}
    alignment = {"applied": True, "right_to_aligned_affine": [[1., 0., 0.], [0., 1., 1.]]}
    assert _left_to_right_depth(depth, spatial, alignment)[1, 3] == 1.


def test_rejected_wrong_source_and_wrong_grid_are_refused():
    discovery = SimpleNamespace(source_sha256="fixture")
    r = record()
    r["accepted"] = False
    r["reason"] = "poor validation"
    with pytest.raises(RegistrationError, match="poor validation"):
        register_display_depth(np.ones((8, 8)), discovery, r)
    r = record()
    with pytest.raises(RegistrationError, match="different source"):
        register_display_depth(np.ones((8, 8)), SimpleNamespace(source_sha256="other"), r)
    with pytest.raises(RegistrationError, match="reference dimensions"):
        register_display_depth(np.ones((4, 4)), discovery, r)
    with pytest.raises(RegistrationError, match="calibrated stereo dimensions"):
        project_left_depth_to_display(np.ones((4, 4)), discovery, r)
    r = record(side="right")
    with pytest.raises(RegistrationError, match="validated rectified"):
        project_left_depth_to_display(np.ones((8, 8)), SimpleNamespace(source_sha256="fixture", spatial_photo={}), r)


def test_unknown_monoscopic_location_cannot_guess_reference_camera():
    result = estimate_display_registration(SimpleNamespace(source_sha256="fixture", spatial_photo={}))
    assert not result["accepted"]
    assert "unknown" in result["reason"]


def test_spatially_distributed_heldout_fit_accepts_known_transform():
    rng = np.random.default_rng(3)
    stereo = rng.uniform([0, 0], [799, 599], (1024, 2))
    display = (stereo - [3., -2.]) * 2.
    result = _fit_matches(display, stereo, (1200, 1600), (600, 800))
    assert result["accepted"]
    assert result["heldout_count"] == 256
    assert result["heldout_median_error_pixels"] < .001
    np.testing.assert_allclose(result["display_to_stereo_homography"],
                               [[.5, 0., 3.], [0., .5, -2.], [0., 0., 1.]], atol=1e-5)


def test_global_fit_rejects_depth_dependent_parallax_and_local_bad_cells():
    rng = np.random.default_rng(7)
    display = rng.uniform([0, 0], [1599, 1199], (1024, 2))
    stereo = display / 2.
    stereo[:, 0] += rng.uniform(0., 25., len(stereo))
    result = _fit_matches(display, stereo, (1200, 1600), (600, 800))
    assert not result["accepted"]
    assert result["heldout_median_error_pixels"] > 1.
    assert any(not cell["supported"] for cell in result["spatial_validation_cells"])

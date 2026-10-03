# Classical spatial depth: IMG_1515 surface interiors

The supplied preview's alpha mask matches the pre-fix classical validity mask
exactly. The missing wall and door interiors were discarded by IPDE, rather
than being encoded as black depth in the source photo. The original HEIC and
supplied previews were not modified.

## Causes and corrections

1. **Local support was used as the depth image.** SGBM aggregates costs along
   paths and estimates disparity in weakly textured interiors using surrounding
   evidence and regularization. IPDE required horizontal texture and correlation
   in every 9×9 patch, destroying these interiors. The subsequent component
   filter then operated on the disconnected remnants. Geometry components now
   use visibility and independent reverse consistency first. Local photometric
   support and its component filter produce a separate mask and explicitly
   selected supported-depth product. Preview alpha describes a finite estimate.
2. **Absolute camera brightness biased correspondence costs.** Comparing RGB
   codes from different cameras can confuse exposure offsets with spatial shifts.
   On a synthetic plane with a brightness ramp, an 8-pixel shift and a 30-code
   exposure offset, the previous cost selected 63-pixel disparities, even with
   reverse consistency. A fixed 31×31 per-channel local mean subtraction on
   inference copies recovers the 8-pixel shift in over 98% of the test interior.
   It uses a fixed 128-code offset, nearest-even rounding and uint8 clipping;
   there is no per-image range stretch. Source arrays remain untouched.
3. **The classical support export was only a finite-value mask.** It now exports
   the actual local support returned by the matcher. Missing support cannot be
   silently replaced by a finite-value mask. `stereo-depth` provides the regularized
   metric estimate separately from `stereo-supported-depth`.

The SGBM parameters, 1/16-pixel disparity representation, calibration, native
left-view grid, vertical registration and independent reverse tolerance remain
unchanged. Positive geometry, original-image visibility, both reverse neighbors
within one pixel, and geometric component checks still gate classical estimates.
No depth hole filling, smoothing, percentile clipping or resizing was added.
RAFT inference is unchanged.

The [OpenCV StereoSGBM reference](https://docs.opencv.org/4.13.0/d2/d85/classcv_1_1StereoSGBM.html)
describes its block costs, smoothness penalties and built-in validity checks.
The separate local support policy is IPDE's own heuristic, not an OpenCV
requirement or a calibrated probability.

## Measurements

All three photos were freshly decoded with pillow-heif. Stereo grids are
2688×2016. Input hashes remain identical before and after inference.

| Photo | Previous finite output | New finite estimate | New local support |
| --- | ---: | ---: | ---: |
| IMG_1515 | 22.0871% | 62.2573% | 21.8685% |
| IMG_0835 | 34.8742% | 60.6577% | 34.7234% |
| IMG_0845 | 74.9873% | 90.5182% | 74.9078% |

For IMG_1515, merely removing the texture gate exposes distances as large as
66.44 meters. The corrected cost and geometric component checks produce a
0.290693–4.181503 meter range. This range is inferred, not surveyed ground truth.
The new estimate includes 2,188,671 pixels without strict local support. About
37.7% of the image still has no estimate; ambiguous texture, occlusions and
matching failures remain visible as transparency/NaN. Coverage is not accuracy.

Independent SIFT comparisons request 12,000 features, use a 0.6 descriptor ratio,
positive horizontal disparity and a vertical residual below 0.5 pixels after
registration. On common feature locations, absolute disparity errors are:

| Photo | Common locations | Median before → after | p90 before → after | p99 before → after |
| --- | ---: | ---: | ---: | ---: |
| IMG_1515 | 294 | 0.215 → 0.207 px | 0.582 → 0.529 px | 1.169 → 1.039 px |
| IMG_0835 | 3,880 | 0.162 → 0.162 px | 0.414 → 0.413 px | 0.864 → 0.865 px |
| IMG_0845 | 4,427 | 0.161 → 0.161 px | 0.386 → 0.386 px | 0.645 → 0.625 px |

The 32 newly retained feature locations in IMG_1515 have median/p90/p99 errors
of 0.492/1.452/2.270 pixels. This checks textured feature correspondences, not
the depth of every restored smooth surface. Use the support product when that
distinction matters.

## Verification and outputs

- 75 tests and six parameter subtests pass. New regressions cover uniform plane
  interiors, exposure-induced false parallax, separate support/preview/metric
  exports, exact data preservation, selective export and collision preflight.
  Existing reverse-disagreement, off-image and disconnected-outlier tests pass.
- A synthetic unequal-detail plane retains sub-0.2-pixel error at the 99th
  percentile; all retained estimates are within one pixel. The broader estimate
  includes samples that the previous local mask discarded.
- All six IMG_1515 EXR/PNG outputs passed exact writer readback and independent
  checks of disparity, calibrated meters, masked meters, support, displacement,
  and preview gray/alpha. Hashes are recorded in the verification JSON.
- Qt 6.12.0 build and native Cocoa launch smoke test pass. Bundled Python modules
  match source, and the bundled inventory exposes the new metric-depth product.
  Interactive GUI click-through was not performed.

Source SHA-256:
`bad3f18486194de14cdb5fde9c1e4765a26e14ba41bab85d30278cc1693ba1dc`.

Outputs: `out/IMG_1515-fixed-20261003/`.
Comparison and detailed checks: `out/IMG_1515-audit-20261003/comparison.png` and
`out/IMG_1515-audit-20261003/verification.json`.
The audit directory also retains a read-only baseline module and `verify.py`.

To reproduce in a new output directory:

```sh
env PYTHONPATH=src .venv/bin/python -m ipde /path/to/IMG_1515.HEIC \
  --output-dir /path/to/new-output \
  --select stereo-height --select stereo-depth --select stereo-preview \
  --select stereo-support --select stereo-supported-depth \
  --select stereo-displacement --no-npy --manifest
env PYTHONPATH=src .venv/bin/python -m pytest -q
make build
build/IPDE.app/Contents/MacOS/IPDE --smoke-test
```

The manifest in this validation export is explicitly requested. Normal exports
still omit manifests unless selected by the user.

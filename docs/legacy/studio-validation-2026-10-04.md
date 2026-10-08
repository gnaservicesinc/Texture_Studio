# Studio validation — 4 October 2026

Validated locally on Apple silicon with Python 3.14, the project virtual
environment, and the installed Qt 6.12.0. `make build` produces IPDE Studio
with Extractor, Dataset Studio, RAFT Studio, Photo Studio and Raw Studio inside
`build/IPDE Studio.app/Contents/Applications`.

## Crashes and project sessions

The supplied crash reached `QMainWindow::statusBar()` from a socket disconnect
callback after window destruction. Session callbacks now belong to a live Qt
context and run after the socket signal completes. Shutdown clears handlers
before socket teardown. Process callbacks are disconnected before their owning
window starts destruction.

The native CTest regression covers connected exit, destroyed windows, cancelled
queued callbacks, reentrant deletion, rejected/fragmented registration and
session-lock lifetime. Packaged smoke checks authenticate all five subapps
against a Unix-socket hub, buffer a coalesced startup change event, exit cleanly
and release their role locks. These GUI checks use Qt's offscreen platform.

## Data preservation

- Real portrait `IMG_6678.HEIC`: all nine decoded assets retained exact sample
  bits, including native float16 depth, uint16 auxiliary data and people mattes.
  Cutout RGB samples remained identical; only the alpha channel was composed.
- Privacy copies of that portrait and spatial `IMG_0835.HEIC` and
  `IMG_1515.HEIC`: compressed image payloads, decoded arrays, effective
  orientation and functional stereo/depth calibration were retained. Identifying
  metadata and ICC descriptive/device fields were removed. Original versus
  scrubbed ICC profiles produced bit-identical RGB conversion through LCMS.
- ProRAW `IMG_0864.DNG`: the stored 10-bit JPEG XL LinearRaw raster decoded as
  uint16 without clipping code 1024. A 96 × 64 crop matched an independent TIFF
  SubIFD decode exactly. TIFF/native calibration was retained in companion JSON.
- Local DepthPro inference on that RAW photograph ran on MPS and saved native
  float32 depth separately from source-grid depth and the matching untouched
  float32 camera RGB crop. Camera RGB retained negative and above-one values;
  clipping applied only to the model's separate input copy.
- Numeric regressions cover NaN bits, source dtype, selected auxiliary exports,
  transactional rollback, foreign camera/grid rejection, matte union, ICC
  retention, original-file export without a decoder, and oriented Vision input.

## Limits confirmed by measurement

None of the three checked Apple HEIC fixtures supports the current native
embedded-repair path with exact preservation. With unchanged disparity supplied,
ImageIO changed about 6.6–10.9% of retained display RGB samples, also changed
depth/matte/gain-map samples, and dropped either opaque auxiliary planes or
spatial grouping/calibration. Photo Studio disables repair for those sources.
The failed candidates were isolated and deleted. Privacy export instead uses
the verified metadata-only container path and preserves the compressed imagery.
Main color, opaque auxiliary and encoded-depth replacement remains unavailable.

Native click-through and a real GIMP clipboard paste could not be checked:
the computer-use service failed to start. Layout was inspected from offscreen
screenshots. Clipboard code retains exact PNG bytes and exported file URLs.

## Checks

Python validation passed 131 archived HEAD baseline tests with 18 subtests,
and 29 current `verification/` regressions with 23 subtests. The native CTest
and all five packaged session smoke checks passed. The baseline ran from a
temporary directory; the user's deleted `tests/` files were left deleted.
Run current targeted regressions with:

```sh
env PYTHONPATH=src .venv/bin/python -m pytest -q verification
ctest --test-dir build --output-on-failure
.venv/bin/python verification/smoke_subapps.py
```

No source photographs were modified. No changes were committed or pushed.

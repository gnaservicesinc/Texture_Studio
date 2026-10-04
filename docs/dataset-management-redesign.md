# Dataset management redesign — 4 October 2026

Dataset Studio now separates sidebar selection, photo/teacher membership edits,
and automatic training-set preparation. **Photos and depth** supports bulk
removal/restoration, linked-component split changes, per-dataset session drafts,
new version names, new-photo generation and importing an existing dataset's
targets. **Training split** controls automatic collection splitting.

## Backend validation

The current Python verification suite passed **98 tests**, including **15 new
dataset-edit regressions**:

```sh
PYTHONPATH=/opt/ipde/ipde/src .venv/bin/python -m unittest discover -s verification -v
```

The edit checks cover removals and additions; stable base IDs and namespaced
added IDs; exact file/array hashes, dtype, byte order, signed zero and NaN
payloads; raw auxiliary planes and metadata; source-directory removal after
publication; and existing NPY/NPZ containers with identical numerical values but
different file hashes. No array normalization, gamma correction or inference
occurs when editing existing entries.

Split regressions verify transitive photo/teacher/burst/scene components,
preserved authored groups after removal and merging independently edited
subsets, category/scene namespace separation, conflicting overrides and
contradictory imported splits. Review exposes the same management-component IDs
used when saving. Failure checks cover corrupted arrays, changed manifests,
invalid selections, existing/raced-in destinations, parallel transfer rollback,
KeyboardInterrupt and CLI SIGTERM cancellation cleanup. Sources stay unchanged.

`git diff --check` passed for the implementation at this validation point.

## Application validation

The rebuilt native applications passed all six Qt regression tests, the
embedded edit CLI/review checks and all five packaged subapp session smoke
checks. These automated checks include dataset selection and management flows.
Qt screenshots were rendered for visual inspection. Interactive native click-through
could not run because the computer-use service reported `Sky Computer Use native
pipe startup failed`; the automated results do not establish that every visible
control has been manually exercised.

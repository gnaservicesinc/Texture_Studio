# Runtime and lifecycle audit — 2026-10-16

The review covered Texture Studio's recipe, adapter, render-cache and export
paths; Factory's model-operation ownership, run deletion and comparison picker;
and Workshop's folder polling, settings and interrupted-run recovery. The three
applications have independent repositories and receive separate commits.

## Fixes

### Texture Studio

- Opening a recipe previously changed the shared LoRA selection before reading
  its photo, and could leave a partially changed selection after an error. It
  now loads the inputs first and validates the complete adapter selection before
  committing shared state. Conflicting adapters are rejected together.
- A successful recipe load now clears the previous photo's review and export
  link, preventing stale review provenance in subsequent exports.
- Cached exports now stage every map and the updated metadata before publishing
  files. Both precision paths preserve existing destination files and clean up
  failed exports. Fresh exports reject collisions before expensive rendering.
- Inspector updates no longer rehash both bundled photo-enhancer payloads on
  every redraw. Availability is cached for the two immutable bundled products;
  explicit model loads and inference retain checksum validation.
- Temporary Core ML compilations are reclaimed. Inference compilations remain
  available until their cached models are released on the service's actor.
  Caller-provided compiled models are preserved.

### Loom Factory

- A matching process ID no longer allows unrelated project actors to overlap
  model operations. Queues lend their lease through an explicit task-local,
  single-use permit; concurrent borrowers and expired inherited tasks are denied.
- Run deletion holds an exclusive project lease through validation and every
  Trash move, including when another application or queue owns the project.
- Model comparisons and classifier categorization now use the same operation
  lease as training and saved-model testing.
- Comparison options index each model's sample manifests and ancestry once,
  replacing repeated full-catalog parsing for every candidate image. Membership
  remains based on whole materials and all recorded ancestors.
- Classifier validation and categorization reclaim their temporary compiler
  output on success, cancellation and failure.

### Loom Workshop

- Unchanged folder polls preserve existing image-metadata bytes and filesystem
  change times instead of rewriting identical tags.
- Idle catalog inspection reuses the validated installed-model catalog while
  its complete filesystem inventory is unchanged. Nanosecond change times,
  modification times, identity, mode and size invalidate the cache. Packages
  containing symlinks use full inspection. Training and publication retain full
  checksum validation.
- Failed options writes preserve the last persisted settings in memory.
- A completed result record without a published adapter no longer strands valid
  checkpoints after interrupted publication.
- Empty or repeated category model entries are rejected before map counting can
  trap on duplicate dictionary keys.

## Verification

- Texture Studio: full Debug Xcode suite, 112 tests executed, four optional
  integration skips, zero failures.
- Factory: 33 focused tests, zero failures or skips. SwiftPM also compiled the
  application, queue executable and native targets.
- Workshop: 37 XCTest cases, two optional integration skips, zero failures;
  13 Swift Testing UI, rendering and termination cases passed.
- Native build-tool, installer, three-application staging, version and diff
  checks passed. Installer tests used temporary fixtures.

Regression coverage includes multi-adapter conflicts and corruption, missing
recipe inputs, stale photo provenance, export-copy and metadata failures, lease
contention and lifetime, checkpoint publication recovery, failed settings saves,
unchanged metadata, cache invalidation with restored modification times, and
symlinked package contents.

The CPU height-cleanup experiment was reverted because its measurements did not
show a repeatable improvement. The retained performance changes remove repeated
work; no end-to-end speedup or learned-quality improvement is claimed.

The installed applications and the active Factory session were preserved. This
audit did not run GPU training, endurance tests, optional real-model integration,
or interactive installed-app acceptance. Local logs are retained under
`out/code-audit-2026-10-16/`; that generated evidence directory is not versioned.

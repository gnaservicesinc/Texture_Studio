# Native migration validation — 2026-10-09

Texture Studio's current source, Xcode build, Makefile, CI, installer and Release bundle have no Python runtime dependency. Retired extraction, training, research and compatibility implementations were deleted. Swift ignore rules cover generated Xcode products and local data; no virtual environment was created.

## Native implementation

- ImageIO exports every exposed auxiliary type at every image index. Raw buffers retain padding, encoded orientation, data descriptions and metadata. Scalar NPY exports remove row padding without changing sample bits. ImageIO cannot inventory arbitrary private HEIF items.
- Swift performs dataset registration, exact integer PNG decoding/cropping, checksum checks, durable edits, preparation and owned-file cleanup. Original source photos and maps remain unchanged.
- Metal and Core Image process materials; Accelerate computes numeric statistics. MPSGraph implements the pinned PBRnxt operation sequence, Float32 inference, LoRA gradients, clipping and Adam updates. GPU compilation, inference and training run outside the main thread.
- Safetensors packages retain exact recorded bases, trained factors, configuration and license notices. Native code reads the pinned tensor archive as data without executing model sources. Invalid factor layouts fail before GPU compilation.
- URLSession handles account, catalog, download and upload APIs. Tokens use Apple Keychain. Transfers verify immutable revisions, sizes, checksums and complete package inventories.

## Final checks

| Check | Result |
| --- | --- |
| Full arm64 Xcode unit suite | 247 passed, 1 optional original-photo test skipped, 0 failures |
| Native build, version and transactional installer checks | Passed |
| Optimized Release build and four embedded material tools | Passed |
| Recursive bundle audit and strict signature verification | Passed; no interpreter, environment, executable model source or bundled weights |
| Release native material smoke | Passed; Float16 and Float32 EXR outputs decoded and verified |
| Installed suite in `/Applications` | Matches the validated Release bundle byte-for-byte; strict signatures and installed native smoke passed |
| Release auxiliary export of `IMG_6678.HEIC` | Six raw ImageIO buffers and scalar NPY payloads compared byte-for-byte; original SHA-256 unchanged |
| Published base tensor descriptors | All 2,892 consumed tensor names, shapes and dtypes matched the pinned archive metadata |
| Actual native trainer | Synthetic complete-operation fixture performed updates, validation, checkpoint requests, stop-and-save and package verification |

The final test result is `build/NativeMigration/Logs/Test/Test-TextureStudio-2026.10.09_08-26-53--0400.xcresult`. Verification logs are under `out/native-migration-*` and `out/native-auxiliary-release-verification.log`.

The supplied 08:04 hang report showed GPU graph work in synchronous tests on the main thread. Those tests and production work now use background tasks. A separate debugger investigation identified macOS Quit events terminating an asynchronous test host; test hosts no longer activate the app on launch. The final complete suite passed without host restarts.

## Cleanup and delivery

Removed the verified, inactive Texture Studio managed DA3 environment and its registry (about 1.1 GB), archived Python source snapshots, pytest cache, superseded DerivedData and old generated test results. Original datasets, photographs, model data and the system Python installation were preserved. No model weights were downloaded.

The native Release suite is about 57 MB. The package is `dist/Texture-Studio-macos-arm64.zip`, with a neighboring SHA-256 file. Its SHA-256 is `45f7228a294867d46073a7b9bd0b13d08a05d20b590f745ea271c27359751c8d`. The validated suite was installed at `/Applications/Texture Studio.app`, including its four material tools, after confirming the previous suite was closed. The installed bundle matches the built bundle and contains no retired Python backend. Git commit and remote verification are recorded separately during the handoff.

Full pretrained-weight visual parity, production-grid memory usage and end-to-end speed remain unmeasured because the 349 MB base weights are absent. Synthetic graph and trainer checks establish implementation behavior, not production image quality or a general speedup claim.

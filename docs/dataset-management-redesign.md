# Dataset management redesign — 4 October 2026

Dataset Studio saves photo membership and split edits directly to the selected
dataset. Remove, restore and split actions update the manifest automatically;
existing image, depth, mask and auxiliary array files stay untouched. Removed
entries remain visible and restorable after reopening. Adding raw HEICs records
their paths immediately so unfinished depth generation can be resumed.

Pending changes are journaled per dataset before autosave starts. A save failure
retains the latest edits and additions for retry. Opening a fresh Studio window
recovers them. Closing with unfinished saves offers **Save and close**, **Close
keeping recoverable edits**, or **Cancel**. A recovery-file write failure disables
the recoverable-close choice. An unreadable existing journal is preserved in a
timestamped backup before replacement.

The journal also records the complete operation before launching a save. If
Studio exits after the manifest commit but before acknowledging it, reopening
replays that operation safely. Its ID, original manifest hash, request digest
and committed metadata digest identify the completed write. Already committed
operations are acknowledged without rewriting the manifest; newer edits then
save against the verified result. Unrelated external edits still require reload.

**Training split** accepts 0%–100% validation. **Apply split and open selected
dataset in Trainer** applies that percentage to the same dataset and waits for
the manifest save before opening Trainer. The percentage selects independent
capture components; finite group counts can make the actual photo proportion
differ. At 100%, no included training groups remain and Trainer reports that
training cannot start. At 0%, validation groups are missing. Combining datasets
into a separate training set remains optional.

## Backend verification

The focused dataset-edit checks cover manifest-only removal and restoration,
atomic replacement, stale manifest
conflicts, failed/cancelled writes, exact existing array bytes, same-filesystem
additions, duplicate-addition handling, committed-operation recovery, and linked photo/teacher/burst/scene split
boundaries. Review and training readers honor excluded records, expose them for
restoration, and retain a captured manifest during an already-running training
job. Existing precision checks include dtype, byte order, NaN payloads, signed
zero, raw auxiliary planes and provenance.

The complete verification suite passed **129 tests and 53 subtests**. All six
native Qt checks and the packaged subapp/backend smoke checks also passed.
The bundled executables and changed Python modules match the verified build.

On temporary copies containing only the manifests of the user's 171-entry and
196-entry datasets, in-place remove/restore/split API calls took **231–409 ms**;
CLI calls including Python startup took **367–522 ms**. Operation recovery was
enabled for every update. These measurements used
metadata copies with no array payload files and left the real datasets
untouched. They measure metadata update latency, not teacher inference or model
training.

## Qt verification

The Dataset Studio regression exercises actual second-row clicks, bulk dataset
role buttons, checkbox-free controls, group split edits, simultaneous newer
edits during a save, failed-save retry, durable drafts in a fresh window, close
choices, and unreadable-journal preservation.
It also interrupts a real committed save before its GUI callback, reopens Studio,
and verifies both the completed operation and any newer edit recover without
extra commits or changed array bytes.

Its end-to-end fixture generates a valid scientific dataset through the Python
backend and uses the real GUI autosave process. It removes a target, reopens a
fresh window to verify the saved exclusion, then applies actual 5% and 100%
validation settings before opening Trainer on the same dataset. The 100% result
has no training entries and is marked untrainable. Every scientific array file
is checked byte-for-byte after membership and split updates. Percentage and seed
changes also autosave without opening Trainer and restore their saved values in
a fresh window.

Run the native checks with:

```sh
ctest --test-dir build --output-on-failure
```

The automated checks exercise native Qt event handling under the offscreen
platform. They do not establish that every visible control has been manually
clicked on the development macOS desktop.

# Project working principles

Resolve recoverable prerequisites instead of stopping at the first error. When
the application has a clear, authorized path to obtain a missing public model
or other required resource, implement that path: locate or download it, verify
its identity and format, show progress, support cancellation, and continue the
requested operation. Do not make the user perform routine setup the application
can complete itself. Laziness is not a valid reason to throw an error.

Report an error when the available recovery paths have failed or cannot fulfill
the request, and explain the actual remaining obstacle. Never disguise failure
with fabricated output, substitute incompatible model weights, or discard user
data to make the operation appear successful.

Application outputs and model downloads belong in user storage, outside the
source repository. Preserve users' original images. User-authorized maintenance
of this project's downloadable development corpus may modify or remove those
corpus files when correcting an issue.

Preserve completed, task-related work remotely: commit and push each changed
repository to its configured origin before finishing, including the independent
LoomFactory and LoomWorkshop repositories when they are part of the task. Verify
the remote branch points to the local commit. Do not rely on a local checkout
surviving between runs, include unrelated changes, or force-push shared history.
Report a failed push explicitly. This is standing user authorization for these
task-scoped commits and pushes unless the user later requests otherwise.

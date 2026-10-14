# Native Loom training and installation — 2026-10-13

Loom Factory and Loom Workshop 0.2.0 build 2 replace private Python training
workers with Swift/Metal/MPSGraph, native ImageIO preparation, Safetensors I/O,
Core ML export and native Hub publishing. Their Release bundles declare
`LoomTrainingBackend = swift-mpsgraph-metal` and reject interpreter/worker
resources during staging. The suite installer and its tests use Swift.

Completed source revisions:

- Loom Factory: `206cc4dd180cf25fd4ed8731ef1a447dc44bd492`.
- Loom Workshop: `b5340c2e73547163f8c84dd33bf5129d27afe412`.

Both configured remote `main` refs matched those commits. Workshop pins its
native library to the Factory revision and resolved/built it from GitHub; a
sibling checkout is needed only when explicitly using the development override.

## Execution evidence

Factory: 99 checks, 2 optional fixture skips, no failures. This includes actual
1024-pixel Bark training, analytic full-size masked gradients, independent
optimizer equations, native/Core ML parity, source preservation, whole-material
validation, cancellation and continuation. Workshop: 41 checks, 2 optional
fixture skips, no failures, plus four native workflow tests passed using the
pinned remote dependency. LoRA training freezes the exact base and exports real
low-rank tensors plus the fused native graph. Native build/installer tests passed.

Explicit contiguous GPU result storage fixes nondeterministic gradients observed
with implicit MPSGraph result buffers. Repeated full-size runs, reloaded weights
and NaN-prefilled output buffers produced identical gradients and final weights;
the precise internal MPSGraph cause remains unproven.

First-use real PNG pair preparation for exact 1024-pixel crops on this M2 Max:

| Original grid | Previous Python | Native Swift |
| --- | ---: | ---: |
| 1024 | 0.689 s | 0.069 s |
| 2048 | 2.785 s | 0.190 s |
| 4096 | 11.629 s | 0.642 s |

GPU compute remains around 2.7 seconds per update for the width-164, four-block,
1.94-million-parameter 1024 model. The final real-source cancellation probe
requested Stop at 6.5 seconds and returned at 8.789 seconds, preserving two
completed updates and unchanged source hashes. Python's already-cached NumPy
crop alone was faster than native crop conversion; these first-use results
measure decoding/preparation, not an overall training or prediction-quality claim.

Raw reports and test logs are in the user's storage at
`~/Library/Application Support/Loom Factory/Validation/native-training-2026-10-13`.
Factory also commits its measurement summary under `Docs/Validation`.

## Installed state

`make install` built, installed and launched all three applications as Release.
The installed binaries matched their built counterparts byte-for-byte; strict
deep signatures and native-only bundle guards passed. No Python training process
was running after launch. Installed Texture Studio smoke passed.

Installed executable SHA-256 values:

- Loom Factory: `893b2a7b326f91e4eb1bbcfc4ae9b41f777c453fb4e424a67f3b3366c5354ce7`.
- Loom Workshop: `0e3eec76f8b1e3ba9568058c9d9688097a5855b21268de0b0656366638ec0aee`.
- Texture Studio: `68e6d7831afcb2814c16054f048100263ee2e27adc04b8619e7c0ef1868e47c0`.

Texture Studio was built as the shell without a Texture Loom base package.
Workshop production adaptation still requires the exact installed base; its
native training/export tests used a valid disposable base fixture. Original user
images, project settings and existing saved weights were preserved. Old private
Python environments in source projects are unused and retained in place.

The native computer-use service did not start, so no live button click is claimed.
Stop behavior is covered through the actual project actor and worker, with
separate offscreen UI checks and installed executable/bundle verification.

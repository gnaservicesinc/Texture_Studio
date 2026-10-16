# Real capture data for phone photo preparation

Survey and actual acquisition: 2026-10-16. The missing data is photographs of
**the same thing at different quality**, not more PBR materials. Research used
Hugging Face discovery/metadata plus the original authors' sites. A Hub license
label does not override the original dataset's terms.

## Recommended sources

| Source | What it teaches | Decision |
| --- | --- | --- |
| [SIDD-CC on HF](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC), based on [original SIDD](https://abdokamel.github.io/sidd/) | Real phone noise with matched clean references; repeated camera/ISO/light conditions | First restoration sample acquired, including carpet. Original source declares MIT; retain both original and reprocessing attribution. |
| [RealBlur, author HF](https://huggingface.co/datasets/rimchang/RealBlur), [author repository](https://github.com/rimchang/RealBlur) | Real blur/sharp pairs, including a telephoto variant | First blur candidate. Author declares CC BY 4.0. Large archive acquisition deferred; metadata checked. The same repo contains synthetic BSD-B and pretrained models, which must not be mistaken for real capture pairs. |
| [NIND, author dataset page](https://commons.wikimedia.org/wiki/Natural_Image_Noise_Dataset), [author code](https://github.com/trougnouf/nind-denoise) | Identical scenes photographed at varying ISO | Next source for more surface diversity. Check each image's attribution/license and actual alignment. Not acquired in this pass; no verified HF distribution was selected. |
| [MIT Multi-Illumination](https://projects.csail.mit.edu/illumination/), [author SDK](https://github.com/lmurmann/multi_illumination) | Fixed-camera photographs under 25 lighting conditions | Two real scenes acquired as a lighting diagnostic. Original data declared CC-BY. A changed light direction is not a neutral diffuse target. |

SIDD-CC is pinned at `5eb2ab6375d7f99f36b59bb5aacb46af18409412`.
Its sRGB train release has 230 full-resolution noisy/reference pairs from 115
capture instances. It re-estimates reference images from DNG bursts; the target
is not an arbitrary low-ISO photo. Our inventory found only ten physical scene
numbers, so scene grouping is essential. Camera/ISO variants of one scene are
not independent surfaces. See the [pinned card](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/blob/5eb2ab6375d7f99f36b59bb5aacb46af18409412/README.md).

RealBlur metadata is pinned at `afdb28d4aeb7739bc326b181ded390d4e2ac21a3`.
The main RealBlur archive is about 13.1 GB and the telephoto archive about 3.58 GB.
Do not fetch the entire HF repository as a first experiment. Sample the real
blur/sharp subset with known scene/capture identity and inspect alignment before
using reconstruction loss. Still photography references do not automatically
remove material lighting.

## What was actually inspected

Data lives outside `/opt` at:

`~/Library/Application Support/Loom Factory/PhotoDiffuseResearch-2026-10-16/`

| Local review | Actual contents | Interpretation |
| --- | --- | --- |
| `real-phone-pairs/` | Four paired captures, physical scenes 001 and 002, ISO 100/1600, 126.621 MiB of original PNGs | Generic noise/color diagnostics: ColorChecker and marker scenes. Not category coverage. |
| `scene-reference-survey/` | All ten physical scene references, with eight additional original downloads and two reused references | Reviewed the corpus composition instead of relying on a dataset name. `ten-scene-survey.jpg` is a resized review sheet only. |
| `carpet-phone-pairs/` | Two paired captures of patterned carpet, physical scene 008, iPhone ISO 100/1600, 73.874 MiB of original PNGs | Useful native woven-detail/noise example. One carpet surface does not establish broad carpet support. |
| `real-lighting-captures/` | Two upstream mip1 scene archives, 46.07 MiB; five selected light directions extracted per scene | Real lighting variation. Archives contain the full 25 directions; ten selected JPEGs are byte-identical extracts. |
| `diagnostic-split.json` | Six SIDD captures, three scene groups, verified original hashes and native crop pixels | Train scene 001, validation 002, test 008. A tiny staging diagnostic, explicitly not a qualified training dataset. |

The carpet captures are `0184-001` and `0191-135`. Each noisy frame is paired
with its own supplied clean reference, never with an unrelated frame from a
lower-ISO capture. Native 1024 crops are not resized: decoded pixels are checked
against the recorded source crop and retain its ICC profile. Source PNG bytes
are unchanged. The side-by-side review has the noisy input on the left and the
clean dataset reference on the right. **It is not our model's prediction.** Both
retain scene lighting; neither is certified flat diffuse.

The repeated-light scenes are `main_experiment120` (hair/wig on a table) and
`elm_revis_kitchen14` (a kitchen counter with glare and shadow). They are broad
lighting examples, not material category training endorsements. Each selected
JPEG is 3000×2000 at the author's provided mip1 level, itself a derivative of
higher-resolution capture. We did not resize those downloads. The contact sheet
is a display derivative. Hold out rooms/buildings where other viewpoints share
surfaces; scene identifiers alone may be insufficient for that larger dataset.

## Attractive sources with limitations

* [ISTD HF mirror](https://huggingface.co/datasets/Donghyun99/ISTD): real shadow
  pairs are relevant, but the [original author's terms](https://github.com/DeepInsight-PCALab/ST-CGAN#license)
  restrict use to research/noncommercial purposes. The mirror's MIT label does
  not establish permission for product training. Not downloaded.
* [MIDIntrinsics](https://github.com/compphoto/MIDIntrinsics#license): derived
  albedo targets are CC BY-NC-SA 4.0, research-only, with additional methodology
  IP information. These terms differ from the underlying MIT repeated-light
  photographs. Derived albedos were not downloaded or assumed commercially usable.
* [LOL-v2 real mirror](https://huggingface.co/datasets/okhater/lolv2-real): low-light
  enhancement is relevant, but normal exposure is not flat material color. Verify
  upstream rights and paired alignment before admission; small images limit
  native 1K fine-detail testing. Not acquired.
* [SRD mirror](https://huggingface.co/datasets/Donghyun99/SRD): potentially useful
  shadow pairs; original redistribution/training terms remain unverified here.
* [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/): artist retouches
  teach photographic preferences, not necessarily removal of surface lighting;
  research terms need attention. Lower priority.
* [DPED](https://huggingface.co/datasets/sumathiselvan/DPED-streaming): phone/DSLR
  correspondence can teach enhancement, but camera differences/parallax require
  alignment checks. It is not evidence of a pixel-exact neutral-color target.
* [OpenIllumination](https://huggingface.co/datasets/OpenIllumination/OpenIllumination)
  and [M2AD](https://huggingface.co/datasets/ChengYuQi99/M2AD): varying views/lights
  of objects are useful adjacent research. Fixed-view pairing and domain fit
  matter; they do not directly fill our phone-surface cleanup dataset.

No inspected dataset established a ready, commercially cleared, broad collection
of phone surface photos paired with artist-ready diffuse outputs for all board
categories. No suitable repeated-quality alpha rope/net collection was verified.
These are gaps to address with a targeted capture/editing session later, rather
than assuming another PBR download solves them.

## Next experiment

Use the acquired real pairs for an initial frozen RGB restoration A/B, including
clean-reference identity controls, when the user's GPU training is free. Compare
native fibers/grain and whole-image appearance. A pretrained SIDD restoration
model may already have seen the sampled scenes, including carpet. Our diagnostic
split does not remove its pretraining exposure; do not present that comparison
as independent generalization. Expand physical surface diversity
before training or making category claims. Handle lighting as a separate stage
and allow controlled detail generation as described in the [product plan](../PhotoToDiffusePlan.md).

Data tools are CPU-only development utilities. They do not ship a Python runtime,
change material tags/selections, or start a training job. Receipts record hashes,
scene groups, unreviewed status and target limitations. Re-run their checks before
any import; manual approval and a native RGB product contract are still required.

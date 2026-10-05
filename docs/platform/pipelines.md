# Pipelines

A pipeline is a chain of steps run on each clip: detect, track, measure against a reference. It doesn't
name a clip; you choose the clips when you run it. The [steps and templates](#steps) are listed below.

<!-- SCREENSHOT: the pipeline builder -->

## Build one

**Pipelines → New**, or start from a template (Foraging trips, Flower / ROI visitation, Pollen assay,
Interactions, Individual bee IDs). Drag steps from the palette and connect them.

## Run it

- **On clips:** in **Processing**, select clips (or all that match the filter), choose the pipeline and
  **Run pipeline**. Each clip becomes a run, and together they form a batch.
- **On a schedule:** on the device page, under **Scheduled processing**, run a pipeline on each day's new
  clips.
- **From code:** see [API](api.md).

Runs that use the GPU (Detect, Track, Identify species) use your account's credits.

## See the output

**Runs** lists every run and batch. A run shows each step's output: the events and interactions tables,
counts, and **CSV** downloads. A batch page combines its clips (tracks, events, interactions, GPU time)
and has **Download batch data** for all of them. The columns are documented in
[Results](results.md).

When clips fail, the batch page groups them by cause. You can rerun only the failed clips or the whole
batch. A rerun is a new batch, and the old one is kept for comparison.

## Steps

| Step | What it does |
|---|---|
| **Video** | The clip being analysed. Every pipeline starts here. |
| **Detect** | Finds one class in each frame with YOLO (fast) or SAM 3 (text prompt, slower). Can analyse every frame (needed for tracking) or a sample of frames (for objects that don't move). |
| **MOT: Track objects** | Links detections into tracks (BeeTrack). Runs inside the same GPU pass as Detect. |
| **Reference: Saved layout** | The device's saved ROI and nest tubes, or regions you draw. No GPU needed. |
| **Events** | One row per enter or exit of a reference. |
| **Interactions** | One row per episode of contact: insect with reference (a visit), insect with insect, or both. |
| **Detection count** | Distinct objects, most common count per frame, totals, per frame, or binned over time. |
| **Identify species** | After tracking, classifies every saved crop of each track and gives the track the species with the most votes. Model: BeeMachine (354 bee taxa) or BioCLIP (limited to species recorded near the device). |
| **Read bee marker** | The same vote, over every crop, for each track's paint mark, to tell individuals apart. |

### Templates

Each template is the same few steps put together in a different way:

| Template | Steps |
|---|---|
| **Foraging trips** | Video → Detect (bee) → Track → Reference (device layout) → Events |
| **Flower / ROI visitation** | Video → Detect (bee) → Track → Reference (drawn regions) → Interactions (insect ↔ reference) |
| **Pollen assay** | Video → Detect (bee) → Track → Reference (one drawn region per pollen tube) → Interactions (insect ↔ reference) |
| **Interactions** | Video → Detect (bee) → Track; Detect (nest) as the reference → Interactions (all) |
| **Individual bee IDs** | Video → Detect (bee) → Track → Read bee marker |

Copy a template and change it: swap the class ("wasp", "fly"), use your own model, add Identify species,
or add a second Detect step for a reference that moves.

### Settings worth knowing

- **Gap tolerance (frames)**: how long a track can disappear (occlusion, a missed detection) and still
  count as the same episode. Default 15.
- **Event confidence**: how sure the entry/exit classifier must be before an event counts (default 0.6).
- **Insect ↔ insect radius**: how close two insects must be, as a percentage of the frame width, to count
  as interacting.

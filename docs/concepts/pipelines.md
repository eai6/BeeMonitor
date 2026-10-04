# Pipelines and steps

A **pipeline** is a reusable recipe of steps, drawn as a graph in the pipeline builder. It does not name a
clip; you choose the clips when you run it (see [Pipelines](../platform/pipelines.md)).

## The steps

| Step | What it does |
|---|---|
| **Video** | The clip being analysed. Every pipeline starts here. |
| **Detect** | Finds one class in each frame with YOLO (fast) or SAM 3 (text prompt, slower). Can analyse every frame (needed for tracking) or a sample of frames (for objects that don't move). |
| **MOT: Track objects** | Links detections into tracks (BeeTrack). Runs inside the same GPU pass as Detect. |
| **Reference: Saved layout** | The device's saved ROI and nest tubes, or regions you draw. No GPU needed. |
| **Events** | One row per enter or exit of a reference. |
| **Interactions** | One row per episode of contact: insect with reference (a visit), insect with insect, or both. |
| **Detection count** | Distinct objects, most common count per frame, totals, per frame, or binned over time. |
| **Identify species** | Names each track's species (BeeMachine, 354 bee taxa) by majority vote over the track's frames. |
| **Read bee marker** | Reads each track's paint or QR mark to tell individuals apart. Also works on videos analysed earlier. |

## Templates

Each template is the same few steps put together in a different way:

| Template | Steps |
|---|---|
| **Foraging trips** | Video → Detect (bee) → Track → Reference (device layout) → Events |
| **Flower / ROI visitation** | Video → Detect (bee) → Track → Reference (drawn regions) → Interactions (insect ↔ reference) |
| **Colony activity** | Video → Detect (bee) → Track → Detection count (over time) |
| **Interactions** | Video → Detect (bee) → Track; Detect (nest) as the reference → Interactions (all) |
| **Individual bee IDs** | Video → Detect (bee) → Track → Read bee marker |

Copy a template and change it: swap the class ("wasp", "fly"), use your own model, add Identify species,
or add a second Detect step for a reference that moves.

## Settings worth knowing

- **Gap tolerance (frames)**: how long a track can disappear (occlusion, a missed detection) and still
  count as the same episode. Default 15.
- **Event confidence**: how sure the entry/exit classifier must be before an event counts (default 0.6).
- **Insect ↔ insect radius**: how close two insects must be, as a percentage of the frame width, to count
  as interacting.

## Lessons

Each template has a short lesson on the platform covering the question, how the pipeline answers it, and
what to look at in the results. Open them at **Pipelines → Lessons**
([beemonitor.edwardamoah.com/pipelines/lessons/](https://beemonitor.edwardamoah.com/pipelines/lessons/)).
They work well for teaching.

**Next:** [Events & interactions](results.md)

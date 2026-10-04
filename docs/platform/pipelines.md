# Pipelines

A pipeline is a chain of steps run on each clip: detect bees, track them, then count what you care about.

<!-- SCREENSHOT: the pipeline builder -->

## Build one

**Pipelines → New**, or start from a template. Drag steps from the palette and connect them:

| Step | Does |
|---|---|
| Video | The clip being analysed |
| Detect objects | Finds bees (YOLO, or SAM 3 with a text prompt) |
| Track | Follows each bee between frames |
| Reference layout | The device's nest tubes |
| Visitation / Events | Counts nest entries and exits |
| Foraging trips | Pairs exits with returns |
| Count detections | Counts per frame, distinct, or over time |
| Identify species | Species of each tracked bee |

## Run it

From **Processing**, select clips (or all that match the filter) and **Run pipeline**. Or schedule it on the
device page under **Scheduled processing** to run on new clips every day.

## Results

A run shows each step's output — tables, counts, the annotated video — and **CSV** downloads. A batch run
combines its clips.

# Pipelines

A pipeline is a chain of steps run on each clip: detect, track, measure against a reference. The steps and
templates are explained in [Pipelines and steps](../concepts/pipelines.md). This page covers using them.

<!-- SCREENSHOT: the pipeline builder -->

## Build one

**Pipelines → New**, or start from a template (Foraging trips, Flower / ROI visitation, Colony activity,
Interactions, Individual bee IDs). Drag steps from the palette and connect them.

## Run it

- **On clips:** in **Processing**, select clips (or all that match the filter), choose the pipeline and
  **Run pipeline**. Each clip becomes a run, and together they form a batch.
- **On a schedule:** on the device page, under **Scheduled processing**, run a pipeline on each day's new
  clips.
- **From code:** see [API](api.md).

Runs that use the GPU (Detect, Track, Identify species) use your account's credits.

## Results

**Runs** lists every run and batch. A run shows each step's output: the events and interactions tables,
counts, and **CSV** downloads. A batch page combines its clips (tracks, events, interactions, GPU time)
and has **Download batch data** for all of them. The columns are documented in
[Events & interactions](../concepts/results.md).

When clips fail, the batch page groups them by cause. You can rerun only the failed clips or the whole
batch. A rerun is a new batch, and the old one is kept for comparison.

# 35 — Collapse the analyze layer to two primitives

**Status:** design + plan. Supersedes the per-analyzer results work in
[32] and [33] at the *presentation* layer; the aggregation fixes from
[34] all carry forward unchanged.

## The problem

The analyze layer has four blocks that are really four *questions*, each with
its own output shape, its own `table_kind`, its own aggregator, and its own
panel on the batch page:

| block | output | what it actually computes |
|---|---|---|
| Foraging Trips | `events` | pairs Exit→Entry on the GPU's nest events |
| Visitation Count | `table` | contiguous frame runs a track spends in a reference |
| Interactions | `table` | proximity episodes between two tracks, or track↔nest |
| Detection Count | `table` | how many things were in frame |

The first three are three views of two underlying facts, and the code says so
if you read it side by side:

- `ops.compute_visitation` finds *contiguous runs of frames a track spends
  inside one reference* — that is an episode of contact with a reference.
- `InteractionAnalyzer.analyze_reference_interactions` finds *contiguous runs
  of frames a track spends within N pixels of a nest* — the same thing, computed
  on the GPU against detected nests instead of the user's ROI, and written to a
  different CSV with different column names.
- The GPU's `events.csv` records Entry/Exit against nests — which are exactly
  the **boundaries** of those same episodes, produced by a third code path.

So one physical fact (a bee was at tube 3 from frame 400 to frame 700) is
computed three times, stored three ways, and rendered by three aggregators. Any
new question means a fourth block, a fourth aggregator, a fourth panel. The
results interface can't be simple while the layer under it isn't.

## The abstraction

Two primitives. Everything the ecologist asks is a read over them.

**Event — a boundary crossing.** Something entered or exited something.

```
frame, time, subject, subject_kind, action, target, target_kind, source
                                    enter|exit
```

**Interaction — an episode of proximity or contact.** Two things were together
for a span of time.

```
start_frame, end_frame, start_time, end_time, duration_sec,
a, a_kind, b, b_kind, relation, source
                      organism|reference    proximity|contact|inside
```

`source` on both is `gpu` or `derived`, following the provenance rule from
[34]: a number that came from an assumption must say so.

**The two are one pass.** An interaction's start and end *are* the enter and
exit events. Compute episodes once; emit both tables from them. That is the
whole simplification — not a rename, a deletion of two of the three code paths.

### What the ecologist's questions become

| question | read over |
|---|---|
| Foraging trips | events: pair `exit(nest N)` → next `enter(nest N)` |
| Visitation count | interactions where `b_kind = reference`, grouped by `b` |
| Dwell time | same rows, sum `duration_sec` |
| Insect ↔ insect | interactions where `b_kind = organism` |
| Time at each tube | interactions grouped by `b`, binned by `start_time` |

None of those needs a pipeline block. They are filters and group-bys on two
CSVs — which is exactly what a downstream user with a spreadsheet or an R
script wants anyway.

## Plan

**Phase 1 — the primitives (backend)**
1. `ops.compute_episodes(tidy, refs, fps, gap_frames)` — refactor of
   `compute_visitation`, returning episode records rather than a visit tally.
   The existing gap/reference-change logic is already correct and is kept
   verbatim; only the return shape changes.
2. `ops.episodes_to_events()` and `ops.episodes_to_interactions()` — the two
   projections.
3. `ops.load_events()` — normalise the GPU's `events.csv` into the same Event
   schema (it is schema-drifted, like the tracking CSV; reuse the `_pick`
   pattern).
4. `ops.load_interactions()` — normalise the GPU's `interactions.csv` into the
   Interaction schema. `summarize_interactions` already picks these columns
   tolerantly; that mapping moves here.

**Phase 2 — the blocks**
5. New `analyze.events`; `analyze.interaction` → `analyze.interactions`, now
   fed by local ROI episodes as well as the GPU CSV (today it only reads the
   GPU CSV and reports zero when there isn't one).
6. Retire `analyze.foraging_trips` and `analyze.visitation` as blocks. Existing
   pipelines that reference them keep running — the executor maps them onto the
   new primitives and returns the same summary shape, so no saved pipeline
   breaks and no historical run re-renders differently.
7. Keep `analyze.detection_count` — "how much was there" is a different axis
   from "who went where", and the Biodiversity Count pipeline depends on it.
   Keep `analyze.colony_activity` untouched.

**Phase 3 — the results interface**
8. Two tables, and a preset selector over them rather than four panels. Rollups
   (trips, visits per reference) become named presets that set a filter and a
   group-by, so adding a question is adding a preset, not a block.
9. `batch_rows` currently shows Tracks / Events / Trips for every batch
   whatever it ran — the Biodiversity Count batch renders three columns of
   zeros. Columns follow the primitives the batch actually produced.

**Phase 4 — regression floor**
10. A fixture where one bee visits two tubes: the episodes, the events derived
    from them, and the trips derived from those must all agree on the same
    clip — the invariant that three code paths could never be held to.

## Also removing (asked for directly)

- **"Annotated video" config field** on `detect.objects` and `track.mot`. It
  roughly doubles runtime, is silently unavailable for chunked runs (the merge
  cannot produce one and says so), and the CSVs never needed it.
- **The editor's "▶ Run" button.** It runs a pipeline with no clip selected;
  running happens on the Processing page, where clips are chosen. Its endpoint
  (`pipelines:run` → `run_pipeline`) has no other caller and becomes dead —
  flagged rather than deleted, since removing a URL is wider than what was
  asked.

## Carried forward from [34]

Every fix from the consistency audit survives this refactor and several get
*easier*:

- One `fps_with_source` resolver — episodes convert frames to seconds in one
  place now instead of three.
- Coverage denominators attach to the two primitives rather than to four
  aggregators.
- `same_track` on trips stays exactly as it is: trips are still paired on the
  events table, so the confirmed/inferred split is unchanged.
- Cross-clip identity is still not attempted, and the disclosure stays.

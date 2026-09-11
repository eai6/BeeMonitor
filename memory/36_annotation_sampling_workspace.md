# 36 — Choosing a representative sample for annotation

**Status:** audit complete, plan for execution.

## The problem

Picking which clips to annotate is the decision that determines what the model
learns, and it is currently made through the worst interface on the platform.

`annotations/detail.html` § "Add videos" is a **checkbox list of titles inside a
`max-h-60` box** — about five rows visible at a time — capped at 500 rows with
"showing 500, narrow further" when there are more. The rows carry a title, a
device name and a date. No still, no preview, no way to see what a clip
contains without leaving the page.

So the user is choosing a training sample from a library of thousands by
reading filenames through a five-row window. Two consequences:

1. **You cannot see what you are picking.** A clip's title says nothing about
   whether it shows a bee on a flower, an empty hotel, or a wasp.
2. **You cannot see what you have picked.** Nothing says how the selection is
   distributed across devices or across the day — which is exactly what makes a
   sample representative or biased. A set drawn entirely from one hotel at noon
   trains a model that works at noon on that hotel.

Meanwhile `analysis/processing.html` already solves (1) properly: a filter rail
with per-device dots and time-of-day windows, a day-grouped grid of stills,
hover preview, an inline viewer. It was built for exactly this task — scanning
footage to decide what to run — and annotation needs the same thing.

## Two failures worth not repeating

The annotation view **re-implements the filters by hand** (`views.py:337-351`:
its own `title__icontains`, `device_id`, `site_name`, year/month/day,
`bee_confirmed`) instead of calling `analysis.views.apply_video_filters`, which
exists and is already shared between the hub and the pipeline runner. So the
two pages already disagree: the hub understands multi-device selection, hour
windows (`hfrom`/`hto`), date ranges and `analysis=never`; the annotation page
understands none of them.

That is the same disease as the frame-rate resolver and the three episode code
paths — one decision implemented twice, drifting. **The fix is extraction, not
a second copy.**

## Plan

**Phase 1 — extract the workspace** (no behaviour change)
1. `apps/videos/workspace.py`: `filter_options(user)`, `device_rows(user, selected)`,
   `group_by_day(videos, dots)` — the context the rail and grid need, lifted
   out of `ProcessingHubView` verbatim.
2. `apply_video_filters` moves there too, re-exported from
   `analysis.views` so existing imports keep working.
3. `videos/templates/videos/_review_rail.html` and `_review_grid.html`:
   the rail and the day-grouped grid, parameterised by what the host page does
   with a selection (run a pipeline / add to a project).
4. `processing.html` switches to the includes. Its tests must pass untouched —
   that is the proof the extraction was faithful.

**Phase 2 — annotation uses it**
5. "Add videos" becomes the workspace: stills, day grouping, hover preview,
   inline viewer, the full filter set. Selection posts to `add_videos`.
6. Clips already in the project stay visible and marked, rather than being
   excluded — "I already have 12 from this hotel" is the thing you need to
   know while choosing, and hiding them is what makes over-sampling invisible.

**Phase 3 — the part that is actually about representativeness**
7. **Coverage map**: a device x hour-of-day grid, cells showing
   *in project / available*. This is the instrument the current page lacks
   entirely — it makes a biased sample visible at a glance, and it is cheap
   (one `values().annotate()` over the library).
8. **Stratified draft**: "take N per device per hour bucket" pre-selects a
   balanced set the user then adjusts. A starting point, not an oracle —
   the user still looks at the clips.
9. Both read from the same filtered queryset, so the map always describes what
   is on screen.

**Phase 4 — training, which is downstream of all this**
10. `training/new.html` picks a project and classes and says nothing about what
    is *in* the project. Show the dataset's composition — frames per class, per
    device, per hour — on the same page, so an unrepresentative training set is
    visible before the GPU spend rather than after the model underperforms.
11. This is the cheap half of "optimise the training pages"; anything more
    (comparing runs, per-class metrics over time) is a separate piece of work
    and not planned here.

## Carried forward

- One implementation per decision. Phase 1 exists because the filters are
  already duplicated and already diverging.
- Show the denominator. Coverage on the batch page made partial results
  legible; the coverage map does the same for a training sample.
- Don't hide inconvenient rows. Excluding already-added clips is the same
  mistake as dropping references with zero visits: the absence is the finding.

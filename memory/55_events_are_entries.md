# 55 — Events are going in and coming out; one count everywhere

**Status:** built 2026-10-22.

## Why
- A bee crossing a 60-tube hotel from the front passed over five tube boxes on the way to the one it went into, and the events table listed 12 rows (enter + exit for every box touched, one-frame touches included). Busy clips ran to hundreds.
- The clip's results page recounted on every load (against today's layout, without the analyzer's label filter) while the batch table read the run's saved count: 453 vs 47 for the same clip.

## Rules (apps/pipelines/ops.py `compute_nest_events`)
- **Enter** = a track ENDS on a reference (the insect went in). **Exit** = a track STARTS on one (it came out).
- Not an event: under 0.2 s on the reference; a new track starting within 1 s and 5% of the frame (tracker lost it); a track cut by a *chosen* window's start/end (part of a clip). A whole clip's own ends are real: motion-triggered clips start/stop because a bee came out/went in.
- Passing over a reference is an **interaction** (with duration). Interactions now default to a 0.5 s minimum when the node doesn't set one; a saved 0 still counts everything.
- The worker's own nest Entry/Exit rows (`source=gpu`) are unchanged.

## One count
- The results page reads the run's saved rows (same as batch table, CSVs, studies); it recomputes only for a job no run spawned.
- "Recount with the current layout" on the results page (owner or superuser): re-resolves the reference steps, reruns events/interactions from the saved tracks. No GPU.
- `PipelineRun.counting_rules` (old runs 1, new 2): the reconcile loop recounts finished runs below `recount.CURRENT_RULES`, 20 per pass / 60 s budget, newest first, each once. Bump `CURRENT_RULES` (and the model default) when the rules change again. The sweep applies the 0.5 s interactions minimum to runs whose node saved 0.

**Feedback:**

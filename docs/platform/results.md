# Results

Tracking produces two tables, and every study is answered from them. BeeMonitor doesn't have a separate
"foraging" or "flower" analysis. Time on a flower is a sum over interactions, and a foraging trip is a pair of
events. You derive what your study needs.

| Table | One row is | Answers |
|---|---|---|
| **Events** | Something entered or exited something, at a moment | Arrivals, departures, foraging trips |
| **Interactions** | Two things were together for a span of time | Visits, time on a flower or in a zone, encounters |

Both come from one pass over the tracks, so they always agree: an interaction's start and end *are* its enter
and exit events.

## Events

| Column | Meaning |
|---|---|
| `frame` | Frame number of the crossing |
| `time_sec` | Seconds from the start of the clip |
| `subject` | The track id of what crossed (for example a bee) |
| `subject_kind` | `organism` |
| `action` | `enter` or `exit` |
| `target` | What it crossed into or out of (a nest tube, a flower, a zone) |
| `target_kind` | `reference` |
| `source` | `gpu` (the entry/exit classifier) or `derived` (computed from the tracks and the reference) |

## Interactions

| Column | Meaning |
|---|---|
| `start_frame`, `end_frame` | First and last frame of the episode |
| `start_sec`, `end_sec` | The same in seconds |
| `duration_sec` | How long it lasted |
| `a`, `a_kind` | The first party: a track id, `organism` |
| `b`, `b_kind` | The second party: another track (`organism`) or a reference (`reference`) |
| `relation` | `inside` (a's centre was within b's shape) or `proximity` (within the distance threshold) |
| `min_distance` | Closest approach, for proximity episodes |
| `source` | `gpu` or `derived` |

The columns are a fixed contract. New columns may be added at the end, but existing ones are never renamed or
reordered, so your scripts keep working.

## From the tables to your question

The setup decides what the *reference* is: a nest tube, a flower, a zone in an arena. The analysis is
always the same read.

| Question | Read |
|---|---|
| Foraging trips at a nest | Events: each `exit` from a tube paired with the next `enter` into it; trip length = the time between them |
| Visits to a flower | Interactions with `b_kind = reference`, counted per `b` |
| Time on a flower (dwell) | The same rows, summing `duration_sec` per `b` |
| Pollen assay: which tube bees prefer | Interactions per pollen tube (`b`): count the rows for the number of interactions, sum `duration_sec` for the total time |
| Insect-to-insect encounters | Interactions with `b_kind = organism` |
| Activity through the day | Bin events or interactions by time, or use **Detection count** |
| Per individual or per species | Join the marker or species label to `subject` / `a` |

```python
import pandas as pd

ev = pd.read_csv("events.csv")
ia = pd.read_csv("interactions.csv")

# Time on each flower (or zone): interactions with a reference
on_ref = ia[ia.b_kind == "reference"]
dwell = on_ref.groupby("b").agg(visits=("a", "size"),
                                 total_sec=("duration_sec", "sum"),
                                 mean_sec=("duration_sec", "mean"))

# Foraging trips: exit from a tube, then the next entry into the same tube
ev = ev.sort_values("time_sec")
ev["next_action"] = ev.groupby("target").action.shift(-1)
ev["next_time"] = ev.groupby("target").time_sec.shift(-1)
trips = ev[(ev.action == "exit") & (ev.next_action == "enter")].assign(
    trip_sec=lambda d: d.next_time - d.time_sec)[["target", "time_sec", "trip_sec"]]
```

The run and batch pages show these tables with summaries, and you can always download them as CSV to
derive your own measures.

## Provenance and limits

- `source` tells you how a row was produced. A `derived` row depends on the reference you supplied and the
  gap tolerance, so if you change either, rerun the pipeline.
- Time columns are empty when a clip's frame rate is unknown. Frame numbers are still exact.
- Track ids identify an animal *within one clip*. To follow an individual across clips, mark the bees and add
  the **Read bee marker** step.

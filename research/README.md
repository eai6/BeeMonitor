# BeeMonitor 2024 field study: paper figures

Code and data to reproduce the main figures of *BeeMonitor: Automated IoT video surveillance and an AI-powered video processing system for monitoring the foraging and nesting behavior of cavity-nesting solitary bees*.

## Run

```bash
pip install -r requirements.txt          # Python 3.11+
jupyter lab beemonitor_figures.ipynb      # then Run All (about 30 s)
```

The notebook is saved with its outputs, so it can be read without running. Figures (PNG and PDF) are written to `output/`.

## Figures

All code is in the notebook: one section per figure, no separate scripts.

| Section | Figure | Output |
|---|---|---|
| 2 | Recorded video by site, day and time of day | `fig_recorded_video` |
| 3 | CV evaluation: precision and recall of events and trips | `fig_cv_evaluation` |
| 4 | Foraging trips and provisioned cells per tube, Site 1 and Site 2 | `fig_tube_maps_site1`, `fig_tube_maps_site2` |
| 5 | Daily foraging trips and land cover | `fig_daily_trips_landcover` |
| 6 | Foraging trips vs provisioned cells (single regression) | `fig_trips_vs_provisioned_cells` |
| 7 | Median trip duration per tube, Mann–Whitney U | `fig_trip_duration` |

## Data

| File | Rows | Contents |
|---|---|---|
| `files.csv` | 6,305 | Every recorded 10-min clip: site, date, start time, duration, processing status |
| `trips.csv` | 21,378 | Every foraging trip detected across the season: site, tube (`reference`), `exit_time`, `entry_time`, `duration_sec` |
| `nesting.csv` | 315 | One row per nest cell at collection: site, round (`release`), tube position (`nest_code`), `provision` |
| `review_events.csv` | 5,710 | Exit/entry events in reviewed clips, with the reviewer's decision (`review`) |
| `review_trips.csv` | 1,302 | Trips in reviewed clips, with the reviewer's decision and `source` (pipeline or reviewer) |

Site codes: `mendels` = Site 1, `natalies` = Site 2.

## Definitions

- **Trip window:** ecological figures use trips that start and end between 08:00 and 20:00 on the same day (`DAY_HOURS`).
- **Rounds:** nests were swapped on 16 May during a recording gap (Site 1: no video 10:50–17:11; Site 2: video only from 18:31). Trips leaving before 14:00 on 16 May count toward Round 1, later trips toward Round 2 (`SWAP_TIME`).
- **Provisioned cell:** a cell containing pollen and/or larvae (`PROVISIONED`). **Brood cell:** contains larvae, alive or dead (`LARVAE`).
- **Tube-round:** one nest tube in one round; the unit for the regression and the trip-duration test.
- **CV scoring, events:** accepted = TP, rejected = FP, added by reviewer = FN. **Trips:** pipeline accepted = TP; rejected, event rejected or re-paired = FP; reviewer-created = FN. Disputed and unreviewed items are excluded; 95% CIs are Wilson intervals.

## Key results reproduced

- 980 h of video recorded: Site 1, 458 h on 50 days; Site 2, 522 h on 46 days (including about 92 h of night video from a scheduler fault, 24–28 Apr).
- Events: precision 0.91 (exit and entry), recall 0.98 (exit) and 0.97 (entry). Trips: precision 0.83 (95% CI 0.80–0.85), recall 0.94 (0.93–0.96).
- Trips vs provisioned cells, both sites and rounds: r = 0.76, R² = 0.58, n = 96 tube-rounds, p < 0.001.
- Median trip duration per tube-round: Site 2 8.18 min (n = 37) vs Site 1 5.96 min (n = 59); U = 750, z = −2.57, p = .010, r = .26.

Tested with Python 3.13, numpy 2.5, pandas 3.0, scipy 1.18, matplotlib 3.11.

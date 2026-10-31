# 65 — Filter tracks by insect group before counting

**Status:** built (2026-10-31), from the user's answers below. Design canvas: https://claude.ai/artifact/F57RzPQVbBWTJtyFKbzyRj (boards "Pipeline editor", "Clip results", "Batch page", "Reviewing: change a track's group", "Groups page"). Requested because ants walk into the nest tubes and are counted as bee events. The user wants to filter tracks by identity before events and foraging analysis, but Identify Species gives many different species names (a cockroach for an ant, odd taxa), so the filter must work on general groups.

> Write under any **Feedback:** line. *Proposed* means a starting point, not a decision.

## Decided by the user (2026-10-31)
- **Tracks with no clear group are kept** (counted). A blurry bee must not be lost.
- **The filter lives in Identify Species.** Its output becomes *tracks*, wired MOT → Identify Species → Events. It filters by group, not by species name.
- **The filter can be turned on or off** on the node. Off still names each track's group but counts every track.
- **The grouping must be adjustable while reviewing.** The user isn't confident in a fixed family → group mapping (bee vs wasp especially). It's all web-side, so a change recounts with no GPU (see §1b).

## What exists (audit, 2026-10-31)
- **Identification runs on the GPU after tracking.**
  - `track_vote.vote_tracks` classifies every saved crop of a track; the track takes the top-1 species with the most votes.
  - The winner goes in the tracking CSV's `taxon`, `taxon_votes`, `taxon_vote_share` and `taxon_crops` columns.
  - Every crop's own reading is uploaded as `track_votes.csv` (`summary_stats.track_votes_csv`, `cloud/wrapper/pipeline.py`).
- **The models:**
  - BeeMachine (`identification/species.py`) knows 354 bee taxa plus `NON_BEE_TAXA` (Coleoptera, Diptera, Lepidoptera, Syrphidae, Wasp). It has **no ant class**.
  - BioCLIP (`identification/bioclip.py`) is constrained to up to 300 research-grade iNaturalist insect species within 50 km (`monitor/priors.py`, `region_taxa`). Common ants are often missing from that list, so it names the nearest thing on it. A clip with no location gets the whole Tree of Life, at species rank. 
  - Species rank on 24 px crops is noisy. "Ant or bee" is a family question (Formicidae vs the bee families), and ants, bees and wasps share an order (Hymenoptera), so order alone can't separate them.
- **The node (`registry.py` `identify.species`):**
  - `input_type` tracks, `output_type` **table**.
  - The executor (`_exec_identify_species`) only reads the CSV into a species table.
  - `_pipeline_species` finds it **downstream** of the tracking step and puts only the model in the GPU job config, so its other settings don't change the cache key.
- **Analyzers load the table in `executors._analysis_inputs`.** They take `inputs["tracks"]["result"]`, `load_tracking_df`, then `filter_by_label`. Events, interactions, trips (period.py reads events), the period review and the batch metrics all follow from that one place.
- **`filter.taxon` exists only as a hidden legacy passthrough** (`_exec_filter_passthrough`). It does nothing.
- **Recount (`recount.py`)** re-runs reference steps and `RECOUNTED` analyzers on the saved tracks with no GPU. `identify.species` isn't in it.
- **Config fields** have no multi-select type: only select, number, text, textarea and the model pickers.

## The plan

### 1. Groups from species names (web, no GPU)
- **`apps/pipelines/taxon_groups.py`:**
  - `lineage(name)` returns `{order, family}`.
    - It uses the GBIF backbone match (`api.gbif.org/v1/species/match?name=`) and caches each name in the Django cache.
    - A small static table covers the BeeMachine labels: every genus is a bee; `Wasp`, `Syrphidae`, `Diptera`, `Coleoptera` and `Lepidoptera` map directly.
    - On a failed lookup the name stays unknown, which counts as unclear. An ID is never blocked on it, like `region_taxa`.
  - `group_of(lineage)` maps to these groups:
    - **Bees:** Apidae, Halictidae, Megachilidae, Andrenidae, Colletidae, Melittidae, Stenotritidae. 
    - **Ants:** Formicidae.
    - **Wasps:** any other Hymenoptera.
    - **Flies:** Diptera.
    - **Beetles:** Coleoptera.
    - **Moths & butterflies:** Lepidoptera.
    - **Other crawlers:** Blattodea, Hemiptera, Dermaptera, Orthoptera, Arachnida, any other arthropod.
    - **Unclear:** none of the above, or not found.
- **`track_groups(result)`:**
  - Reads `track_votes.csv` and turns each crop's species into a group.
  - Each track takes the group with the most crops, with its share.
  - **Called** when the share is at least *the threshold* and at least 3 crops were read; otherwise **unclear**.
  - **Fallback:** a run with no `track_votes.csv` uses the track's voted `taxon` as a single vote, and the share then comes from `taxon_vote_share`.
  - So "cockroach A, ant B, ant C, ant D" becomes one group even when no single species wins.

### 1b. Your own group rules, editable while reviewing
- **Data:** a `GroupRule` model:
  - `user`;
  - `kind`: one of `name` (a label exactly as the model gave it), `genus`, `family`, or `track`;
  - `value`: the name, or `job id:track id` for `track`;
  - `group`;
  - `updated_at`.
- **Precedence:** a track rule beats a name rule, then genus, then family, then the built-in default. Rules apply **per crop** before the group vote, so "every *Periplaneta americana* is an ant here" changes every track whose crops said it. A track rule sets that track's group outright.
- **Whose rules:** the run owner's rules apply, so a batch counts the same for everyone viewing it.
- **Where to change a group:**
  - *Reviewing* (the track overlay panel, event review and period review): click a track to see its group and why ("9 ants (4 species), 3 crawlers · most votes *Camponotus pennsylvanicus* · Formicidae"). Then **Change its group**, applied to *this track only*, *every crop read as <name>*, or *the whole family <family>*. **Save and recount.**
  - *Clip results:* the group cell in each species row is the same control.
  - *A Groups page* (Pipelines › Insect groups) lists every name your runs have seen: family · order, default group, counts as (editable), and track count. Filters: "All names" / "Changed by you" and a search box.
- **Recount after a change:**
  - the clip being reviewed is recounted at once;
  - every other run of the owner whose Identify Species output mentions that name, genus or family (the output stores the names its crops gave) is recounted in a background thread, like `LayoutRecountView`.
- **Staleness:** outputs record a `rules_version` (latest `updated_at`). A run counted under older rules shows "Groups changed since this was counted · Recount", in case the background pass missed it.

### 2. Identify Species outputs tracks and filters
- **Registry:**
  - `output_type` becomes `"tracks"`.
  - A new field **Filter by insect group**: On or Off, default On once groups are ticked. Off names groups, counts every track, and hides the checklist.
  - A new field **Count only these groups**, a new `checkboxes` field type in the editor. Nothing ticked means no filter.
  - A new field **Call a group when this share of crops agree**, default 50%.
  - A *proposed* starter template ticks bees and wasps.
- **Executor:**
  - It returns the upstream tracks output (`result`, `label`, …) plus the species table as now, with `group`, `group_share` and `counted` per row.
  - It also returns `kept_groups` and `dropped_tracks`, a list of track ids.
  - `artifact` becomes `"tracks"`; `table_kind` stays `"species"`, so the results panel keeps rendering.
- **`_analysis_inputs`:** when the upstream output carries `dropped_tracks`, drop those track ids from the dataframe before `normalized_tracks`. Every analyzer then follows.
- **Wiring off to the side (old pipelines):** an Identify Species node wired off to the side, as today, filters nothing, and nothing changes for old runs.
- **The editor** lets a `tracks` output feed Events and Interactions.
- **Cost:** `_pipeline_species` still finds the node downstream of MOT. Changing groups or the threshold doesn't change the GPU cache key, so it costs no GPU-hours. Changing the model re-runs tracking, as now.
- **Recount:**
  - Add `identify.species` to the steps recount re-runs, before the analyzers.
  - Run steps in dependency order, not list order (the canvas order can put Events before Identify).

### 3. Show what was dropped
- **Clip results (Identify Species panel):**
  - Group chips ("21 bees · counted", "6 ants · not counted").
  - A row per track: species (most votes), what the crops said by group, group, share, counted.
- **Tracks CSV:** gains `group`, `group_share` and `counted` columns.
- **Track overlay:**
  - Tracks not counted are drawn grey, with their group on the label.
  - The event review and period review show only counted events, because those events no longer exist.
- **Batch page, a new "Insect groups" card:**
  - "41 tracks not counted: 33 ants, 8 other crawlers · 57 events and 12 trips not counted".

### 4. Check it against the reviews
- Event keys are `action|reference|track|frame`. The batch page counts reviewed events whose track is now dropped, and how each was voted: "9 were on tracks not counted: 8 you marked wrong, 1 right (see it)".
- A **With the filter / Without it** toggle on the review metrics. "Without" counts the events of dropped tracks too: they're computed cheaply alongside and kept as `dropped_events` in the output.

### 5. Applying it to existing batches
- **Steps:**
  - edit the pipeline (rewire, tick groups);
  - re-run the batch's clips;
  - optionally name the new batch (memory/64).
- **Effect:**
  - Tracking comes from the StepResult cache, so it costs no GPU-hours.
  - Review votes carry over, because the metrics match votes across runs sharing a GPU job (`shared_job_runs`).

### Not planned now
- **A GPU "insect group" mode for BioCLIP.** It would compare crops against a few group prompts (ant, bee, wasp, cockroach, beetle, fly, spider) instead of species. Do it only if step 1 is still noisy on the reviewed batches; it needs a new image and re-runs.
- **A movement or size cue.** Ants walk on the surface and are smaller in frame; a fallback if the ID can't separate them.

## Build order
1. `taxon_groups.py` (lineage, cache, BeeMachine table, `track_groups`), stubbing GBIF in tests. Tests cover:
   - an ant voted as mixed cockroach and ant species becomes ants;
   - a split vote is unclear;
   - a failed lookup is unclear;
   - a run with no crops CSV falls back to `taxon`.
2. Registry: the output type, the `checkboxes` field type in the editor, and the threshold. The executor returns tracks plus `dropped_tracks`; `_analysis_inputs` drops them. Tests cover:
   - events skip ant tracks;
   - unclear tracks are kept;
   - a side-wired node filters nothing;
   - no change to the GPU cache key.
3. Recount: identify before the analyzers, in dependency order.
4. `GroupRule` model, migration and precedence in `track_groups`. The change endpoint (owner only) recounts the clip now and the affected runs in the background; `rules_version`. Tests cover:
   - a track rule beats a name rule, which beats a family rule;
   - a name rule regroups mixed crops;
   - a non-owner gets 403;
   - affected runs are found by name.
5. Results panel (groups, editable), CSV columns, grey overlay tracks, and the track panel's **Change its group** in the overlay, event review and period review.
6. Groups page.
7. Batch card, the reviewed-events cross-check and the with/without toggle.

## Questions
1. **Default groups when the node is added:** bees and wasps ticked (proposed), or nothing ticked (no filter until chosen)? <bee and wasp is okay as default groups to keep for now.>
2. **The share threshold:** 50% (proposed)? A higher value makes more tracks unclear, and those are kept. <fine. 50% is fine>
3. **Flies, hoverflies, moths:** not counted when unticked, like ants (proposed)? Or should only ants and crawlers ever be dropped? <I should be able to select which groups to filter out as needed. today perhaps ants, but tomorrow or in another study it could be moth>
4. **Group rules:** yours across every batch (proposed), or per pipeline? <should set it in the pipeline like all other programing>
5. **The fixed groups:** is the list enough, or should you be able to add your own groups (e.g. "small dark bees")? <yes I should be able to add groups as needed by my self. \>

**Feedback:**

## Built (2026-10-31)
The user's answers changed the plan in two places:
- **Rules live in the pipeline, not a `GroupRule` model.** They're the Identify Species node's `group_rules` config (JSON: `{"groups": [...], "rules": [{kind, value, group} | {kind: "track", job, track, group}]}`).
  - A change from the review is written into that node in every run of the batch and in the pipeline, when the reviewer owns it.
  - There's no separate Groups page: the node's **Edit** dialog in the editor lists your groups and rules.
- **The checklist says which groups to leave out** (`leave_out`). The default is everything but bees and wasps; Unclear can't be left out.

What's in it:
- **`apps/pipelines/taxon_groups.py`:**
  - GBIF lineage, cached 90 days under hashed keys; BeeMachine's coarse classes need no lookup;
  - `default_group`, `settings`;
  - `crop_readings` (from `track_votes.csv`, else the voted taxon);
  - `track_groups`: rules first, then the group vote, `MIN_CROPS` = 3; a track rule wins outright;
  - `with_rule`.
- **`identify.species`:**
  - outputs `tracks`;
  - new fields `group_filter` (on/off), `leave_out` (`group_checklist`), `agree_share` and `group_rules`;
  - the executor passes `result`/`label` on and adds `track_groups`, `group_counts`, `dropped_tracks` and `names_seen`;
  - the GPU cache key is unchanged (`_pipeline_species` sends only the model).
- **`_analysis_inputs`** drops `dropped_tracks`. **Events** also counts the unfiltered tracks and saves `filtered_out` (tracks, events by group, event keys).
- **Recount** re-runs Identify Species first, in dependency order (`recount.dependency_order`).
- **`apps/pipelines/regroup.py`** and `analysis:track_group` (POST, launcher only):
  - scope track, name, genus or family;
  - recounts this clip now and queues the batch's others (`counting_rules=0`, picked up by the sweep).
- **Player:**
  - tracks left out are grey, with their group on the label;
  - the followed-track panel shows the group and why, with **Change its group**, **Apply to**, **Save and recount** and **Undo that rule**.
- **Tracks, tracking, events and interactions CSVs** gain `group`, `group_share` and `counted`.
- **Run page:** group chips on the Identify Species step.
- **Batch page:** an **Insect groups** card with what was left out by group, and how reviewed events on those tracks were called.
- **Tests:** `apps/pipelines/tests/test_insect_groups.py` (23).

Not built yet:
- **The with/without precision toggle on the batch metrics.** The metrics already exclude left-out events; the card shows how reviewed ones were called.
- **"Change its group" on the period review player.** Its overlay shows the groups and greys left-out tracks, but changing a group is on the clip results player.


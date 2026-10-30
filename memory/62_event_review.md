# 62 — Event review: yes / no on each detected event

**Status:** audit + plan + design (2026-10-29), awaiting review.
Canvas: https://claude.ai/artifact/H9GUb3sgYQshaifeFwPNNf
Related: [[60_activity_extraction_audit]], [[61_events_on_the_video]], [[39_annotation_review_page]].

## Why
The user wants to check the detected events by hand. In "Events near now", each event gets Yes / No. A **review mode** plays the clip, pauses at each event, and waits for a vote.

## Audit
- **No existing verdict on events.** The only review machinery is annotation frames (`Annotation.reviewed`, `reviewed_by`, project roles), and that is about boxes on frames, not events.
- **Events are saved rows** in `PipelineRun.context` (memory/60). A recount replaces them. Their identity is the event itself: action, reference, track, frame. A vote must hang on that, not on a row index.
- **The player** already has everything to drive review mode: events are served live (`?part=events`); `seekFrame`, `onFrame`, the "events near now" list and the tube lanes all exist (memory/61).
- **Who can see a clip:** `Job.objects.for_viewer`, plus device shares. Who can change it: `Video.manageable`.

## Decisions (2026-10-29)
- **A rejected event stays in the CSV.** Columns record the reviewer's call and notes (below).
- **No queue options.** Review walks every event of the clip, or of the period, in order. Voting again changes your vote.
- **Invite reviewers to a batch**, like annotation projects (`ProjectShare`), in addition to the public link (which stays view-only).
- **Reviewers vote on trips too** (decided 2026-10-29). A trip becomes data only when a person confirms it (§4b).
- **New: review a period.** A range of hours in a day, or days in a week, across the adjacent analysed clips, so a bee's exit in one clip and its entry in a later one can be seen and reviewed together.

## Plan

**1. Data.** New model `EventReview`:
- fields: `run` (FK), `event_key` = `"{action}|{reference}|{track_id}|{frame}"`, the event's fields copied (so a vote survives a recount that removes its event), `user`, `verdict` (accepted / rejected), `reason` (optional on a rejection: "not an event", "wrong tube", "wrong action", "can't tell"), and timestamps;
- unique on (run, event_key, user). Clicking your vote again removes it. A recount never touches votes; a vote whose event is gone is kept and shown as "no longer detected".

**2. Who can review: batch reviewers.** New model `BatchMember(batch_id, user, role = reviewer | viewer, created_by)`, shaped like `annotations.ProjectShare`.
- The owner (the batch's launcher) invites people by username or email from a **People** panel on the batch page.
- **Reviewer:** watch, vote, download. **Viewer:** watch, download.
- Neither can re-run, cancel or share.
- Invited batches show under **Runs › Shared with me**.
- The clip results page, overlay, crops and CSVs admit members of a batch the clip is in.
- The owner can always vote. The public link (memory/47) stays view-only.

**3. Clip review (the player).**
- **Events near now:** Accept / Reject (Yes / No) on each row.
- **Review events:** plays from 2 s before each event in order, pauses on it, and asks with Yes (Y) / No (N) / Replay 2 s (R) / Skip (→). After No, optional reason chips. Then it moves to the next event.
- The header shows "12 of 40 reviewed · 3 rejected". The tube lanes ring reviewed marks (green accepted, red rejected).
- **API:** `POST /analysis/<job>/events/review` with `{event_key, verdict|null, reason}`. `?part=events` carries each event's votes.

**4. Period review** (new page, `analysis/review/`).
- **Pick:** a device (or site), a from–to date-time (hours to days), and which pipeline's runs (default: the latest completed run per clip that has an Events step).
- **Clips:** every analysed clip in the range with a known recording time, in time order. Clips without a known start are listed apart; they can't be placed.
- **One time axis for the period:**
  - a clip strip (striped = no recording; click a clip to load it);
  - tube lanes across the whole period, with marks at their real clock times;
  - a toggle that draws a faint band from each exit to the next entry on the same tube, across clips, with the time between them on hover. **Drawn only, never counted or exported**, since trips stay the user's own reading (memory/60).
- **The player** loads the current clip and switches clip by itself at the end of one.
- **Review mode** walks every event in the period in time order. Votes are saved to each clip's own run, so the clip page, batch and CSVs show them too.
- A side panel shows the selected tube's events across the period.
- **Entry points:** **Review this period** on a batch page (its clips' range, by device) and on a device page.
- **Limits:** a period caps at 200 clips or 7 days, whichever comes first, so the page stays fast.

**4b. Trip review** (in period review; board "Trip review").
- **Suggestions, not data.** For each exit, the suggested trip is the next entry into the same tube within the period, across clips.
- **The card:** shows both moments side by side, paused on the exit and on the entry, with each track's clip, track id, species with confidence, marker, and a few crops, since the reviewer is judging whether it's the same bee. "Play both moments" replays them.
- **Votes:**
  - **Same bee, a trip (Y)** → accepted.
  - **Not this pair (N)** → the reviewer can re-link to a different later entry into that tube, or mark "didn't come back in this period" or "can't tell".
  - A re-link stores the pair the reviewer chose, as accepted, with a note that it was re-linked.
- **Data:** new model `TripReview`:
  - `exit_run` + `exit_key`, `entry_run` + `entry_key` (null for "didn't come back"), copies of both events' fields, `user`, `verdict`, `reason`, `relinked` and timestamps;
  - unique on (exit run, exit key, user): one call per reviewer per exit.
- **Export:** a **Reviewed trips CSV** for the period and the batch, with columns `reference, exit_time, entry_time, duration_sec, exit_clip, exit_track, entry_clip, entry_track, species, marker_id, review, reviewed_by, review_notes`. **Only reviewed pairs are in it.** Unreviewed suggestions are never exported, so no trip is in the data unless a person said so. This keeps memory/60's rule: the system doesn't compute trips as results.
- **Tests:**
  - accept, reject, re-link and "didn't come back" each stored;
  - two reviewers disagreeing → disputed;
  - unreviewed suggestions absent from every CSV;
  - a trip whose exit and entry are in different clips (and runs) round-trips.

**5. Data out.** The detected events are never changed by a review.
- **Events CSV, clip and batch:** new columns
  - `review`: accepted / rejected, `disputed` if reviewers disagree, blank if unreviewed;
  - `reviewed_by`: reviewers' usernames, `;`-separated;
  - `review_notes`: reasons given.
- **Clip page:** the Events tile adds "12 reviewed · 9 accepted · 3 rejected".
- **Batch table:** a "Reviewed" column, "12 / 40 · 3 rejected".

**6. Tests.**
- Votes are stored, changed and removed.
- Two reviewers give `disputed`.
- A vote survives a recount.
- The CSV columns.
- Only the owner and batch reviewers can vote; viewers and the public link can't.
- An invited member can open the batch's clips and nothing else.
- A period gathers the right clips in order, across clips; the exit→entry band data is not in any CSV.
- Period review saves votes onto the right runs.

## Build order
1, 3 and 5 (clip review and its data) → 2 (invites) → 4 (period review) → 4b (trip review). Each is shippable on its own.

## Questions
1. Period review across **several devices at once** (one timeline each, stacked), or one device / site at a time (planned)?
2. Which runs for a period when a clip was analysed more than once: **the latest run with an Events step** (planned), or let the user pick a pipeline?
3. Suggested trips pair an exit with the next entry into the same tube, **any track**. When markers are read, should the suggestion prefer the next entry by **the same marker** (and fall back to any)?

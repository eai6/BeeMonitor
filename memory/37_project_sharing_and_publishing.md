# 37 — Sharing annotation projects, and publishing them

**Status:** built — phases 1-4 shipped.
Canvas: https://claude.ai/code/artifact/96d31aa5-6b48-480f-acb6-2cfe70e6074c

## Two things, usually conflated

**Sharing** invites named people to work with you. **Publishing** makes a
finished thing discoverable and reusable by anyone. They need different
mechanics, and building one as a special case of the other produces a system
where either strangers can edit your data or collaborators cannot.

## What already exists

`DeviceShare` is the precedent and it is a good one: a role per (device, user),
`viewer` / `manager`, with `Device.accessible()` for reads and
`Video.manageable()` for writes, and a comment that says plainly which paths
must use which. Annotation projects have none of this — `AnnotationProject.user`
is a single FK and **32 view paths** filter on `user=request.user`.

## The property worth protecting deliberately

**Annotating needs the sampled frames, not the footage.** `FrameImageView`
serves `Annotation.frame_image_path` out of the *processed* bucket; the editor
never touches `raw-videos`. So a collaborator can draw boxes all day without
access to the source clips — or to the device, site and recording times
attached to them.

That falls out of the existing design rather than being engineered, and it is
the single most valuable thing here: it means inviting a student to label does
not hand them a field site's location history. **A share must not widen video
access**, and that has to be asserted by a test, because the natural
implementation — "give them `Video.accessible`" — throws it away.

## Roles

Mirroring `DeviceShare`, with one addition that the annotation case needs:

| role | can |
|---|---|
| **viewer** | see frames, boxes, stats, export |
| **annotator** | + draw/edit boxes, mark reviewed |
| **manager** | + add clips, sample, auto-label, edit classes |
| **owner** | + delete the project, manage shares, publish |

`annotator` is the whole point. Labelling is the work you want help with, and a
labeller must not be able to spend your GPU budget on auto-label or change the
class list halfway through a project — both of which quietly invalidate work
already done.

## Publishing

`visibility`: `private` (default) / `public`. A public project is
**readable and copyable, never writable** by strangers.

"Improve upon" therefore means **copy, then diverge**: frames, boxes and
classes are duplicated into your own project, with `copied_from` recorded.
Shared write with strangers is a moderation problem nobody asked for, and a
public dataset that changes underneath its users cannot be cited.

**Provenance is frozen, not live.** A public project records what it contained
when it was published. A copy taken later records what it took. Neither claims
to be the other.

### What a public project must not leak

Frames and boxes are the dataset. `device`, `site_name`, `location` and
`recorded_at` are field metadata, and for a bee hotel they are close to a
location history. Publishing therefore **omits them by default**, with an
explicit opt-in — the ecological value of publishing recording times is real,
but it is the publisher's call to make knowingly, once.

## Public models

`CustomModel.visibility`, same two values. A public model can be selected in
anyone's pipeline; the weights already live in a bucket the workers read.

Its card records the project it was trained on **and the counts at training
time** — frames, boxes, classes, and how many hotels and hours they spanned.
A model is an artefact of the data it saw; if the source project doubles in
size afterwards, the model did not change and must not appear to have.

## Plan

**Phase 1 — the model and the guards** (no UI)
1. `ProjectShare(project, user, role, created_at, created_by)`, unique on
   (project, user) — `DeviceShare`'s shape.
2. `AnnotationProject.accessible(user)` / `.annotatable(user)` /
   `.manageable(user)` / `.owned(user)`, each documented with which paths must
   use it.
3. Convert all 32 owner-only checks, one at a time, choosing the level
   deliberately per view rather than by search-and-replace.
4. Tests first for the ones that matter: a viewer cannot write a box, an
   annotator cannot spend GPU, nobody but the owner can delete or share.

**Phase 2 — sharing UI**
5. A Shares panel on project settings: invite by username/email, set a role,
   revoke. Mirrors the device share UI.
6. The project list shows shared-with-me projects, marked with the role.

**Phase 3 — publishing**
7. `visibility` on `AnnotationProject` and `CustomModel`, with the
   metadata opt-in.
8. A browse page for public projects and models.
9. Copy-a-project: duplicate frames, boxes and classes into a new project
   owned by the copier, recording `copied_from`.

**Phase 4 — the thing that makes it worth publishing**
10. A public project's card shows its coverage — the same device × hour map the
    picker uses — so someone deciding whether to reuse a dataset can see what it
    actually covers before they take it.

## Risks worth stating

- **The conversion in Phase 1 is where a security bug would live.** 32 call
  sites, and the failure mode is silent: a viewer who can delete looks exactly
  like a viewer until someone deletes. Hence tests before conversion, and one
  view at a time.
- **Frame images are served through a presigning view.** If that view checks
  ownership rather than the new access rule, shares silently do not work — and
  if it checks nothing, publishing leaks every project's frames. It needs its
  own test either way.
- **Copying a large project duplicates rows, not files.** Frames stay in the
  processed bucket and are referenced by path. That is cheap, and it means a
  copy breaks if the original is deleted — so a published project's frames must
  outlive its deletion, or copies must duplicate the objects. Decide before
  shipping Phase 3, not after.


---

## Assignment (added after the mockups)

Collaboration needs more than access: it needs **who is doing which clips**.

`ClipAssignment(project, video, user, assigned_at, assigned_by)`, unique on
(project, video) — a clip has at most one owner of the work.

**Per clip, not per frame.** A clip is a coherent scene: one hotel, one stretch
of time. Two people labelling the same clip keep re-deciding the same judgement
calls, and "who labelled this" stops having an answer. Splitting by clip makes
each person's work independently reviewable.

**An annotator works only what is assigned to them** — by someone else, or by
themselves out of the unassigned pool. Claiming creates an assignment like any
other, so every labelled clip has a record of who took it and when. Nothing is
stranded when a collaborator stops, and nobody is idle waiting to be given
work, but no work happens off the books either.

**Assignment can auto-distribute.** "Split the unassigned evenly between these
people" is one button and the common case; hand-picking 21 clips is not
something to make anyone do twice.

## Roles, revised

Five ranks, still linear — the property that makes `DeviceShare` easy to reason
about and hard to get subtly wrong:

    viewer(1) < annotator(2) < reviewer(3) < manager(4) < owner(5)

| role | can |
|---|---|
| **viewer** | see frames, boxes, stats; export |
| **annotator** | + draw and edit boxes on clips assigned to them; claim from the pool |
| **reviewer** | + mark frames reviewed, on **any** clip, not only their own |
| **manager** | + add clips, sample, auto-label, edit classes, assign work |
| **owner** | + delete the project, manage people, publish |

`reviewer` is the checking role, and it sits above `annotator` deliberately:
someone who spots a bad box while reviewing should be able to fix it rather
than file a complaint. What distinguishes them is *scope* — an annotator is
confined to their own assignments, a reviewer sees and signs off on everyone's.

## What a collaborator's page is

Their own workload, four numbers, and one button into the editor at the first
unlabelled frame of their first unfinished clip. No GPU controls, no project
settings, no footage. The common case is one click from opening the project to
drawing a box.


---

## What shipped

**Phase 1 — model and rules.** `ProjectShare` (viewer < annotator < reviewer <
manager < owner, linear), `ClipAssignment` (per clip, unique per project), and
four access scopes. Rules pinned as properties before any view was converted,
because the failure mode is silent.

**Phase 2 — the views.** ~20 owner-only lookups converted, each to the level
that view needs. Three video lookups filtered on ownership and would have made
every collaborator see an empty editor; they now require the clip to be in the
project.

**Phase 3 — the pages.** People page, per-clip assignment with round-robin
distribution, claim-from-pool, and a collaborator's own landing view.

**Phase 4 — publishing.** `visibility` on projects and models, a public browse,
and copy-a-project.

## Decisions taken during the build

**Copies reference the original frames.** Confirmed safe first: deleting a
project only cascades rows, and video deletion cleans raw video and job CSVs but
never `frame_image_path`. So the JPEGs outlive both and a copy cannot break.
The trade is orphaned frames accumulating in the processed bucket — a storage
leak that already existed and is worth a sweeper eventually.

**A copy does not attach the source clips.** A copy is a dataset — frames and
boxes. Attaching the videos would hand over footage the copier was never
shared, which is the property the whole sharing model protects.

**A copy is not marked reviewed.** Carrying the flag over would claim someone
signed off work they have never seen.

**Attribution survives deletion.** `copied_from` is SET_NULL and
`copied_from_name` is a string, so a copy that outlives its source still says
where it came from.

**Model provenance is frozen at training time.** `trained_on` is a snapshot —
project name, frame count, clips, devices, hours. Reading it live would let a
model trained on two hotels start claiming four the moment the project grew.

**A public model is usable, not just visible.** `CustomModel.usable(user)`
covers own-plus-published, and both the pipeline picker and the executor go
through it; the weights already live in a bucket the workers read.

## Found on the way

**Dataset export was broken for everyone.** `ExportProjectView.get` re-imported
`io` forty lines below its own `io.BytesIO()`, binding the name local for the
whole function. `test_shadowed_imports` now walks every module for that
pattern.

**Three places offered what the rules refuse** — Sample/Auto-label rendered for
everyone and 404'd; the project list offered Edit and Delete on other people's
projects; the empty state told collaborators to use Auto-label. Gating the view
is the security fix, gating the control is the usability one, and only the
first is obvious while writing the view.

## Still open

- Orphaned frame objects are never swept from the processed bucket.
- Public projects have no coverage map on their card yet (plan item 10) — the
  device x hour grid from the picker would go straight in.
- Nothing rate-limits copying, and a copy of a large project bulk-creates one
  row per frame.

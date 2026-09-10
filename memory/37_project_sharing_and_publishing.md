# 37 — Sharing annotation projects, and publishing them

**Status:** audit complete, plan for execution.

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

# Annotation & sharing

Label your own frames to train a detector for your insects and your setting: a new species, a flower, an
arena, or a hotel the built-in models haven't seen.

<!-- SCREENSHOT: the project page with frames to review -->

## A project

1. **Annotations → New project**, with your classes (for example *bee*, *wasp*).
2. **+ Add Videos** — pick clips by hotel and date.
3. **Clips → Sample & label** — the GPU finds the busiest frames of each clip and labels them with SAM 3.
4. **Frames to review** — check each frame: keep, move or delete boxes, add any bee it missed.

## Working as a team

Invite people under **People** as *viewers*, *reviewers* (check and fix frames) or *managers* (also add
clips, sample and assign). See [Sharing & publishing](#sharing-publishing). A manager uses **Assign frames…** to give someone 100, 500 or more frames to review; they see
**Your review queue** on the project.

## Train and use a model

**Export** downloads the labelled dataset. **Training** fine-tunes a model on it; a trained model can then
be chosen in a pipeline's *Detect objects* step. Projects and models can be [published](#publish) under **Browse** for others to copy and use.

## Sharing & publishing

**Sharing** invites named people to work with you. **Publishing** makes a finished dataset or model
available to everyone to copy and use.

### Share a device

On the device page, **Share** with a username or email:

| Role | Can |
|---|---|
| Viewer | See the device, its clips, photos and results |
| Manager | Also change camera, recording and processing settings |

### Share an annotation project

On the project, **People → Invite**:

| Role | Can |
|---|---|
| Viewer | See frames and labels, export |
| Reviewer | Also check and fix boxes |
| Manager | Also add clips, run sampling and auto-labelling, assign frames |

People you share a project with see the sampled frames, **not** the source videos, devices or field
sites. You can invite students to label without giving them a site's location history.

### Publish

A published project or trained model is listed under **Browse**. Anyone can read and copy it, but nobody else
can change it.

- **Projects**: copying a project duplicates its frames, boxes and classes into a new project owned by the
  copier, which records where it came from.
- **Models**: a public model can be chosen in anyone's pipeline *Detect* step. Its card records the dataset
  it was trained on and how large that dataset was at training time.

Recording times, sites and devices are **left out** of a published project unless you choose to include them.

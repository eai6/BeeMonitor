# Annotation & training

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
clips, sample and assign). See [Sharing & publishing](sharing.md). A manager uses **Assign frames…** to give someone 100, 500 or more frames to review; they see
**Your review queue** on the project.

## Train and use a model

**Export** downloads the labelled dataset. **Training** fine-tunes a model on it; a trained model can then
be chosen in a pipeline's *Detect objects* step. Projects and models can be [published](sharing.md#publish) under **Browse** for others to copy and use.

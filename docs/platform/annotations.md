# Annotations

Label your own frames to teach a detector your insects and your setting: a new species, a flower, an arena,
or a hotel the built-in models haven't seen. Once a project is labelled, [train a model](training.md) on it.

## A project

1. **Annotations → New project**, with your classes (for example *bee*, *wasp*).
2. **+ Add Videos**: pick clips by unit and date.
3. **Clips → Sample & label**: the GPU finds the busiest frames of each clip and labels them with SAM 3.
4. **Frames to review**: check each frame. Keep, move or delete boxes, and add anything it missed.

**Export** downloads the labelled dataset in YOLO format.

## Working as a team

On the project, **People → Invite** someone by username or email:

| Role | Can |
|---|---|
| Viewer | See frames and labels, export |
| Reviewer | Also check and fix boxes |
| Manager | Also add clips, run sampling and auto-labelling, assign frames |

A manager uses **Assign frames…** to give someone 100, 500 or more frames to review; they see them under
**Your review queue** on the project.

People you invite see the sampled frames, **not** the source videos, devices or field sites. You can invite
students to label without giving them a site's location history.

## Publishing a project

**Publish** lists a finished project under [Browse](browse.md), where anyone can read it and take a copy.
Nobody else can change yours. Recording times, sites and devices are **left out** unless you choose to
include them.

# Sharing & publishing

**Sharing** invites named people to work with you. **Publishing** makes a finished dataset or model
available to everyone to copy and use.

## Share a device

On the device page, **Share** with a username or email:

| Role | Can |
|---|---|
| Viewer | See the device, its clips, photos and results |
| Manager | Also change camera, recording and processing settings |

## Share an annotation project

On the project, **People → Invite**:

| Role | Can |
|---|---|
| Viewer | See frames and labels, export |
| Reviewer | Also check and fix boxes |
| Manager | Also add clips, run sampling and auto-labelling, assign frames |

People you share a project with see the sampled frames, **not** the source videos, devices or field
sites. You can invite students to label without giving them a site's location history.

## Publish

A published project or trained model is listed under **Browse**. Anyone can read and copy it, but nobody else
can change it.

- **Projects**: copying a project duplicates its frames, boxes and classes into a new project owned by the
  copier, which records where it came from.
- **Models**: a public model can be chosen in anyone's pipeline *Detect* step. Its card records the dataset
  it was trained on and how large that dataset was at training time.

Recording times, sites and devices are **left out** of a published project unless you choose to include them.

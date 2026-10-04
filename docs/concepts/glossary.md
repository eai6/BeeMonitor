# Glossary

**Unit / device**: a BeeMonitor recording module (Raspberry Pi, camera, enclosure) enrolled on your account.

**Video / clip**: one recording. A unit cuts its recordings into clips, one per burst of motion or one every
10 minutes in continuous mode.

**Site**: where a video was recorded. Set from the device's location, or by you when uploading.

**ROI (region of interest)**: the part of the picture that matters. On a unit, motion only starts a
recording inside the ROI.

**Reference / reference object**: what behaviour is measured against, such as a nest tube, a flower or a
drawn region.

**Layout**: a device's ROI plus its reference objects. Every change is kept as a new version, so older clips
are analysed with the layout they were recorded with.

**Detection**: one box around one object in one frame.

**Track**: the detections of one animal linked across frames, with an id that is valid within its clip.

**Event**: a track entering or exiting a reference at a moment.

**Interaction / episode**: two things (two tracks, or a track and a reference) together for a span of time.

**Foraging trip**: an exit from a nest tube followed by the next entry into the same tube.

**Visit**: an interaction between a track and a reference.

**Dwell time**: the duration of a visit.

**Pipeline**: a reusable graph of steps. A **run** is one pipeline applied to one clip, and a **batch** is
the same pipeline run on many clips.

**Annotation project**: frames and boxes you label to train a model.

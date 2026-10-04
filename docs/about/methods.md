# How it works

## On the device: motion-gated recording

The camera writes video into a short buffer all the time. A small 640×480 copy of each frame goes through
background subtraction (MOG2). When insect-sized blobs move inside the ROI, the buffer and what follows are
saved as a clip, which ends 5 s after the motion stops. A daily calibration measures the insects in your own
clips to tune the blob sizes. Because the unit records only when something is active, it saves 10–100×
less data, which makes uploading practical.

## In the cloud: detection and tracking

- **Detection**: fine-tuned YOLO26 models find each class you ask for (bees, nest tubes, other insects) in
  every frame. SAM 3 detects classes from a text prompt when no trained model exists yet.
- **Tracking**: BeeTrack follows each animal. A Kalman filter predicts where it goes, the Hungarian algorithm
  matches detections to tracks, and a lost track can be recovered within 0.5 s.
- **Species**: BeeMachine (354 bee taxa) classifies every frame of a track and takes the majority vote.
- **Two-mode processing**: lightweight motion detection skips idle stretches, and full tracking runs only
  when something moves (2.9× faster).

## From tracks to behaviour

One pass over the tracks and the [reference](../concepts/index.md#3-the-reference) finds **episodes**: the
contiguous runs of frames in which a track is inside a reference or near another track. A short gap
(15 frames by default) doesn't break an episode. Each episode becomes an **interaction**, and its start and
end become **enter** and **exit** events. Because both tables come from the same pass, they always agree.

At nest tubes, a Random Forest classifier also judges entries and exits. It uses 20 features of each track
near a tube (shape, speed, distance to the nest, direction) to separate real entries and exits from bees
that just walk past. Those rows are marked `source = gpu`.

## Accuracy

Validated on bee hotels: on 110 minutes of video with 300 hand-annotated foraging events, precision was
93.9%, recall 87.7% and F1 0.907 (full tracking). Other setups (flowers, lab assays) use the same detection and tracking but
haven't been validated separately yet. Check a sample of your own results before relying on them.

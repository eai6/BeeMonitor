# How it works

## On the device: motion-gated recording

The camera writes video into a short buffer all the time. A small 640×480 copy of each frame goes through
background subtraction (MOG2): when bee-sized blobs move inside the ROI, the buffer and what follows are
saved as a clip, ending 5 s after motion stops. A daily calibration measures the bees in your own clips to
tune the blob sizes. Recording only activity cuts data 10–100×, so it can upload.

## In the cloud: detection and tracking

- **Detection** — fine-tuned YOLO26 models find the hotel's nest tubes (once per clip) and bees (every frame).
- **Tracking** — BeeTrack follows each bee: a Kalman filter predicts where it goes, the Hungarian algorithm
  matches detections to tracks, and lost tracks can be recovered for 0.5 s.
- **Events** — a Random Forest classifier, on 20 features of each track near a tube (shape, speed, distance
  to the nest, direction), decides real entries and exits from noise.
- **Two-mode processing** — lightweight motion detection skips idle stretches; full tracking runs only when
  something moves (2.9× faster).

## Accuracy

On 110 minutes of video with 300 hand-annotated foraging events: precision 93.9%, recall 87.7%, F1 0.907
(full tracking).

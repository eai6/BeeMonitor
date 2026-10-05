# 42 · Keep fine detail on the OAK; species + marker ID from every crop of a track

Status: **Part B built** (2026-10-05; needs a GPU image build + tag bump). Part A (OAK) next, tested on one unit. Track crops are already padded and saved for
every detected frame (commit 149794b, not yet in a GPU image).

## Goal

A track's crops, from its whole trajectory, carry enough detail to identify the
species (BeeMachine or BioCLIP, chosen in the pipeline) and the paint marker.
Each crop votes; the track takes the taxon / marker with the most votes.

## Part A — record without losing fine detail (device, OAK)

### What limits detail today
- H.264 Main at the encoder's auto bitrate: ~13.6 Mbit/s at 12 MP 20 fps
  ≈ 0.06 bit/pixel. Fine texture is quantised away before upload.
- The ISP's luma/chroma denoise and sharpening smooth fine texture.
- Auto-exposure at 1/fps-ish shutter: a flying bee smears across pixels —
  motion blur is detail no encoder can bring back.

### Hardware facts (Luxonis docs/forum)
- H.264/H.265 encoder: 248 MPix/s (4K30 or 12 MP@20). 12 MP@20 is the ceiling.
- Myriad X HEVC: Main profile, no B-frames, 1 ref frame, no adaptive
  quantisation, CBR or VBR. Bitrate set with setBitrateKbps (CBR).
- "All codecs are lossy except lossless MJPEG." MJPEG: up to 450 MPix/s.
- Measured here (oak.py docstring): continuous 12 MP JPEG beside 12 MP H.264
  stalls both encoders; a 12 MP q95 JPEG is ~4 MB.

### Truly lossless is not feasible continuously
| stream | per frame | 20 fps | sink |
|---|---|---|---|
| raw NV12 12 MP | 18 MB | ~365 MB/s | > Pi 4 USB3 + any SD card |
| lossless MJPEG 12 MP | ~10–15 MB | ~250 MB/s | > SD (~40–90 MB/s) |
| MJPEG q95 12 MP | ~4 MB | ~80 MB/s | > SD; ok only on USB SSD |

### Plan (all as config, tested on one unit before rollout)
1. **High-bitrate CBR.** `OAK_BITRATE_KBPS` default 0 → a measured value.
   Sweep 30 / 60 / 100 / 150 Mbit/s; keep the highest the encoder holds at
   20 fps with no dropped frames. ~60 Mbit/s ≈ 0.25 bit/px ≈ 27 GB per hour of
   recorded video (motion-gated, so far less per day). Upload time rises in
   proportion (see hotspot discussion) — accepted.
2. **H.265 option.** Better detail per bit than H.264 on the same encoder.
   Cost: browsers. Safari plays HEVC; Chrome/Edge only with hardware decode;
   Firefox no. Either keep H.264 at high bitrate, or record H.265 and transcode
   a small H.264 preview in the cloud for the clip page. Decide after the sweep.
3. **ISP:** luma denoise 0, chroma denoise low, sharpness 0 (sharpening halos
   look like detail but aren't). Test noise vs. bitrate.
4. **Shutter:** cap exposure (e.g. ≤ 1/1000 s) with gain to compensate, so a
   moving bee isn't smeared. Likely the biggest single gain for crops.
5. **Later, if still short:** record a full-quality crop of the ROI only
   (MJPEG q≈98 or lossless on the hotel region) next to the full-frame stream.

## Part B — species and marker from every crop (GPU job, after tracking)

Today: species runs inline per frame on its own unpadded boxes (incl.
predicted frames), with a 0.5 floor, max 25 votes; every failure is silent.
Markers are read on the web server from 12 sampled crops.

New: after tracking, on the GPU worker, over the saved crops:
1. For each track, load all its crops; classify in large batches with the
   pipeline's model: **BeeMachine** (ONNX, already in the image) or **BioCLIP**
   (pybioclip added to the GPU image; optional candidate-species list).
2. Every crop votes its top-1. Track taxon = most votes (tie → higher mean
   confidence). Optional confidence floor, default off.
3. Same pass for markers: colour decoder over every crop, vote.
4. Outputs: tracking CSV `taxon`, `taxon_votes`, `taxon_vote_share`,
   `marker_id`…; `track_votes.csv` with every crop's reading (auditable).
5. `species_status` in summary_stats: model loaded?, crops classified, why not.
   The Identify species step shows that instead of "no labels".
6. Remove the inline per-frame classification from the tracker loop.
7. Pipeline steps: Identify species gets **Model: BeeMachine | BioCLIP**;
   Read marker becomes a GPU-job flag (now actually consumed). Old jobs keep
   the CPU marker fallback.

## Decisions (2026-10-05)
- BioCLIP: the region's candidate species (monitor/priors.region_taxa from the
  device's lat/lon + recording month); whole Tree of Life when none.
- No confidence floor: every crop votes its top-1.
- Order: Part B first, then Part A.

## Built (Part B)
- `beemonitor/identification/track_vote.py` (vote over crops),
  `bioclip.py` (BioClipIdentifier), `SpeciesIdentifier.classify_images`.
- Worker: `identify_tracks` after tracking → tracking CSV taxon/bee_id
  columns + `track_votes.csv`; `summary_stats.identification` status. The
  inline per-frame species path is no longer wired (still in the package).
- Platform: Identify species has Model (beemachine|bioclip); Read marker sets
  identify_markers (now consumed, so it re-runs the GPU job);
  `_candidate_taxa` sends the region list for BioCLIP (not hashed).
- GPU image: pybioclip + baked BioCLIP / ToL weights.

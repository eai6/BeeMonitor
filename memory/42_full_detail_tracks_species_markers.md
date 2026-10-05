# 42 · Keep fine detail on the OAK; species + marker ID from every crop of a track

Status (2026-10-05): **Part B built** (species/marker votes over crops; GPU
image + tag bump pending). **Part A code built** — OAK detail settings are in
the device software as per-unit config; **next: run the test on the OAK unit**
(runbook below), then pick defaults.

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


## Built (Part A) — OAK detail settings

All per-unit, in `/etc/beemonitor/uploader.env` (read by
`beemonitor-recorder.service`). Unset = today's behaviour exactly.

| variable | values | what it does |
|---|---|---|
| `BEEMONITOR_OAK_CODEC` | `h264` (default) / `h265` | encoder profile; H.265 work files are `.h265`, remuxed with `-f hevc -tag:v hvc1` |
| `BEEMONITOR_OAK_BITRATE_KBPS` | `0` (auto) or kbit/s | CBR at that rate (`setRateControlMode(CBR)` + `setBitrateKbps`) |
| `BEEMONITOR_OAK_LUMA_DENOISE` | `-1` (default) or 0..4 | ISP luma denoise |
| `BEEMONITOR_OAK_CHROMA_DENOISE` | `-1` or 0..4 | ISP chroma denoise |
| `BEEMONITOR_OAK_SHARPNESS` | `-1` or 0..4 | ISP sharpening |
| `BEEMONITOR_OAK_MAX_EXPOSURE_US` | `0` (no cap) or µs | auto-exposure shutter limit (`setAutoExposureLimit`) |

Code: `hardware/motion/config.py` (settings), `motion/oak.py`
(`OakCamera(codec=, bitrate_kbps=, isp=)`, `_apply_isp()` after start),
`motion/remux.py` (HEVC remux), `motion/recorder.py` (wiring + log line
`camera: OAK … h264 at 60000 kbit/s CBR`). Test tool:
`hardware/oak_quality_test.py`. Unit tests: `hardware/test_oak_settings.py`.
All depthai calls checked against depthai 3.10.0 stubs; **none run on an OAK
yet** — that is the point of the runbook.

## Runbook for the OAK unit (for Claude on the Pi)

Goal: find the most detail this unit's OAK can record without dropping frames,
and set it. Large files are fine; losing fine detail is not. Report the numbers
back (results.json + which PNGs look sharpest) so the defaults can be chosen.

1. **Update the unit** to a commit that has `hardware/oak_quality_test.py`
   (dashboard → device → Update, or however this unit is updated). Confirm:
   `python3 hardware/test_oak_settings.py` prints "all OAK settings checks passed".
2. **Point the camera at the real scene** (the hotel / flowers, in daylight).
   Fine texture matters — that is what is being judged.
3. **Stop the recorder** (it holds the OAK):
   `sudo systemctl stop beemonitor-recorder`
4. **Sweep codec × bitrate**, camera defaults otherwise (~4 min), from the
   `hardware/` directory with the recorder's venv python:
   `python3 oak_quality_test.py`
   Output: one line per setting (OK/DROP, fps, dropped %, Mbit/s, MB/min) and
   `oak_quality/<time>/` with an `.mp4` + a lossless `.png` per setting.
   - A setting that DROPs or FAILs (encoder "out of resources") is too high.
   - Note the highest OK bitrate for h264 and for h265 separately.
5. **ISP + shutter**, at the best codec/bitrate from step 4, e.g.:
   `python3 oak_quality_test.py --codecs h265 --bitrates 100000 --isp 0,0,0`
   `python3 oak_quality_test.py --codecs h265 --bitrates 100000 --isp 0,0,0 --max-exposure-us 1000`
   `python3 oak_quality_test.py --codecs h265 --bitrates 100000 --isp 0,1,0 --max-exposure-us 2000`
   Compare the PNGs at 100 % zoom on the textured area: crisp texture without
   smearing is the goal; some grain from denoise 0 is expected and fine. With
   an exposure cap, check the scene isn't too dark/noisy in shade.
6. **Set the chosen values** in `/etc/beemonitor/uploader.env`, e.g.
   ```
   BEEMONITOR_OAK_CODEC=h265
   BEEMONITOR_OAK_BITRATE_KBPS=100000
   BEEMONITOR_OAK_LUMA_DENOISE=0
   BEEMONITOR_OAK_CHROMA_DENOISE=0
   BEEMONITOR_OAK_SHARPNESS=0
   BEEMONITOR_OAK_MAX_EXPOSURE_US=1000
   ```
   then `sudo systemctl start beemonitor-recorder` and check
   `journalctl -u beemonitor-recorder -n 50` shows
   `camera: OAK … h265 at 100000 kbit/s CBR` and `OAK ISP: {...}` and no
   "H.264 stream died" / "delivered nothing" errors.
7. **Watch real clips for a while**: wave at the camera; a clip should land in
   `RECORD_DIR/<day>/` and upload. Check:
   - `ffprobe` on a clip: codec, ~20 fps, expected bitrate;
   - SD card use (`df -h`) — at 100 Mbit/s a minute of clip is ~750 MB;
   - the clip plays on the platform's clip page. **H.265 caveat:** Chrome /
     Edge play HEVC only with hardware decode, Firefox not at all; Safari does.
     If it won't play in the user's browser, say so — the cloud then needs an
     H.264 preview transcode (not built yet), or fall back to h264.
8. **Report back**: results.json from steps 4–5, the chosen env values, SD
   use per hour of recording, and whether clips play on the platform.

To undo: remove those lines from uploader.env and restart the recorder.

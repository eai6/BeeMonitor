# BeeMonitor

**An open-source camera trap and platform for monitoring cavity-nesting solitary bees at bee hotels.**

![A BeeMonitor unit](assets/beemonitor_hardware.png)

BeeMonitor watches a bee hotel, records every bee that arrives or leaves, and turns the clips into
nest visits, foraging trips and species — without anyone watching hours of footage.

## How it works

1. **Record.** A Raspberry Pi with a camera watches the hotel. Motion inside the hotel's region of
   interest (ROI) starts a short clip; full-resolution photos can be taken too.
2. **Upload.** Clips and photos upload over WiFi. Health and commands also work over 4G. A Witty Pi
   switches the unit on and off on a schedule, and a solar panel keeps it running in the field.
3. **Analyse.** On the [BeeMonitor platform](https://beemonitor.edwardamoah.com), pipelines detect and
   track bees, count nest entries and exits, and identify species. You can label your own data and train
   your own models.

## Where to start

<div class="grid cards" markdown>

- **[Build a unit](hardware/index.md)** — parts, 3D-printed enclosure, assembly and field deployment.
- **[Set up a unit](software/setup.md)** — flash the card, enrol it on your account, power on.
- **[Use the platform](platform/devices.md)** — draw the ROI, focus, record, review and analyse.
- **[Cite BeeMonitor](about/license.md)** — open source under AGPLv3.

</div>

## Accuracy

On 110 minutes of video with 300 hand-annotated foraging events:

| Mode | Precision | Recall | F1 |
|---|---|---|---|
| Full tracking | 93.9% | 87.7% | 0.907 |
| Two-mode adaptive | 92.0% | 84.3% | 0.880 |

## Acknowledgements

NSF Research Traineeship Program (INSECT NET, Grant 2243979) · USDA NIFA Hatch and Smith-Lever
Appropriations (PEN04943, PEN08801) · Penn State Joan Luerssen Faculty Enhancement Fund.

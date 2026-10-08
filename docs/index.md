# BeeMonitor

**Open-source hardware and an open-source analysis platform for studying pollinator behaviour from video.**

![A BeeMonitor unit](assets/beemonitor_hardware.png)

BeeMonitor started as a camera trap for bee hotels. It is now a general system: video of insects from a
BeeMonitor unit or any other camera, analysed with computer-vision pipelines you build from blocks —
detect, track, identify species and individuals, and measure behaviour.

## Start here

<div class="grid cards" markdown>

- **[Get started](get-started.md)**: get video in, run a pipeline, read the results.
- **[Build a unit](hardware/index.md)**: parts, costs and ten assembly steps, with a printable manual.
- **[Use the platform](platform/index.md)**: devices, videos, pipelines, annotation, API.

</div>

Everything (the hardware, device software, platform, analysis code and these docs) is open source under
AGPLv3 on [GitHub](https://github.com/eai6/BeeMonitor). See [About](about.md) to cite it or contribute.

## Accuracy

Validated on bee hotels: 110 minutes of video with 300 hand-annotated foraging events
([preprint](https://www.biorxiv.org/content/10.64898/2026.07.10.737879v1)).

| Mode | Precision | Recall | F1 |
|---|---|---|---|
| Full tracking | 93.9% | 87.7% | 0.907 |
| Two-mode adaptive | 92.0% | 84.3% | 0.880 |

## Acknowledgements

NSF Research Traineeship Program (INSECT NET, Grant 2243979) · USDA NIFA Hatch and Smith-Lever
Appropriations (PEN04943, PEN08801) · Penn State Joan Luerssen Faculty Enhancement Fund.

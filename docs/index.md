# BeeMonitor

**Open-source hardware and an open-source analysis platform for studying pollinator behaviour from video.**

![A BeeMonitor unit](assets/beemonitor_hardware.png)

BeeMonitor started as a camera trap for bee hotels. It is now a general system: any video of insects,
whether from a BeeMonitor unit or another camera, goes through the same pipeline and comes
out as two tables, **events** and **interactions**.

## Two tables, any question

Every pipeline produces the same two tables:

- **Events**: something entered or exited something. *A bee left tube 3 at 12.4 s.*
- **Interactions**: two things were together for a while. *A bee was on the flower from 3.0 s to 9.5 s.*

You then derive what your study needs from them: foraging trips at a nest hotel from events; visits and time
on a flower, or which pollen tube in a lab assay gets more (and longer) bee interactions, from interactions; encounters between insects
from interactions between two tracks. See [Results](platform/results.md).

## Start here

<div class="grid cards" markdown>

- **[Get started](get-started.md)**: get video in, run a pipeline, read the results.
- **[Build a unit](hardware/index.md)**: parts, enclosure, assembly, setup and deployment.
- **[Use the platform](platform/index.md)**: devices, videos, pipelines, annotation, API.

</div>

Everything (the hardware, device software, platform, analysis code and these docs) is open source under
AGPLv3 on [GitHub](https://github.com/eai6/BeeMonitor). See [About](about.md) to cite it or contribute.

## Accuracy

Validated on bee hotels: 110 minutes of video with 300 hand-annotated foraging events.

| Mode | Precision | Recall | F1 |
|---|---|---|---|
| Full tracking | 93.9% | 87.7% | 0.907 |
| Two-mode adaptive | 92.0% | 84.3% | 0.880 |

## Acknowledgements

NSF Research Traineeship Program (INSECT NET, Grant 2243979) · USDA NIFA Hatch and Smith-Lever
Appropriations (PEN04943, PEN08801) · Penn State Joan Luerssen Faculty Enhancement Fund.

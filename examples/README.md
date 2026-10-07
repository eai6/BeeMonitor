# Example clips

Two short motion-triggered clips from a BeeMonitor unit at a bee hotel (site `mendels`, 8 May 2024). Use
them to try the analysis engine:

```python
from beemonitor import BeeMonitor
from beemonitor.core.config import Config

results = BeeMonitor(config=Config.default()).analyze_video(
    "examples/mendels_2024-05-08_15_57_03.mp4", output_folder="output/")
print(results.events)
```

The device software's motion-gate replay tool reads them too:
`python3 hardware/motion_replay.py examples/mendels_2024-05-08_15_57_03.mp4`.

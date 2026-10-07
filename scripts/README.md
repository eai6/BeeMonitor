# Scripts

| Script | What it does |
|---|---|
| `train_event_classifier.py` | Trains the entry/exit event classifier (random forest) shipped in `models/event_classifier_model.pkl` |
| `cross_validation_classifier.py` | Leave-one-video-out cross-validation of that classifier |
| `camera-flip-test.sh` | Checks on a unit whether the camera's 180° rotation actually reaches the picture |
| `switch-to-ov5647.sh` | Switches a unit's camera overlay from the OwlSight to the OV5647 module |
| `sync-claude-memory.sh` | Backs up or restores the device runbook notes in `hardware/claude-memory/` |

The two classifier scripts expect the evaluation videos and manual annotations in `data/`, which is not
in the repository (see [`research/`](../research/)).

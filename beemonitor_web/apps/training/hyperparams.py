"""Advanced training settings: the Ultralytics arguments a user may set.

Most people never touch these, so the form hides them under "Advanced
settings" and a job stores only the ones that were set (``TrainingJob.
hyperparams``); anything left blank is Ultralytics' own default. The trainer
(sagemaker_backend/train.py) passes the same names straight to
``YOLO.train``, and keeps its own copy of this list so an unknown key in a
manifest is never handed to it.
"""

# (name, label, kind, min, max, Ultralytics default, help). kind: float | int
# | bool | choice (min holds the choices).
OPTIMIZERS = ("auto", "SGD", "Adam", "AdamW")

SPEC = [
    ("Optimisation", [
        ("optimizer", "Optimizer", "choice", OPTIMIZERS, None, "auto",
         "Auto picks one (and its learning rate) from the dataset size."),
        ("lr0", "Learning rate", "float", 1e-6, 0.1, 0.01,
         "Starting learning rate. Needs an optimizer other than Auto."),
        ("lrf", "Final LR fraction", "float", 0.001, 1.0, 0.01,
         "The learning rate ends at learning rate × this."),
        ("momentum", "Momentum", "float", 0.5, 0.999, 0.937,
         "SGD momentum / Adam beta1. Needs an optimizer other than Auto."),
        ("weight_decay", "Weight decay", "float", 0.0, 0.01, 0.0005, ""),
        ("warmup_epochs", "Warm-up epochs", "float", 0.0, 10.0, 3.0, ""),
        ("cos_lr", "Cosine LR schedule", "bool", None, None, False,
         "Lower the learning rate along a cosine curve instead of a line."),
    ]),
    ("Stopping and fine-tuning", [
        ("patience", "Early-stop patience", "int", 0, 500, 100,
         "Stop after this many epochs without improvement (0 = never)."),
        ("freeze", "Frozen layers", "int", 0, 24, 0,
         "Keep the first N layers fixed — useful for small datasets."),
        ("seed", "Random seed", "int", 0, 2**31 - 1, 0,
         "Same seed, same data, same settings → a repeatable run."),
    ]),
    ("Augmentation", [
        ("mosaic", "Mosaic", "float", 0.0, 1.0, 1.0,
         "Chance of stitching four frames into one."),
        ("fliplr", "Flip left–right", "float", 0.0, 1.0, 0.5, "Chance per frame."),
        ("flipud", "Flip up–down", "float", 0.0, 1.0, 0.0, "Chance per frame."),
        ("degrees", "Rotation (°)", "float", 0.0, 180.0, 0.0, "Random rotation up to ±this."),
        ("scale", "Scale", "float", 0.0, 0.9, 0.5, "Random zoom of ±this fraction."),
        ("translate", "Translate", "float", 0.0, 0.9, 0.1, "Random shift of ±this fraction."),
        ("hsv_v", "Brightness", "float", 0.0, 1.0, 0.4, "Random brightness change of ±this fraction."),
    ]),
]

FIELDS = {f[0]: f for _, group in SPEC for f in group}
NEEDS_OPTIMIZER = ("lr0", "momentum")


def labelled(values):
    """``[(label, value)]`` for a job page, in the form's order."""
    values = values or {}
    return [(FIELDS[k][1], values[k]) for k in FIELDS if k in values]

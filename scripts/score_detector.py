#!/usr/bin/env python3
"""Score a YOLO detector on a test project exported from BeeMonitor (memory/57).

    python scripts/score_detector.py model.pt "Test days_yolo_dataset.zip"
    python scripts/score_detector.py model.pt export_dir --map bee=insect --json scores.json

The test project should hold only clips from the left-out days. Export it
with only the frames a person checked: add ``?reviewed=1`` to the project's
Export link (``/annotations/<id>/export/?reviewed=1``). All exported frames
(both the export's train and val folders) are scored. Prints mAP50, mAP50-95, precision and recall overall, per
site and per day. Site and day come from each frame's file name, which starts
with the clip title (``mendels_2024-04-21_09_00_00…``).

The model's classes are matched to the dataset's by name. ``--map bee=insect``
scores the model's "bee" boxes against the dataset's "insect" boxes;
``--single-cls`` ignores classes altogether. Needs ``ultralytics``.
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

TITLE = re.compile(r"^(?P<site>.+?)_(?P<day>\d{4}-\d{2}-\d{2})")
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def read_names(data_yaml):
    """Class names from an exported data.yaml (``names:`` then ``  i: name``)."""
    names, in_names = {}, False
    for line in Path(data_yaml).read_text().splitlines():
        if line.strip().startswith("names:"):
            in_names = True
            continue
        m = re.match(r"\s+(\d+):\s*(.+?)\s*$", line)
        if in_names and m:
            names[int(m.group(1))] = m.group(2)
        elif in_names and line and not line[0].isspace():
            break
    return [names[i] for i in sorted(names)]


def group_of(stem):
    m = TITLE.match(stem)
    return (m.group("site"), m.group("day")) if m else ("unknown", "unknown")


def collect(root):
    """``[(image, label_or_None)]`` over every split of an export."""
    pairs = []
    for img in sorted((root / "images").rglob("*")):
        if img.suffix.lower() not in IMAGE_EXT:
            continue
        rel = img.relative_to(root / "images").with_suffix(".txt")
        label = root / "labels" / rel
        pairs.append((img, label if label.exists() else None))
    return pairs


def class_lookup(model_names, data_names, mapping):
    """data class id -> model class id. Raises SystemExit naming any data
    class with no model class to score it against."""
    model_by_name = {n.lower(): i for i, n in model_names.items()}
    data_to_model_name = {d.lower(): m.lower() for m, d in mapping.items()}
    out, missing = {}, []
    for i, name in enumerate(data_names):
        target = data_to_model_name.get(name.lower(), name.lower())
        if target in model_by_name:
            out[i] = model_by_name[target]
        else:
            missing.append(name)
    if missing:
        sys.exit(f"No model class for dataset class(es) {missing}. Model classes: "
                 f"{sorted(model_names.values())}. Use --map model_class=dataset_class "
                 "or --single-cls.")
    return out


def build_subset(pairs, dest, lookup, names_for_yaml):
    """Copy the frames into ``dest`` as one YOLO split with labels rewritten
    to the model's class ids. Returns its data.yaml path."""
    (dest / "images" / "val").mkdir(parents=True)
    (dest / "labels" / "val").mkdir(parents=True)
    for img, label in pairs:
        shutil.copyfile(img, dest / "images" / "val" / img.name)
        lines = []
        if label is not None:
            for line in label.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                parts[0] = str(lookup[cid] if lookup is not None else 0)
                lines.append(" ".join(parts))
        (dest / "labels" / "val" / f"{img.stem}.txt").write_text("\n".join(lines))
    yaml = dest / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in sorted(names_for_yaml.items()))
    yaml.write_text(f"path: {dest}\ntrain: images/val\nval: images/val\n"
                    f"nc: {len(names_for_yaml)}\nnames:\n{names}\n")
    return yaml


def score(model, yaml, args):
    r = model.val(data=str(yaml), split="val", imgsz=args.imgsz, conf=args.conf,
                  iou=args.iou, single_cls=args.single_cls, plots=False, verbose=False)
    return {"mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4),
            "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("weights", help="model .pt file")
    ap.add_argument("export", help="exported project: the .zip or its unzipped folder")
    ap.add_argument("--map", action="append", default=[], metavar="MODEL=DATASET",
                    help="score a model class against a differently named dataset class")
    ap.add_argument("--single-cls", action="store_true", help="ignore classes")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001, help="ultralytics' default for mAP")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU")
    ap.add_argument("--json", help="also write the scores here")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    mapping = {}
    for item in args.map:
        model_cls, sep, data_cls = item.partition("=")
        if not sep:
            sys.exit(f"--map needs MODEL=DATASET, got {item!r}")
        mapping[model_cls.strip()] = data_cls.strip()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = Path(args.export)
        if root.suffix == ".zip":
            with zipfile.ZipFile(root) as z:
                z.extractall(tmp / "export")
            root = tmp / "export"
        data_names = read_names(root / "data.yaml")
        pairs = collect(root)
        if not pairs:
            sys.exit(f"No images under {root / 'images'}")

        model = YOLO(args.weights)
        model_names = dict(model.names)
        if args.single_cls:
            lookup, yaml_names = None, {0: "object"}
        else:
            lookup, yaml_names = class_lookup(model_names, data_names, mapping), model_names

        groups = {"all": pairs}
        by_site, by_day = defaultdict(list), defaultdict(list)
        for img, label in pairs:
            site, day = group_of(img.stem)
            by_site[site].append((img, label))
            by_day[f"{site} {day}"].append((img, label))
        groups.update({f"site {k}": v for k, v in sorted(by_site.items())})
        groups.update({f"day {k}": v for k, v in sorted(by_day.items())})

        results = {}
        for n, (name, subset) in enumerate(groups.items()):
            yaml = build_subset(subset, tmp / f"g{n}", lookup, yaml_names)
            results[name] = {"frames": len(subset), **score(model, yaml, args)}

    print(f"{'group':34} {'frames':>6} {'mAP50':>7} {'mAP50-95':>8} {'P':>6} {'R':>6}")
    for name, r in results.items():
        print(f"{name:34} {r['frames']:6d} {r['mAP50']:7.3f} {r['mAP50_95']:8.3f} "
              f"{r['precision']:6.3f} {r['recall']:6.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"weights": args.weights, "export": args.export, "imgsz": args.imgsz,
             "single_cls": args.single_cls, "map": mapping, "groups": results}, indent=2))


if __name__ == "__main__":
    main()

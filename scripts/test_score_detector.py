#!/usr/bin/env python3
"""score_detector's file handling, without ultralytics (memory/57).

    python3 scripts/test_score_detector.py
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_detector as sd  # noqa: E402


class ScoreDetectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "export"
        for split in ("train", "val"):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
        (self.root / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\nnc: 2\nnames:\n  0: insect\n  1: nest\n")
        for split, stem, label in (("train", "mendels_2024-04-21_09_00_00.mp4_f000010", "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1"),
                                   ("val", "natalies_2024-05-04_12_00_00.mp4_f000020", "")):
            (self.root / "images" / split / f"{stem}.jpg").write_bytes(b"jpg")
            (self.root / "labels" / split / f"{stem}.txt").write_text(label)

    def tearDown(self):
        self.tmp.cleanup()

    def test_names_and_groups(self):
        self.assertEqual(sd.read_names(self.root / "data.yaml"), ["insect", "nest"])
        self.assertEqual(sd.group_of("mendels_2024-04-21_09_00_00.mp4_f000010"), ("mendels", "2024-04-21"))
        self.assertEqual(sd.group_of("odd"), ("unknown", "unknown"))

    def test_every_split_is_scored(self):
        self.assertEqual(len(sd.collect(self.root)), 2)

    def test_classes_map_by_name_or_refuse(self):
        model = {0: "bee", 1: "Nest"}
        self.assertEqual(sd.class_lookup(model, ["insect", "nest"], {"bee": "insect"}), {0: 0, 1: 1})
        with self.assertRaises(SystemExit):
            sd.class_lookup(model, ["insect", "nest"], {})

    def test_labels_rewritten_to_model_ids(self):
        pairs = sd.collect(self.root)
        yaml = sd.build_subset(pairs, Path(self.tmp.name) / "g", {0: 1, 1: 0}, {0: "nest", 1: "insect"})
        labels = sorted((yaml.parent / "labels" / "val").glob("*.txt"))
        self.assertEqual(labels[0].read_text().splitlines()[0].split()[0], "1")
        self.assertEqual(labels[1].read_text(), "")
        single = sd.build_subset(pairs, Path(self.tmp.name) / "s", None, {0: "object"})
        self.assertEqual({l.split()[0] for l in sorted((single.parent / "labels" / "val").glob("*.txt"))[0]
                          .read_text().splitlines()}, {"0"})


if __name__ == "__main__":
    unittest.main()

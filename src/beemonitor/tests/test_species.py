"""BeeMachine species classification + per-track voting.

The model file itself isn't in the repo (89 MB, and it needs an offline ONNX
conversion), so these drive a stub session. What that still pins down is
everything around the model, which is where the bugs live: preprocessing, the
softmax/argmax contract, the class-list mapping, and the voting that turns 900
noisy per-frame readings into one answer per animal.
"""

import json
import pathlib
import unittest

import numpy as np

from beemonitor.identification import species as sp
from beemonitor.identification.species import (
    IMAGE_SIZE, NON_BEE_TAXA, SpeciesIdentifier, SpeciesVote, taxa, taxon_ranks,
)


class StubSession:
    """Stands in for an onnxruntime session; returns a fixed distribution."""

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float32)[None, :]
        self.calls = 0

    def run(self, _outputs, feed):
        self.calls += 1
        self.batch = next(iter(feed.values()))
        return [self.probs]

    def get_inputs(self):
        class _In:
            name = "input_1"
        return [_In()]


def one_hot(index, n=None, peak=0.9):
    n = n or len(taxa())
    probs = np.full(n, (1.0 - peak) / (n - 1), dtype=np.float32)
    probs[index] = peak
    return probs


def crop(size=64, colour=(40, 60, 90)):
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = colour
    return img


class TaxaListTests(unittest.TestCase):
    def test_the_published_class_count(self):
        """354 taxa — the number BeeMachine's own report states."""
        self.assertEqual(len(taxa()), 354)

    def test_names_are_unique_and_ordered(self):
        names = taxa()
        self.assertEqual(len(set(names)), len(names))
        self.assertIsInstance(names, tuple)  # cached + immutable

    def test_asset_records_the_model_it_belongs_to(self):
        """A class list is only meaningful next to the model that produced it."""
        with sp._TAXA_FILE.open() as fh:
            meta = json.load(fh)
        self.assertEqual(meta["image_size"], IMAGE_SIZE)
        self.assertIn("EfficientNetV2S", meta["model"])

    def test_rank_parsing(self):
        self.assertEqual(taxon_ranks("Bombus_impatiens"),
                         {"genus": "Bombus", "species": "Bombus impatiens"})
        self.assertEqual(taxon_ranks("Andrena"), {"genus": "Andrena"})
        self.assertEqual(taxon_ranks("Bombus_vagans_sandersoni")["species"],
                         "Bombus vagans sandersoni")

    def test_non_bee_classes_get_no_genus(self):
        """Syrphidae/Wasp/Diptera are coarse groups, not genera."""
        for name in NON_BEE_TAXA:
            self.assertEqual(taxon_ranks(name), {}, name)
        self.assertTrue(NON_BEE_TAXA <= set(taxa()))


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.identifier = SpeciesIdentifier(session=StubSession(one_hot(0)))

    def test_resizes_to_the_models_input(self):
        batch = self.identifier.preprocess(crop(size=37))
        self.assertEqual(batch.shape, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
        self.assertEqual(batch.dtype, np.float32)

    def test_scales_to_0_1(self):
        """Verified against the model's own reference image: at 0..255 it
        saturates and returns a confidently WRONG taxon. This export carries no
        internal rescaling layer."""
        batch = self.identifier.preprocess(crop(colour=(255, 255, 255)))
        self.assertLessEqual(float(batch.max()), 1.0)
        self.assertAlmostEqual(float(batch.max()), 1.0, delta=0.01)

    def test_converts_bgr_to_rgb(self):
        """OpenCV hands us BGR; the model was trained on RGB."""
        batch = self.identifier.preprocess(crop(colour=(255, 0, 0)))  # blue in BGR
        pixel = batch[0, IMAGE_SIZE // 2, IMAGE_SIZE // 2]
        self.assertAlmostEqual(float(pixel[2]), 1.0, delta=0.01)  # -> blue in RGB
        self.assertAlmostEqual(float(pixel[0]), 0.0, delta=0.01)

    def test_rejects_degenerate_input(self):
        for bad in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((8, 8), np.uint8)):
            self.assertIsNone(self.identifier.preprocess(bad))


class IdentifyTests(unittest.TestCase):
    def test_maps_argmax_to_the_right_taxon(self):
        index = taxa().index("Bombus_impatiens")
        ident = SpeciesIdentifier(session=StubSession(one_hot(index)))

        name, method, confidence = ident.identify(crop())

        self.assertEqual(name, "Bombus_impatiens")
        self.assertEqual(method, "species")
        self.assertAlmostEqual(confidence, 0.9, places=3)

    def test_low_confidence_is_discarded(self):
        """The softmax always picks a winner — the floor is what stops a blurred
        wing being reported as a species."""
        probs = np.full(len(taxa()), 1.0 / len(taxa()), dtype=np.float32)
        ident = SpeciesIdentifier(session=StubSession(probs), min_confidence=0.5)

        self.assertIsNone(ident.identify(crop()))

    def test_logits_are_softmaxed(self):
        """Tolerate an export that left the final activation off."""
        logits = np.full(len(taxa()), -5.0, dtype=np.float32)
        logits[3] = 12.0
        ident = SpeciesIdentifier(session=StubSession(logits))

        name, _method, confidence = ident.identify(crop())

        self.assertEqual(name, taxa()[3])
        self.assertLessEqual(confidence, 1.0)
        self.assertGreater(confidence, 0.9)

    def test_output_size_mismatch_refuses_to_guess(self):
        """A model with a different head must not be silently mapped onto our
        class list — that would invent species."""
        ident = SpeciesIdentifier(session=StubSession(one_hot(0, n=10)))
        self.assertIsNone(ident.identify(crop()))

    def test_bbox_selects_the_region(self):
        session = StubSession(one_hot(0))
        ident = SpeciesIdentifier(session=session)
        frame = np.zeros((100, 200, 3), np.uint8)

        ident.identify(frame, (10, 10, 60, 60))

        self.assertEqual(session.calls, 1)
        self.assertEqual(session.batch.shape, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

    def test_never_raises(self):
        class Exploding:
            def get_inputs(self):
                raise RuntimeError("boom")

            def run(self, *a, **k):
                raise RuntimeError("boom")

        self.assertIsNone(SpeciesIdentifier(session=Exploding()).identify(crop()))

    def test_missing_model_path_does_not_crash_the_frame_loop(self):
        self.assertIsNone(SpeciesIdentifier().identify(crop()))


class BatchAndGatingTests(unittest.TestCase):
    """The two things that make this affordable: batching and skipping."""

    class BatchStub:
        def __init__(self, n_taxa):
            self.n = n_taxa
            self.batch_sizes = []

        def get_inputs(self):
            class _In:
                name = "input_1"
            return [_In()]

        def run(self, _outputs, feed):
            batch = next(iter(feed.values()))
            self.batch_sizes.append(batch.shape[0])
            probs = np.full((batch.shape[0], self.n), (1 - 0.9) / (self.n - 1),
                            dtype=np.float32)
            probs[:, 0] = 0.9
            return [probs]

    def test_batch_is_one_forward_pass_for_many_crops(self):
        session = self.BatchStub(len(taxa()))
        ident = SpeciesIdentifier(session=session)
        frame = np.zeros((200, 400, 3), np.uint8)
        frame[:] = (40, 60, 90)
        boxes = [(0, 0, 60, 60), (80, 0, 140, 60), (200, 0, 260, 60)]

        results = ident.identify_batch(frame, boxes)

        self.assertEqual(len(results), 3)
        self.assertTrue(all(r is not None for r in results))
        self.assertEqual(session.batch_sizes, [3])  # one call, not three

    def test_batch_result_order_matches_input_order(self):
        session = self.BatchStub(len(taxa()))
        ident = SpeciesIdentifier(session=session)
        frame = np.zeros((100, 300, 3), np.uint8)
        frame[:] = (40, 60, 90)
        # Middle box is too small to classify — the gap must land at index 1.
        boxes = [(0, 0, 60, 60), (70, 0, 75, 5), (150, 0, 210, 60)]

        results = ident.identify_batch(frame, boxes)

        self.assertIsNotNone(results[0])
        self.assertIsNone(results[1])
        self.assertIsNotNone(results[2])
        self.assertEqual(session.batch_sizes, [2])  # the tiny crop never ran

    def test_tiny_crops_are_skipped(self):
        """A few pixels upscaled to 300x300 is noise, and a bad vote is worse
        than no vote."""
        ident = SpeciesIdentifier(session=self.BatchStub(len(taxa())),
                                  min_crop_side=24)
        self.assertIsNone(ident.identify(crop(size=8)))
        self.assertIsNotNone(ident.identify(crop(size=40)))

    def test_empty_batch_makes_no_call(self):
        session = self.BatchStub(len(taxa()))
        ident = SpeciesIdentifier(session=session)
        self.assertEqual(ident.identify_batch(np.zeros((50, 50, 3), np.uint8), []), [])
        self.assertEqual(session.batch_sizes, [])


class SpeciesVoteTests(unittest.TestCase):
    def test_majority_wins(self):
        vote = SpeciesVote()
        vote.add("Bombus_impatiens", 0.8)
        vote.add("Apis_mellifera", 0.95)     # single strong outlier
        vote.add("Bombus_impatiens", 0.7)
        vote.add("Bombus_impatiens", 0.75)

        taxon, confidence, votes = vote.winner()

        self.assertEqual(taxon, "Bombus_impatiens")
        self.assertEqual(votes, 3)
        self.assertAlmostEqual(confidence, 0.75, places=3)

    def test_ties_break_on_summed_confidence(self):
        vote = SpeciesVote()
        vote.add("Apis_mellifera", 0.6)
        vote.add("Bombus_impatiens", 0.95)

        self.assertEqual(vote.winner()[0], "Bombus_impatiens")

    def test_empty_vote_has_no_winner(self):
        self.assertIsNone(SpeciesVote().winner())
        self.assertEqual(SpeciesVote().as_dict()["taxon"], None)

    def test_agreement_reports_how_unanimous_the_track_was(self):
        vote = SpeciesVote()
        for _ in range(9):
            vote.add("Bombus_impatiens", 0.9)
        vote.add("Apis_mellifera", 0.9)

        summary = vote.as_dict()

        self.assertEqual(summary["taxon"], "Bombus_impatiens")
        self.assertEqual(summary["taxon_votes"], 9)
        self.assertEqual(summary["taxon_frames"], 10)
        self.assertAlmostEqual(summary["taxon_agreement"], 0.9, places=3)

    def test_blank_readings_are_ignored(self):
        vote = SpeciesVote()
        vote.add("", 0.9)
        vote.add(None, 0.9)
        self.assertIsNone(vote.winner())


class RealModelTests(unittest.TestCase):
    """End-to-end against the converted ONNX + the reference image.

    Skipped unless both are present (neither is committed — 83 MB). This is the
    test that caught the 0..255 preprocessing error, which every stubbed test
    happily passed.
    """

    MODEL = pathlib.Path("models/beemachine/beemachine_v2s_300.onnx")
    IMAGE = pathlib.Path("models/beemachine/test.jpg")

    def setUp(self):
        if not (self.MODEL.exists() and self.IMAGE.exists()):
            self.skipTest("converted model / reference image not present")
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime not installed in this interpreter")

    def test_reference_bumblebee_is_identified_as_a_bumblebee(self):
        import cv2

        ident = SpeciesIdentifier(str(self.MODEL), min_confidence=0.5)
        result = ident.identify(cv2.imread(str(self.IMAGE)))

        self.assertIsNotNone(result, "no reading for the reference image")
        taxon, method, confidence = result
        self.assertTrue(taxon.startswith("Bombus"),
                        f"expected a bumblebee, got {taxon}")
        self.assertEqual(method, "species")
        # Calibrated, not saturated. Exactly 1.0 was the signature of the
        # 0..255 bug.
        self.assertGreater(confidence, 0.5)
        self.assertLess(confidence, 0.999)

    def test_output_width_matches_the_class_list(self):
        import cv2

        ident = SpeciesIdentifier(str(self.MODEL))
        batch = ident.preprocess(cv2.imread(str(self.IMAGE)))
        probs = np.asarray(ident._infer(batch)).reshape(-1)

        self.assertEqual(probs.size, len(taxa()))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()

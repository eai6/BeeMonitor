"""A clip the detector found nothing in is a measurement, not a failure.

The worker only uploads a tracking CSV when it has one to upload, so a clip with
no insects comes back with tracking_csv_path empty. track.mot treated that as a
failed step, which failed the whole run and cascaded "Upstream step failed." to
every analyzer — while the clip's own page said the job completed. On one
58-clip batch that was 18 of the 19 "failures", and the message was absent from
the taxonomy, so the page could only call it "Failed for another reason".
"""

from django.test import SimpleTestCase, TestCase

from apps.pipelines import executors, failures


def _detector(scope="full"):
    return {"id": "d", "block_type": "detect.objects",
            "config": {"run_scope": scope, "classes": ["bee"]}}


class _Run:
    """The two attributes _exec_mot reads off a run."""

    def __init__(self, steps):
        self.steps = steps


def _mot(detector_scope="full", tracking="", with_detector=True):
    detector = _detector(detector_scope)
    steps = ([detector] if with_detector else []) + [
        {"id": "m", "block_type": "track.mot", "config": {},
         "inputs": {"detections": "d"}}]
    result = {"unique_tracks": 0}
    if tracking:
        result["tracking_csv_path"] = tracking
    inputs = {"detections": {"artifact": "detections", "result": result,
                             "job_id": 42}}
    return executors._exec_mot(steps[-1], _Run(steps), {}, inputs, len(steps) - 1)


class EmptyClipTests(SimpleTestCase):
    def test_a_clip_with_nothing_in_it_completes(self):
        out = _mot()

        self.assertNotIn("error", out)
        self.assertEqual(out["artifact"], "tracks")
        self.assertEqual(out["unique_tracks"], 0)
        self.assertTrue(out["empty"])

    def test_it_says_the_clip_is_empty_not_that_the_run_broke(self):
        self.assertIn("found nothing", _mot()["note"])

    def test_the_result_is_carried_through_so_analyzers_still_run(self):
        """Downstream reads result; an empty table must still reach them."""
        out = _mot()

        self.assertIn("result", out)
        self.assertEqual(out["job_id"], 42)

    def test_a_reference_only_detector_is_still_an_error(self):
        """That IS a misconfigured graph, and the message exists to say so."""
        out = _mot(detector_scope="reference_only")

        self.assertIn("error", out)
        self.assertIn("Reference only", out["error"])

    def test_an_uninspectable_graph_keeps_the_error(self):
        """Guessing "empty clip" would hide a misconfiguration behind empty tables."""
        out = _mot(with_detector=False)

        self.assertIn("error", out)

    def test_a_clip_with_tracks_is_unaffected(self):
        out = _mot(tracking="1/j/tracking_results.csv")

        self.assertNotIn("error", out)
        self.assertNotIn("empty", out)

    def test_an_upstream_failure_is_still_a_failure(self):
        steps = [{"id": "m", "block_type": "track.mot", "config": {}}]
        out = executors._exec_mot(steps[0], _Run(steps), {},
                                  {"detections": {"error": "boom"}}, 0)

        self.assertIn("error", out)


class TaxonomyCoverageTests(SimpleTestCase):
    """Every message the system writes should land somewhere better than UNKNOWN."""

    CASES = {
        "No detections to track. The upstream Detector is set to 'Reference only'":
            "no_detections",
        "SageMaker inference failed: RuntimeError(...)": "container_failure",
        "GPU job failed.": "gpu_job_failed",
        "The analysis job for this step no longer exists.": "job_vanished",
        "Amazon SageMaker could not get a response from the endpoint":
            "endpoint_unresponsive",
        "Timed out: no result after 7h": "timed_out",
        "Upstream step failed.": "upstream",
    }

    def test_each_known_message_classifies(self):
        for message, key in self.CASES.items():
            with self.subTest(message=message):
                self.assertEqual(failures.classify(message)["key"], key)

    def test_the_unknown_bucket_asks_to_be_shown(self):
        """Its detail promises the recorded text; the flag is what delivers it."""
        cause = failures.classify("something nobody has seen before")

        self.assertEqual(cause["key"], "unknown")
        self.assertTrue(cause["show_sample"])

    def test_a_classified_cause_does_not_dump_raw_text(self):
        self.assertFalse(failures.classify("Upstream step failed.").get("show_sample"))


class FailurePanelTests(TestCase):
    def test_the_recorded_text_reaches_the_group(self):
        groups = failures.group([(7, "totally novel explosion", None)])

        self.assertEqual(groups[0]["sample"], "totally novel explosion")
        self.assertTrue(groups[0]["cause"]["show_sample"])

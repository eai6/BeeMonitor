"""Job cost — priced from the hardware the run reports.

The bug these pin: cost used to be `execution_seconds x GPU_TIERS[gpu_tier]`,
where gpu_tier was a dropdown that never reached SageMaker. Picking "A100" on a
T4 endpoint billed 2.85x. And the arithmetic was copy-pasted into two views, so
a fix in one left the other stale.
"""

from django.test import TestCase

from apps.analysis import pricing


class InstanceResolutionTests(TestCase):
    def test_reported_gpu_names_map_to_their_instances(self):
        for device, expected in [
            ("cuda:Tesla T4", "ml.g4dn.xlarge"),
            ("cuda:NVIDIA A10G", "ml.g5.xlarge"),
            ("cuda:NVIDIA L4", "ml.g6.xlarge"),
            ("cuda:NVIDIA L40S", "ml.g6e.xlarge"),
            ("cuda:NVIDIA A100-SXM4-40GB", "ml.p4d.24xlarge"),
        ]:
            self.assertEqual(pricing.instance_for_device(device), expected, device)

    def test_l40s_is_not_swallowed_by_the_l4_rule(self):
        """Substring matching has to try the longer name first."""
        self.assertEqual(pricing.instance_for_device("cuda:NVIDIA L40S"), "ml.g6e.xlarge")

    def test_unknown_or_missing_device_falls_back_to_the_endpoint_instance(self):
        for device in (None, "", "cpu", "cuda:Some Future GPU"):
            self.assertEqual(pricing.instance_for_device(device), pricing.DEFAULT_INSTANCE)

    def test_the_fallback_is_the_cheap_instance_not_the_dear_one(self):
        """An unknown device must not bill as if it were an A100."""
        rates = [pricing.rate_per_second(i) for i in pricing.INSTANCE_HOURLY_USD]
        self.assertEqual(pricing.rate_per_second(pricing.DEFAULT_INSTANCE), min(rates))

    def test_every_instance_has_a_tier_label(self):
        for instance in pricing.INSTANCE_HOURLY_USD:
            self.assertIn(instance, pricing.INSTANCE_TIER)


class PriceRunTests(TestCase):
    def test_the_september_job_is_priced_at_the_t4_rate(self):
        # The run that displayed $0.0935: 305.6 s billed at the A10G rate while
        # executing on a T4. Correctly priced it is ~$0.0625.
        priced = pricing.price_run(
            {"execution_seconds": 305.6, "device": "cuda:Tesla T4"})

        self.assertEqual(priced["compute_cost_usd"], 0.0625)
        self.assertEqual(priced["gpu_tier"], "T4")
        self.assertEqual(priced["instance_type"], "ml.g4dn.xlarge")

    def test_the_tier_describes_the_hardware_not_a_choice(self):
        priced = pricing.price_run(
            {"execution_seconds": 10, "device": "cuda:NVIDIA A10G"})

        self.assertEqual(priced["gpu_tier"], "A10G")

    def test_gpu_seconds_and_stages_pass_through(self):
        priced = pricing.price_run({
            "execution_seconds": 100,
            "gpu_seconds": 42.5,
            "stage_seconds": {"decode": {"seconds": 30.0, "calls": 900}},
            "device": "cuda:Tesla T4",
        })

        self.assertEqual(priced["gpu_seconds"], 42.5)
        self.assertEqual(priced["stage_seconds"]["decode"]["calls"], 900)

    def test_billing_uses_handler_time_not_gpu_time(self):
        """gpu_seconds is the diagnostic; the instance is busy for all of it."""
        priced = pricing.price_run(
            {"execution_seconds": 100, "gpu_seconds": 10, "device": "cuda:Tesla T4"})

        self.assertEqual(
            priced["compute_cost_usd"],
            round(100 * pricing.rate_per_second("ml.g4dn.xlarge"), 4))

    def test_a_missing_result_prices_to_zero_rather_than_raising(self):
        priced = pricing.price_run({})

        self.assertEqual(priced["compute_cost_usd"], 0.0)
        self.assertEqual(priced["credits"], 0)
        self.assertEqual(priced["stage_seconds"], {})

    def test_credits_track_gpu_seconds(self):
        self.assertEqual(pricing.price_run({"execution_seconds": 305.6})["credits"], 305)


class SingleSourceOfTruthTests(TestCase):
    def test_both_completion_paths_call_price_run(self):
        """analysis/views.py and api/views.py had verbatim copies of the math;
        a fix to one silently left the API reporting the old number."""
        from pathlib import Path

        import apps.analysis.views as analysis_views
        import apps.api.views as api_views

        for module in (analysis_views, api_views):
            source = Path(module.__file__).read_text()
            self.assertIn("price_run(", source, module.__name__)
            self.assertNotIn("cost_per_sec", source, f"{module.__name__} still prices inline")

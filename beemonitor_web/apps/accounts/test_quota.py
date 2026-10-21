"""GPU credits: the arithmetic, the gate at the entry points, and charging once.

memory/50 §3 and §5. The property that matters most: a launch that doesn't fit
is refused before anything is created, so no batch is ever half-started.
"""

from datetime import date
from decimal import Decimal
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.accounts import quota
from apps.accounts.models import Coupon, UsageEvent, UserProfile
from apps.analysis.models import Job
from apps.videos.models import Video

User = get_user_model()
T4, A10G = quota.VIDEO_INSTANCE, quota.SAM3_INSTANCE


class CreditsTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("bee", password="x")
        self.profile, _ = UserProfile.objects.get_or_create(user=self.user)

    def clip(self, seconds=600, n=0):
        return Video.objects.create(
            user=self.user, title=f"c{n}", storage_key=f"k/{n}.mp4", file_size_bytes=1,
            status=Video.Status.READY, duration_seconds=seconds)

    def grant(self, credits):
        UserProfile.objects.filter(user=self.user).update(granted_gpu_credits=Decimal(credits))


class ArithmeticTests(CreditsTestCase):
    def test_an_hour_on_each_machine(self):
        self.assertEqual(quota.credits_for(3600, T4), Decimal("1"))
        self.assertEqual(quota.credits_for(3600, A10G), Decimal("2"))
        self.assertEqual(quota.credits_for(1800, "ml.unknown"), Decimal("0.5"))  # priced as the video machine

    def test_a_new_account_has_ten_free_credits(self):
        self.assertEqual(quota.available(self.user), Decimal("10"))

    def test_granted_credits_add_to_the_free_ones(self):
        self.grant(5)
        self.assertEqual(quota.available(self.user), Decimal("15"))

    def test_a_new_month_renews_the_free_credits_but_keeps_granted(self):
        UserProfile.objects.filter(user=self.user).update(
            credit_period_start=date(2020, 1, 1), monthly_gpu_credits_used=10,
            granted_gpu_credits=3)
        self.assertEqual(quota.available(self.user), Decimal("13"))
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.credit_period_start, timezone.localdate().replace(day=1))

    def test_estimates_use_recent_runs_once_there_are_enough(self):
        self.assertEqual(quota.seconds_per_unit("analysis", T4), 1.07)  # fallback, no history
        for i in range(quota.MIN_HISTORY):
            UsageEvent.objects.create(user=self.user, kind="analysis", gpu_seconds=1200,
                                      instance_type=T4, credits=0, units=600,
                                      source_type="t", source_id=str(i))
        self.assertEqual(quota.seconds_per_unit("analysis", T4), 2.0)


class ChargeTests(CreditsTestCase):
    def test_free_credits_are_spent_before_granted_ones(self):
        self.grant(5)
        quota.charge(self.user, "analysis", 12 * 3600, T4, "job", 1)   # 12 credits
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.monthly_gpu_credits_used, Decimal("10"))
        self.assertEqual(self.profile.granted_gpu_credits, Decimal("3"))
        self.assertEqual(quota.available(self.user), Decimal("3"))

    def test_an_overrun_can_go_negative(self):
        quota.charge(self.user, "analysis", 11 * 3600, T4, "job", 1)
        self.assertEqual(quota.available(self.user), Decimal("-1"))

    def test_the_same_work_is_charged_once(self):
        quota.charge(self.user, "analysis", 3600, T4, "job", 7)
        quota.charge(self.user, "analysis", 3600, T4, "job", 7)
        self.assertEqual(UsageEvent.objects.filter(source_id="7").count(), 1)
        self.assertEqual(quota.available(self.user), Decimal("9"))

    def test_nothing_ran_nothing_charged(self):
        self.assertIsNone(quota.charge(self.user, "analysis", 0, T4, "job", 8))
        self.assertFalse(UsageEvent.objects.exists())

    def test_superusers_are_recorded_but_never_run_out(self):
        self.user.is_superuser = True
        self.user.save()
        quota.charge(self.user, "analysis", 50 * 3600, T4, "job", 9)
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.granted_gpu_credits, Decimal("0"))
        self.assertEqual(UsageEvent.objects.get().credits, Decimal("50"))
        quota.check(self.user, Decimal("1000"))   # never raises


class CheckTests(CreditsTestCase):
    def test_refuses_what_does_not_fit(self):
        with self.assertRaises(quota.QuotaExceeded) as cm:
            quota.check(self.user, [quota.Work("analysis", T4, 11 * 3600 / 1.07)])
        self.assertIn("Nothing was started", cm.exception.message)

    def test_work_already_queued_is_set_aside(self):
        Job.objects.create(user=self.user, video=self.clip(9 * 3600 / 1.07),
                           status=Job.Status.QUEUED, modal_job_id="q1")
        self.assertAlmostEqual(float(quota.pending(self.user)), 9.0, places=2)
        with self.assertRaises(quota.QuotaExceeded):
            quota.check(self.user, [quota.Work("analysis", T4, 2 * 3600 / 1.07)])

    def test_pro_is_never_refused(self):
        UserProfile.objects.filter(user=self.user).update(tier="pro")
        quota.check(self.user, Decimal("10000"))

    def test_sam3_costs_twice_as_much_per_hour(self):
        sam = quota.estimate([quota.Work("analysis", A10G, 600)])
        self.assertEqual(sam, quota.credits_for(600 * quota.FALLBACK_SECONDS_PER_UNIT[("analysis", A10G)], A10G))


class EntryPointTests(CreditsTestCase):
    """Out of credits, every launch path refuses and creates nothing."""

    def setUp(self):
        super().setUp()
        self.client.force_login(self.user)
        # 100 hours of footage: ~107 credits, far over 10.
        self.big = self.clip(seconds=100 * 3600)

    def test_batch_analysis_refuses_with_402(self):
        r = self.client.post(reverse("analysis:batch"), {"video_ids": [self.big.pk]},
                             HTTP_X_REQUESTED_WITH="XMLHttpRequest")
        self.assertEqual(r.status_code, 402)
        self.assertIn("Not enough credits", r.json()["error"])
        self.assertFalse(Job.objects.exists())

    def test_pipeline_launch_refuses_and_starts_no_runs(self):
        from apps.pipelines.models import Pipeline, PipelineRun
        pipeline = Pipeline.objects.create(user=self.user, title="p", steps=[
            {"id": "v", "block_type": "input.video", "config": {}},
            {"id": "d", "block_type": "detect.bee", "config": {}, "inputs": {"video": "v"}},
        ])
        r = self.client.post(reverse("pipelines:run_on_videos"),
                             {"pipeline": pipeline.pk, "video_ids": [self.big.pk]},
                             HTTP_X_REQUESTED_WITH="XMLHttpRequest")
        self.assertEqual(r.status_code, 402)
        self.assertFalse(PipelineRun.objects.exists())
        self.assertFalse(Job.objects.exists())

    def test_a_clip_that_fits_still_runs(self):
        small = self.clip(seconds=600, n=1)
        with mock.patch("apps.analysis.views._spawn_gpu_batch"):
            r = self.client.post(reverse("analysis:batch"), {"video_ids": [small.pk]},
                                 HTTP_X_REQUESTED_WITH="XMLHttpRequest")
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(Job.objects.count(), 1)

    def test_api_job_create_refuses_with_402(self):
        from rest_framework.test import APIClient
        api = APIClient()
        api.force_authenticate(self.user)
        r = api.post("/api/v1/jobs/", {"video_id": self.big.pk}, format="json")
        self.assertEqual(r.status_code, 402, r.content)
        self.assertFalse(Job.objects.exists())


class CouponTests(CreditsTestCase):
    def test_a_credit_coupon_grants_credits_that_survive_the_month(self):
        Coupon.objects.create(code="FIELD", coupon_type="credits", credits_amount=20)
        self.client.force_login(self.user)
        self.client.post(reverse("accounts:redeem_coupon"), {"coupon_code": "field"})
        self.assertEqual(quota.available(self.user), Decimal("30"))


class EnsureSuperusersTests(TestCase):
    def promote(self, raw):
        from django.core.management import call_command
        with mock.patch.dict("os.environ", {"BEEMONITOR_SUPERUSERS": raw}):
            call_command("ensure_superusers")

    def test_username_and_email_must_both_match(self):
        User.objects.create_user("manager", email="owner@example.com", password="x")
        self.promote("manager:someone-else@example.com, ghost:g@example.com, manager")
        self.assertFalse(User.objects.get(username="manager").is_superuser)
        self.promote("manager:OWNER@example.com")
        u = User.objects.get(username="manager")
        self.assertTrue(u.is_superuser and u.is_staff)
        self.assertTrue(quota.is_unlimited(u))


class UsagePageTests(CreditsTestCase):
    def test_gpu_time_is_shown_in_hours(self):
        quota.charge(self.user, "analysis", 5400, T4, "job", 1)   # 1.5 h
        self.client.force_login(self.user)
        r = self.client.get(reverse("accounts:usage"))
        self.assertContains(r, "1.5 GPU-hours")
        self.assertNotContains(r, "5400s")

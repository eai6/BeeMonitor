"""Raw footage retention: nothing is deleted without three notices that were
actually sent, and nothing that should be kept is touched (memory/50 §6).

Time is passed in (``today``), so a test walks one clip from its first notice
to its deletion 30 days later.
"""

from datetime import timedelta
from unittest import mock

from django.contrib.auth import get_user_model
from django.core import mail
from django.test import TestCase, override_settings
from django.utils import timezone

from apps.accounts.models import AccountRequest, RetentionNotice, UserProfile
from apps.videos import retention
from apps.videos.models import Video

User = get_user_model()
LOCMEM = "django.core.mail.backends.locmem.EmailBackend"


@override_settings(EMAIL_BACKEND=LOCMEM, RAW_FOOTAGE_DELETION="on")
class RetentionTests(TestCase):
    def setUp(self):
        self.today = timezone.localdate()
        self.alice = User.objects.create_user("alice", email="alice@lab.edu")
        self.old = self.clip(self.alice, days_old=100)
        self.new = self.clip(self.alice, days_old=10)
        self.s3 = mock.Mock()
        patcher = mock.patch("config.storage.get_s3_client", return_value=self.s3)
        patcher.start()
        self.addCleanup(patcher.stop)

    def clip(self, user, days_old, key=None):
        v = Video.objects.create(user=user, title="c.mp4", storage_key=key or f"{user.pk}/{days_old}/c.mp4",
                                 file_size_bytes=2_000_000_000, status=Video.Status.READY)
        Video.everything.filter(pk=v.pk).update(uploaded_at=timezone.now() - timedelta(days=days_old))
        return Video.everything.get(pk=v.pk)

    def day(self, n):
        return retention.run(self.today + timedelta(days=n))

    def test_old_footage_gets_three_notices_then_goes_after_thirty_days(self):
        self.day(0)
        self.old.refresh_from_db()
        self.assertEqual(self.old.raw_delete_on, self.today + timedelta(days=30))
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("will be deleted on", mail.outbox[0].subject)
        self.assertIsNone(Video.everything.get(pk=self.new.pk).raw_delete_on)  # 10 days old: not yet
        for n in range(1, 30):
            self.day(n)
            self.assertIsNone(Video.everything.get(pk=self.old.pk).raw_deleted_at, n)
        self.assertEqual(sorted(RetentionNotice.objects.values_list("days_before", flat=True)), [7, 15, 30])
        self.day(30)
        self.old.refresh_from_db()
        self.assertIsNotNone(self.old.raw_deleted_at)
        self.s3.delete_blob.assert_called_once_with("raw-videos", self.old.storage_key)
        self.assertIsNone(Video.everything.get(pk=self.new.pk).raw_deleted_at)

    def test_no_email_means_no_deletion(self):
        with mock.patch("django.core.mail.EmailMultiAlternatives.send", side_effect=Exception("MessageRejected")):
            for n in range(0, 45):
                self.day(n)
        self.assertIsNone(Video.everything.get(pk=self.old.pk).raw_deleted_at)
        self.assertFalse(RetentionNotice.objects.filter(status="sent").exists())
        self.s3.delete_blob.assert_not_called()

    def test_a_late_notice_pushes_deletion_back(self):
        self.day(0)
        with mock.patch("django.core.mail.EmailMultiAlternatives.send", side_effect=Exception("down")):
            for n in range(1, 26):
                self.day(n)   # the 15- and 7-day notices fail
        for n in range(26, 33):
            self.day(n)       # they go out on day 26; 7 days later is day 33
        self.assertIsNone(Video.everything.get(pk=self.old.pk).raw_deleted_at)
        self.day(41)
        self.assertIsNotNone(Video.everything.get(pk=self.old.pk).raw_deleted_at)

    def test_pro_admins_and_waiting_storage_requests_keep_everything(self):
        pro = User.objects.create_user("pro", email="p@x.org")
        UserProfile.objects.filter(user=pro).update(tier="pro")
        boss = User.objects.create_superuser("boss", "b@x.org", "pw-12345678")
        keep = [self.clip(pro, 300), self.clip(boss, 300)]
        AccountRequest.objects.create(user=self.alice, kind="storage", contact_email="a@x.org",
                                      until=self.today + timedelta(days=400))
        for n in range(0, 60):
            self.day(n)
        for v in keep + [self.old]:
            v.refresh_from_db()
            self.assertIsNone(v.raw_delete_on)
            self.assertIsNone(v.raw_deleted_at)
        self.s3.delete_blob.assert_not_called()

    def test_approved_storage_cancels_a_schedule(self):
        self.day(0)
        UserProfile.objects.filter(user=self.alice).update(raw_retention_until=self.today + timedelta(days=200))
        self.day(1)
        self.assertIsNone(Video.everything.get(pk=self.old.pk).raw_delete_on)
        for n in range(2, 60):
            self.day(n)
        self.assertIsNone(Video.everything.get(pk=self.old.pk).raw_deleted_at)

    def test_footage_in_an_external_bucket_is_never_touched(self):
        ext = self.clip(self.alice, 300, key="s3://their-bucket/c.mp4")
        for n in range(0, 40):
            self.day(n)
        self.assertIsNone(Video.everything.get(pk=ext.pk).raw_delete_on)

    def test_a_deleted_clip_cannot_be_analysed_again(self):
        Video.everything.filter(pk=self.old.pk).update(raw_deleted_at=timezone.now())
        self.assertNotIn(self.old.pk, Video.manageable(self.alice).values_list("pk", flat=True))
        self.assertIn(self.old.pk, Video.accessible(self.alice).values_list("pk", flat=True))

    def test_the_account_page_shows_what_is_scheduled(self):
        self.day(0)
        self.client.force_login(self.alice)
        r = self.client.get("/accounts/usage/")
        self.assertContains(r, "will be deleted on")
        self.assertEqual(r.context["deletions"][0]["clips"], 1)


@override_settings(EMAIL_BACKEND=LOCMEM, RAW_FOOTAGE_DELETION="dry_run")
class DryRunTests(TestCase):
    def test_a_dry_run_reports_and_changes_nothing(self):
        alice = User.objects.create_user("alice", email="alice@lab.edu")
        v = Video.objects.create(user=alice, title="c", storage_key="k/c.mp4", file_size_bytes=10)
        Video.everything.filter(pk=v.pk).update(uploaded_at=timezone.now() - timedelta(days=100))
        with mock.patch("config.storage.get_s3_client") as s3:
            retention.run()
        v.refresh_from_db()
        self.assertIsNone(v.raw_delete_on)
        self.assertEqual(len(mail.outbox), 0)
        s3.assert_not_called()
        (row,) = retention.preview()
        self.assertEqual((row["user"], row["clips"]), (alice, 1))
        admin = User.objects.create_superuser("boss", "b@x.org", "pw-12345678")
        self.client.force_login(admin)
        self.assertContains(self.client.get("/platform/"), "Dry run: nothing is sent or deleted")


@override_settings(EMAIL_BACKEND=LOCMEM)
class DailyTests(TestCase):
    def test_each_task_runs_once_a_day(self):
        from apps.accounts.daily import run_daily
        today = timezone.localdate()
        self.assertEqual(set(run_daily(today)), {"raw_footage", "pro_reminders"})
        self.assertEqual(run_daily(today), {})
        self.assertEqual(set(run_daily(today + timedelta(days=1))), {"raw_footage", "pro_reminders"})

    def test_pro_is_reminded_once_a_month_before_it_ends(self):
        from apps.accounts.daily import pro_reminders
        bee = User.objects.create_user("bee", email="bee@lab.edu")
        UserProfile.objects.filter(user=bee).update(tier="pro", tier_expires_at=timezone.now() + timedelta(days=20))
        self.assertEqual(pro_reminders(), 1)
        self.assertEqual(pro_reminders(), 0)
        self.assertIn("Pro plan ends", mail.outbox[0].subject)

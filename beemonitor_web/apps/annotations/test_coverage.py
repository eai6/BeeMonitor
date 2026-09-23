"""A lopsided annotation sample must be visible before it is annotated.

The picker was a checkbox list of filenames, so a project drawn entirely from
one hotel at one hour looked exactly like a project that had covered
everything. These pin the instrument that tells them apart.
"""

from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.annotations import coverage
from apps.annotations.models import AnnotationProject
from apps.devices.models import Device
from apps.videos.models import Video

User = get_user_model()


class CoverageTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("cov", password="x")
        self.a = Device.objects.create(owner=self.user, name="Hotel A",
                                       key_hash="ha", prefix="bmk_a")
        self.b = Device.objects.create(owner=self.user, name="Hotel B",
                                       key_hash="hb", prefix="bmk_b")
        self.project = AnnotationProject.objects.create(user=self.user, name="P")
        self.n = 0

    def clip(self, device, hour, *, in_project=False):
        self.n += 1
        when = timezone.now().replace(hour=hour, minute=0) - timedelta(days=self.n % 3)
        v = Video.objects.create(
            user=self.user, device=device, title=f"c{self.n}",
            storage_key=f"cov/{self.n}.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=when)
        v.hour = hour
        v.save(update_fields=["hour"])
        if in_project:
            self.project.videos.add(v)
        return v

    def build(self):
        from apps.videos import workspace
        devices = [self.a, self.b]
        rows = workspace.device_rows(devices, [], {})
        return coverage.build(Video.accessible(self.user), self.project.videos.all(),
                              devices, workspace.dots_by_device(rows))


class CoverageGridTests(CoverageTestCase):
    def test_a_cell_with_footage_and_no_labels_is_a_gap(self):
        self.clip(self.a, 9)                      # available, not annotated

        grid = self.build()

        cell = next(c for c in grid["rows"][0]["cells"] if c["hour"] == 9)
        self.assertEqual(cell["state"], "gap")
        self.assertEqual(cell["available"], 1)

    def test_a_cell_with_no_footage_is_not_a_gap(self):
        """Absence of footage is not a hole in the sample — only unlabelled
        footage is, and conflating them makes the map cry wolf."""
        self.clip(self.a, 9, in_project=True)

        grid = self.build()

        row = grid["rows"][1]                      # Hotel B recorded nothing
        self.assertTrue(all(c["state"] == "none" for c in row["cells"]))
        self.assertEqual(grid["gaps"], 0)

    def test_a_hotel_contributing_nothing_is_named(self):
        self.clip(self.a, 9, in_project=True)
        for _ in range(4):
            self.clip(self.b, 10)

        findings = self.build()["findings"]

        self.assertTrue(any("Hotel B contributes nothing" in f for f in findings), findings)

    def test_an_hour_nobody_annotated_is_named(self):
        self.clip(self.a, 9, in_project=True)
        for _ in range(12):
            self.clip(self.b, 15)

        findings = self.build()["findings"]

        self.assertTrue(any("15:00" in f for f in findings), findings)

    def test_a_single_hour_holding_most_of_the_project_is_named(self):
        for _ in range(9):
            self.clip(self.a, 9, in_project=True)
        self.clip(self.b, 14, in_project=True)

        findings = self.build()["findings"]

        self.assertTrue(any("over a third" in f for f in findings), findings)

    def test_a_balanced_project_says_nothing_alarming(self):
        for hour in (8, 10, 12, 14):
            self.clip(self.a, hour, in_project=True)
            self.clip(self.b, hour, in_project=True)

        grid = self.build()

        self.assertEqual(grid["findings"], [])
        self.assertEqual(grid["gaps"], 0)

    def test_hours_span_the_day_they_cover_without_padding_it(self):
        self.clip(self.a, 9, in_project=True)
        self.clip(self.a, 11, in_project=True)

        self.assertEqual(self.build()["hours"], [9, 10, 11])

    def test_an_empty_library_has_no_map_rather_than_an_empty_one(self):
        self.assertIsNone(self.build())


class DraftTests(CoverageTestCase):
    def test_it_picks_only_cells_the_project_is_missing(self):
        self.clip(self.a, 9, in_project=True)
        missing = self.clip(self.b, 9)

        picks = coverage.draft(Video.accessible(self.user), self.project.videos.all())

        self.assertIn(missing.pk, [p["id"] for p in picks])

    def test_it_takes_at_most_the_requested_number_per_cell(self):
        for _ in range(6):
            self.clip(self.a, 9)

        picks = coverage.draft(Video.accessible(self.user), self.project.videos.all(),
                               per_cell=2)

        self.assertEqual(len(picks), 2)

    def test_it_spreads_across_cells_rather_than_filling_one(self):
        for _ in range(5):
            self.clip(self.a, 9)
        for _ in range(5):
            self.clip(self.b, 15)

        picks = coverage.draft(Video.accessible(self.user), self.project.videos.all(),
                               per_cell=2)
        hours = set(Video.objects.filter(pk__in=[p["id"] for p in picks]).values_list("device_id", flat=True))

        self.assertEqual(len(picks), 4)
        self.assertEqual(hours, {self.a.id, self.b.id})

    def test_a_fully_covered_project_drafts_nothing(self):
        self.clip(self.a, 9, in_project=True)

        self.assertEqual(
            coverage.draft(Video.accessible(self.user), self.project.videos.all()), [])


class WorkspacePageTests(CoverageTestCase):
    def setUp(self):
        super().setUp()
        self.client.force_login(self.user)

    def test_the_picker_renders_the_shared_rail_and_grid(self):
        self.clip(self.a, 9)

        html = self.client.get(reverse("annotations:add_videos_page",
                                       args=[self.project.pk])).content.decode()

        self.assertIn("Bee hotels", html)        # from videos/_review_filters
        self.assertIn("vid-card", html)          # from videos/_review_grid
        self.assertIn("Coverage", html)

    def test_clips_already_added_stay_visible_and_marked(self):
        """Hiding them is what makes over-sampling one hotel invisible."""
        self.clip(self.a, 9, in_project=True)

        html = self.client.get(reverse("annotations:add_videos_page",
                                       args=[self.project.pk])).content.decode()

        self.assertIn("in project", html)

    def test_the_membership_filter_can_still_narrow_to_unadded(self):
        added = self.clip(self.a, 9, in_project=True)
        self.clip(self.b, 10)

        html = self.client.get(
            reverse("annotations:add_videos_page", args=[self.project.pk]),
            {"member": "out"}).content.decode()

        self.assertNotIn(f'name="video_ids" value="{added.pk}"', html)

    def test_the_shared_filter_understands_hour_windows(self):
        """The old hand-rolled filter had no concept of these at all."""
        early = self.clip(self.a, 7)
        late = self.clip(self.a, 19)

        html = self.client.get(
            reverse("annotations:add_videos_page", args=[self.project.pk]),
            {"hfrom": "6", "hto": "12"}).content.decode()

        self.assertIn(f'name="video_ids" value="{early.pk}"', html)
        self.assertNotIn(f'name="video_ids" value="{late.pk}"', html)

    def test_the_draft_endpoint_returns_ids(self):
        self.clip(self.a, 9)

        resp = self.client.get(reverse("annotations:add_videos_draft",
                                       args=[self.project.pk]))

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["count"], 1)

    def test_another_users_project_is_not_reachable(self):
        other = User.objects.create_user("nosy2", password="x")
        self.client.force_login(other)

        resp = self.client.get(reverse("annotations:add_videos_page",
                                       args=[self.project.pk]))

        self.assertEqual(resp.status_code, 404)

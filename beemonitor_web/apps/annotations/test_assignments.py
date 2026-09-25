"""Handing frames out for review, and taking them back.

Assignment is per frame (memory/39): SAM 3 labels the frames, and "give eai7
500" is the unit a manager thinks in. It is the difference between "you can
reach this project" and "these are yours to check".
"""

from datetime import datetime, timedelta, timezone as dt_tz

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.annotations import assignments
from apps.annotations.models import Annotation, AnnotationProject, ProjectShare
from apps.videos.models import Video

User = get_user_model()
BOX = [{"x": 1, "y": 1, "w": 5, "h": 5, "class": "bee"}]


class AssignmentTestCase(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("owner", password="x")
        self.project = AnnotationProject.objects.create(user=self.owner, name="P")
        self.jill = User.objects.create_user("jill", password="x")
        self.kwame = User.objects.create_user("kwame", password="x")
        for u in (self.jill, self.kwame):
            ProjectShare.objects.create(project=self.project, user=u,
                                        role="reviewer", created_by=self.owner)
        self.n = 0

    def clip(self, frames=3, day=0):
        self.n += 1
        v = Video.objects.create(
            user=self.owner, title=f"c{self.n}", storage_key=f"as/{self.n}.mp4",
            file_size_bytes=1, status=Video.Status.READY,
            recorded_at=datetime(2026, 7, 1, tzinfo=dt_tz.utc) + timedelta(days=day))
        self.project.videos.add(v)
        for i in range(frames):
            Annotation.objects.create(project=self.project, video=v,
                                      frame_number=i * 10, boxes=BOX)
        return v

    def frames(self):
        return Annotation.objects.filter(project=self.project)


class PickTests(AssignmentTestCase):
    def test_spread_takes_a_frame_from_each_clip_before_a_second(self):
        clips = [self.clip(frames=5, day=d) for d in range(4)]
        ids = assignments.pick(self.frames(), 4)
        self.assertEqual(sorted(Annotation.objects.filter(pk__in=ids)
                                .values_list("video_id", flat=True)),
                         sorted(c.pk for c in clips))

    def test_spread_covers_the_whole_period_not_its_start(self):
        """5 frames from 50 clips: evenly spaced clips, not the first five."""
        for d in range(50):
            self.clip(frames=1, day=d)
        days = sorted(a.video.recorded_at.day + 30 * (a.video.recorded_at.month - 7)
                      for a in Annotation.objects.filter(
                          pk__in=assignments.pick(self.frames(), 5)).select_related("video"))
        self.assertGreater(days[-1] - days[0], 30)

    def test_whole_clips_keeps_a_clip_together(self):
        first = self.clip(frames=4, day=0)
        self.clip(frames=4, day=1)
        ids = assignments.pick(self.frames(), 4, order="clips")
        self.assertEqual(set(Annotation.objects.filter(pk__in=ids)
                             .values_list("video_id", flat=True)), {first.pk})

    def test_asking_for_more_than_exist_gives_what_there_is(self):
        self.clip(frames=3)
        self.assertEqual(len(assignments.pick(self.frames(), 100)), 3)


class AssignTests(AssignmentTestCase):
    def test_assigning_gives_the_frames_to_one_person(self):
        self.clip(frames=4)
        done = assignments.assign(self.project, self.frames().values_list("pk", flat=True),
                                  self.jill, by=self.owner)
        self.assertEqual(done, 4)
        self.assertEqual(self.frames().filter(assigned_to=self.jill).count(), 4)
        self.assertEqual(self.frames().first().assigned_by, self.owner)

    def test_a_held_frame_is_not_handed_out_twice(self):
        """Two managers assigning at once must not both get the same frame."""
        self.clip(frames=2)
        ids = list(self.frames().values_list("pk", flat=True))
        assignments.assign(self.project, ids[:1], self.jill)
        self.assertEqual(assignments.assign(self.project, ids, self.kwame), 1)
        self.assertEqual(self.frames().filter(assigned_to=self.jill).count(), 1)

    def test_reviewed_frames_are_never_handed_out(self):
        self.clip(frames=2)
        self.frames().filter(frame_number=0).update(reviewed=True)
        ids = self.frames().values_list("pk", flat=True)
        self.assertEqual(assignments.assign(self.project, ids, self.jill), 1)

    def test_release_returns_only_unreviewed_frames(self):
        self.clip(frames=3)
        self.frames().update(assigned_to=self.jill)
        self.frames().filter(frame_number=0).update(reviewed=True)
        self.assertEqual(assignments.release(self.project, self.jill), 2)
        self.assertEqual(self.frames().filter(assigned_to=self.jill).count(), 1)


class WorkloadTests(AssignmentTestCase):
    def test_a_workload_counts_assigned_reviewed_and_left(self):
        self.clip(frames=4)
        self.frames().update(assigned_to=self.jill)
        self.frames().filter(frame_number__in=[0, 10]).update(reviewed=True)
        [w] = assignments.workloads(self.project)
        self.assertEqual((w["username"], w["assigned"], w["reviewed"], w["left"], w["pct"]),
                         ("jill", 4, 2, 2, 50))

    def test_the_pool_is_unreviewed_and_unheld(self):
        self.clip(frames=3)
        self.frames().filter(frame_number=0).update(assigned_to=self.jill)
        self.frames().filter(frame_number=10).update(reviewed=True)
        self.assertEqual(assignments.unassigned_count(self.project), 1)


class AssignViewTests(AssignmentTestCase):
    def post(self, user, **data):
        self.client.force_login(user)
        return self.client.post(reverse("annotations:assign_frames", args=[self.project.pk]), data)

    def test_a_manager_gives_n_frames_from_the_filter(self):
        for d in range(3):
            self.clip(frames=4, day=d)
        self.post(self.owner, reviewer=self.jill.pk, count=5, order="spread", status="review")
        self.assertEqual(self.frames().filter(assigned_to=self.jill).count(), 5)
        # Spread: every clip got some.
        self.assertEqual(self.frames().filter(assigned_to=self.jill)
                         .values("video_id").distinct().count(), 3)

    def test_the_filter_limits_the_pool(self):
        self.clip(frames=2)
        wasp = self.clip(frames=0)
        Annotation.objects.create(project=self.project, video=wasp, frame_number=0,
                                  boxes=[{"x": 1, "y": 1, "w": 2, "h": 2, "class": "wasp"}])
        self.post(self.owner, reviewer=self.jill.pk, count=10, cls="wasp")
        self.assertEqual(list(self.frames().filter(assigned_to=self.jill)
                              .values_list("video_id", flat=True)), [wasp.pk])

    def test_only_a_manager_may_assign(self):
        self.clip(frames=2)
        resp = self.post(self.jill, reviewer=self.jill.pk, count=2)
        self.assertEqual(resp.status_code, 404)
        self.assertFalse(self.frames().filter(assigned_to__isnull=False).exists())

    def test_assigning_to_somebody_not_on_the_project_is_refused(self):
        self.clip(frames=2)
        stranger = User.objects.create_user("stranger", password="x")
        self.post(self.owner, reviewer=stranger.pk, count=2)
        self.assertFalse(self.frames().filter(assigned_to__isnull=False).exists())

    def test_a_viewer_is_not_a_reviewer(self):
        self.clip(frames=2)
        viewer = User.objects.create_user("viewer", password="x")
        ProjectShare.objects.create(project=self.project, user=viewer, role="viewer")
        self.post(self.owner, reviewer=viewer.pk, count=2)
        self.assertFalse(self.frames().filter(assigned_to__isnull=False).exists())

    def test_a_manager_can_take_frames_back(self):
        self.clip(frames=2)
        self.frames().update(assigned_to=self.jill)
        self.post(self.owner, reviewer=self.jill.pk, release=1)
        self.assertFalse(self.frames().filter(assigned_to__isnull=False).exists())


class TakeViewTests(AssignmentTestCase):
    def test_a_reviewer_takes_frames_from_the_pool(self):
        self.clip(frames=3)
        self.frames().filter(frame_number=0).update(assigned_to=self.kwame)
        self.client.force_login(self.jill)
        self.client.post(reverse("annotations:take_frames", args=[self.project.pk]), {"count": 100})
        self.assertEqual(self.frames().filter(assigned_to=self.jill).count(), 2)
        self.assertEqual(self.frames().filter(assigned_to=self.kwame).count(), 1)

    def test_a_reviewer_gives_their_own_back(self):
        self.clip(frames=2)
        self.frames().update(assigned_to=self.jill)
        self.client.force_login(self.jill)
        self.client.post(reverse("annotations:take_frames", args=[self.project.pk]), {"release": 1})
        self.assertFalse(self.frames().filter(assigned_to__isnull=False).exists())

    def test_a_viewer_cannot_take_frames(self):
        self.clip(frames=2)
        viewer = User.objects.create_user("viewer", password="x")
        ProjectShare.objects.create(project=self.project, user=viewer, role="viewer")
        self.client.force_login(viewer)
        resp = self.client.post(reverse("annotations:take_frames", args=[self.project.pk]))
        self.assertEqual(resp.status_code, 404)


class MigrationTests(TestCase):
    """0012 copies each clip assignment onto that clip's unreviewed frames."""

    def test_clip_assignments_become_frame_assignments(self):
        from django.apps import apps as django_apps

        from apps.annotations.migrations import __name__ as pkg
        import importlib
        mod = importlib.import_module(pkg + ".0012_frame_assignment")
        from apps.annotations.models import ClipAssignment

        owner = User.objects.create_user("o", password="x")
        jill = User.objects.create_user("j", password="x")
        project = AnnotationProject.objects.create(user=owner, name="P")
        v = Video.objects.create(user=owner, title="c", storage_key="m/c.mp4",
                                 file_size_bytes=1, status=Video.Status.READY)
        project.videos.add(v)
        todo = Annotation.objects.create(project=project, video=v, frame_number=0, boxes=BOX)
        done = Annotation.objects.create(project=project, video=v, frame_number=1,
                                         boxes=BOX, reviewed=True)
        ClipAssignment.objects.create(project=project, video=v, user=jill, assigned_by=owner)

        mod.clips_to_frames(django_apps, None)

        todo.refresh_from_db()
        done.refresh_from_db()
        self.assertEqual((todo.assigned_to, todo.assigned_by), (jill, owner))
        self.assertIsNone(done.assigned_to)

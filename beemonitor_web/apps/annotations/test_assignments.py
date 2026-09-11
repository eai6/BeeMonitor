"""Handing work out, and taking it back.

Assignment is the difference between "you can reach this project" and "this is
yours to do". Without it a shared project is a room full of people all looking
at the same 59 clips.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.annotations import assignments
from apps.annotations.models import (Annotation, AnnotationProject,
                                     ClipAssignment, ProjectShare)
from apps.videos.models import Video

User = get_user_model()


class AssignmentTestCase(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("owner", password="x")
        self.project = AnnotationProject.objects.create(user=self.owner, name="P")
        self.jill = User.objects.create_user("jill", password="x")
        self.kwame = User.objects.create_user("kwame", password="x")
        for u in (self.jill, self.kwame):
            ProjectShare.objects.create(project=self.project, user=u,
                                        role="annotator", created_by=self.owner)
        self.n = 0

    def clip(self, frames=0, labelled=0):
        self.n += 1
        v = Video.objects.create(user=self.owner, title=f"c{self.n}",
                                 storage_key=f"as/{self.n}.mp4", file_size_bytes=1,
                                 status=Video.Status.READY)
        self.project.videos.add(v)
        for i in range(frames):
            Annotation.objects.create(project=self.project, video=v, frame_number=i,
                                      boxes=[{"x": 1}] if i < labelled else [])
        return v


class HandingOutTests(AssignmentTestCase):
    def test_assigning_gives_the_clips_to_one_person(self):
        clips = [self.clip(), self.clip()]

        assignments.assign(self.project, [c.pk for c in clips], self.jill,
                           by=self.owner)

        self.assertEqual(self.project.assigned_to(self.jill).count(), 2)

    def test_reassigning_moves_a_clip_rather_than_refusing(self):
        """A clip has one owner of the work; silently refusing would leave the
        page disagreeing with the database."""
        c = self.clip()
        assignments.assign(self.project, [c.pk], self.jill, by=self.owner)

        assignments.assign(self.project, [c.pk], self.kwame, by=self.owner)

        self.assertEqual(ClipAssignment.objects.get(video=c).user, self.kwame)
        self.assertEqual(self.project.assigned_to(self.jill).count(), 0)

    def test_distributing_deals_them_out_evenly(self):
        clips = [self.clip() for _ in range(6)]

        tally = assignments.distribute(self.project, [c.pk for c in clips],
                                       [self.jill, self.kwame], by=self.owner)

        self.assertEqual(tally[self.jill.id], 3)
        self.assertEqual(tally[self.kwame.id], 3)

    def test_an_odd_number_splits_as_evenly_as_it_can(self):
        clips = [self.clip() for _ in range(7)]

        tally = assignments.distribute(self.project, [c.pk for c in clips],
                                       [self.jill, self.kwame])

        self.assertEqual(sorted(tally.values()), [3, 4])

    def test_distributing_to_nobody_does_nothing(self):
        c = self.clip()

        self.assertEqual(assignments.distribute(self.project, [c.pk], []), {})
        self.assertFalse(ClipAssignment.objects.exists())


class PoolTests(AssignmentTestCase):
    def test_unassigned_clips_are_the_pool(self):
        self.clip()
        taken = self.clip()
        assignments.assign(self.project, [taken.pk], self.jill)

        pool = assignments.unassigned(self.project)

        self.assertEqual(pool.count(), 1)
        self.assertNotIn(taken, pool)

    def test_claiming_takes_a_clip_from_the_pool(self):
        c = self.clip()

        self.assertTrue(assignments.claim(self.project, c.pk, self.jill))
        self.assertTrue(self.project.may_annotate_video(self.jill, c.pk))

    def test_a_self_claim_records_that_nobody_gave_it_to_them(self):
        c = self.clip()
        assignments.claim(self.project, c.pk, self.jill)

        self.assertTrue(ClipAssignment.objects.get(video=c).self_claimed)

    def test_claiming_someone_elses_clip_is_refused(self):
        """Claiming is for the pool, not a way around an assignment."""
        c = self.clip()
        assignments.assign(self.project, [c.pk], self.kwame, by=self.owner)

        self.assertFalse(assignments.claim(self.project, c.pk, self.jill))
        self.assertEqual(ClipAssignment.objects.get(video=c).user, self.kwame)

    def test_re_claiming_your_own_clip_is_harmless(self):
        c = self.clip()
        assignments.claim(self.project, c.pk, self.jill)

        self.assertTrue(assignments.claim(self.project, c.pk, self.jill))

    def test_you_can_put_your_own_clip_back(self):
        c = self.clip()
        assignments.claim(self.project, c.pk, self.jill)

        self.assertTrue(assignments.release(self.project, c.pk, self.jill))
        self.assertEqual(assignments.unassigned(self.project).count(), 1)

    def test_you_cannot_put_back_someone_elses(self):
        c = self.clip()
        assignments.assign(self.project, [c.pk], self.kwame)

        self.assertFalse(assignments.release(self.project, c.pk, self.jill))

    def test_a_manager_can_take_any_clip_back(self):
        boss = User.objects.create_user("boss", password="x")
        ProjectShare.objects.create(project=self.project, user=boss, role="manager")
        c = self.clip()
        assignments.assign(self.project, [c.pk], self.kwame)

        self.assertTrue(assignments.release(self.project, c.pk, boss))


class WorkloadTests(AssignmentTestCase):
    def test_a_workload_counts_frames_not_just_clips(self):
        """"12 clips" says nothing about whether the work is nearly done."""
        # 12 of 16 — deliberately not a .5 tie, so the test is about the
        # count and not about which way round() breaks one.
        a = self.clip(frames=10, labelled=6)
        b = self.clip(frames=6, labelled=6)
        assignments.assign(self.project, [a.pk, b.pk], self.jill)

        row = next(w for w in assignments.workloads(self.project)
                   if w["username"] == "jill")

        self.assertEqual(row["clips"], 2)
        self.assertEqual(row["frames"], 16)
        self.assertEqual(row["labelled"], 12)
        self.assertEqual(row["pct"], 75)

    def test_everyone_holding_work_appears(self):
        assignments.assign(self.project, [self.clip().pk], self.jill)
        assignments.assign(self.project, [self.clip().pk], self.kwame)

        names = {w["username"] for w in assignments.workloads(self.project)}

        self.assertEqual(names, {"jill", "kwame"})

    def test_somebody_with_nothing_assigned_is_not_listed(self):
        assignments.assign(self.project, [self.clip().pk], self.jill)

        names = {w["username"] for w in assignments.workloads(self.project)}

        self.assertNotIn("kwame", names)

    def test_an_unstarted_workload_reads_zero_rather_than_erroring(self):
        assignments.assign(self.project, [self.clip().pk], self.jill)

        row = assignments.workloads(self.project)[0]

        self.assertEqual((row["frames"], row["labelled"], row["pct"]), (0, 0, 0))

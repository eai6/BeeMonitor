"""GPU sampling (SAMPLING_BACKEND=sagemaker): claim into batches, invoke,
collect pre-labelled frames — safely across processes, cancels, failures and
retries. S3 and SageMaker are faked in-process."""

import json
from datetime import timedelta
from unittest import mock

from botocore.exceptions import ClientError
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from apps.annotations import sampling, sampling_remote as sr
from apps.annotations.models import Annotation, AnnotationProject, FrameSamplingTask, SamplingBatch
from apps.videos.models import Video

User = get_user_model()
GPU = dict(SAMPLING_BACKEND="sagemaker", SAGEMAKER_SAM3_ENDPOINT_NAME="sam3",
           SAGEMAKER_INPUT_BUCKET="in", SAGEMAKER_OUTPUT_BUCKET="out",
           SAMPLING_BATCH_CLIPS=2, SAMPLING_MAX_BATCHES_IN_FLIGHT=5, SAMPLING_CANDIDATES=15)


class FakeAWS:
    """put/get objects in memory; invoke records the request."""
    def __init__(self):
        self.objects, self.invokes = {}, []

    def client(self, name, **kw):
        return self

    def put_object(self, Bucket, Key, Body, **kw):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body = self.objects[(Bucket, Key)]
        return {"Body": mock.Mock(read=lambda: body if isinstance(body, bytes) else body.encode())}

    def invoke_endpoint_async(self, **kw):
        self.invokes.append(kw)
        return {"OutputLocation": f"s3://out/async/{kw['InferenceId']}.out",
                "FailureLocation": f"s3://out/failures/{kw['InferenceId']}.failure"}

    def payload(self, batch):
        return json.loads(self.objects[("in", f"sampling/{batch.batch_id}.json")])

    def finish(self, batch, clips):
        self.objects[("out", batch.result_key)] = json.dumps({"clips": clips}).encode()


@override_settings(**GPU)
class GpuSamplingTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("g", password="x")
        self.project = AnnotationProject.objects.create(user=self.user, name="P",
                                                        classes=["bee", "wasp", "nest"])
        self.videos = []
        for i in range(3):
            v = Video.objects.create(user=self.user, title=f"c{i}", storage_key=f"u/c{i}.mp4",
                                     file_size_bytes=1, status=Video.Status.READY)
            self.project.videos.add(v)
            self.videos.append(v)
        self.aws = FakeAWS()
        self.patch = mock.patch("boto3.client", side_effect=self.aws.client)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def queue(self, videos=None, **kw):
        params = sr.params_for(kw.pop("classes", ["bee"]), **kw)
        return [FrameSamplingTask.objects.create(user=self.user, project=self.project, video=v,
                                                 params=params) for v in (videos or self.videos)]

    def clip(self, task, frames=((40, [("bee", 0.8)]),), **extra):
        return {"task_id": task.pk, "motion": {"profile": [0, 100], "picked": [1], "frames": 80},
                "seconds": {"scan_and_detect": 12.5},
                "frames": [{"n": n, "key": f"frames/k/f{n:06d}.jpg", "w": 1920, "h": 1080,
                            "boxes": [{"x": 10, "y": 20, "w": 30, "h": 30, "class": c, "confidence": p}
                                      for c, p in boxes]} for n, boxes in frames], **extra}

    def test_dispatch_claims_disjoint_batches_and_sends_the_layout(self):
        tasks = self.queue()
        self.assertEqual(sr.dispatch(), 2)                       # 3 clips, 2 per batch
        batches = list(SamplingBatch.objects.order_by("pk"))
        self.assertEqual([b.status for b in batches], ["invoked", "invoked"])
        ids = [t.batch_id for t in FrameSamplingTask.objects.filter(pk__in=[t.pk for t in tasks])]
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual(sorted(len([i for i in ids if i == b.pk]) for b in batches), [1, 2])
        p = self.aws.payload(batches[0])
        self.assertEqual((p["task"], p["classes"], p["candidates"]), ("sample_label", ["bee"], 15))
        self.assertEqual(p["clips"][0]["video_blob_path"], "u/c0.mp4")
        inv = self.aws.invokes[0]
        self.assertEqual((inv["EndpointName"], inv["InferenceId"]), ("sam3", batches[0].batch_id))
        self.assertIn("RequestTTLSeconds", inv)
        self.assertEqual(sr.dispatch(), 0)                       # nothing left to claim

    def test_external_clips_fail_before_they_reach_the_gpu(self):
        self.videos[0].storage_key = "s3://elsewhere/x.mp4"
        self.videos[0].save()
        self.queue(self.videos[:1])
        sr.dispatch()
        t = FrameSamplingTask.objects.get()
        self.assertEqual(t.status, "failed")
        self.assertIn("external bucket", t.error_message)
        self.assertEqual(self.aws.invokes, [])

    def test_collect_writes_pre_labelled_frames_once(self):
        tasks = self.queue(self.videos[:1])
        sr.dispatch()
        batch = SamplingBatch.objects.get()
        self.aws.finish(batch, [self.clip(tasks[0], frames=((40, [("bee", 0.8), ("wasp", 0.5)]),))])
        with mock.patch("apps.accounts.models.UserProfile.charge") as charge:
            self.assertEqual(sr.collect(), 1)
            self.assertEqual(sr.collect(), 0)                    # already collected
        a = Annotation.objects.get()
        self.assertEqual(a.frame_number, 40)
        self.assertEqual([(b["class"], b["class_id"]) for b in a.boxes], [("bee", 0), ("wasp", 1)])
        self.assertFalse(a.sampled_only)
        t = FrameSamplingTask.objects.get()
        self.assertEqual((t.status, t.frames_written), ("completed", 1))
        self.assertEqual(t.motion["picked"], [1])
        self.assertEqual(SamplingBatch.objects.get().status, "collected")
        charge.assert_called_once()
        self.assertEqual(charge.call_args.kwargs["gpu_seconds"], 12.5)

    def test_reviewed_frames_keep_their_boxes_and_replace_drops_only_untouched_frames(self):
        v = self.videos[0]
        human = Annotation.objects.create(project=self.project, video=v, frame_number=40,
                                          boxes=[{"class": "wasp", "class_id": 1}], reviewed=True)
        empty = Annotation.objects.create(project=self.project, video=v, frame_number=3,
                                          boxes=[], sampled_only=True)
        tasks = self.queue([v])
        sr.dispatch()
        self.aws.finish(SamplingBatch.objects.get(), [self.clip(tasks[0])])
        with mock.patch("apps.accounts.models.UserProfile.charge"):
            sr.collect()
        human.refresh_from_db()
        self.assertEqual(human.boxes, [{"class": "wasp", "class_id": 1}])
        self.assertFalse(Annotation.objects.filter(pk=empty.pk).exists())

    def test_a_cancelled_task_gets_no_frames(self):
        tasks = self.queue(self.videos[:1])
        sr.dispatch()
        FrameSamplingTask.objects.filter(pk=tasks[0].pk).update(status="cancelled")
        self.aws.finish(SamplingBatch.objects.get(), [self.clip(tasks[0])])
        sr.collect()
        self.assertFalse(Annotation.objects.exists())
        self.assertEqual(SamplingBatch.objects.get().status, "collected")

    def test_a_clip_error_fails_only_that_clip_and_archived_reads_clearly(self):
        tasks = self.queue(self.videos[:2])
        sr.dispatch()
        self.aws.finish(SamplingBatch.objects.get(), [
            self.clip(tasks[0]),
            {"task_id": tasks[1].pk, "frames": [], "error": "ClientError: InvalidObjectState"}])
        with mock.patch("apps.accounts.models.UserProfile.charge"):
            sr.collect()
        ok, bad = FrameSamplingTask.objects.get(pk=tasks[0].pk), FrameSamplingTask.objects.get(pk=tasks[1].pk)
        self.assertEqual((ok.status, bad.status), ("completed", "failed"))
        self.assertEqual(bad.error_message, sr.ARCHIVED_MESSAGE)

    def test_a_failed_batch_retries_clips_alone_then_gives_up(self):
        tasks = self.queue(self.videos[:2])
        sr.dispatch()
        b1 = SamplingBatch.objects.get()
        self.aws.objects[("out", f"failures/{b1.batch_id}.failure")] = b"ModelError: out of memory"
        sr.collect()
        self.assertEqual(SamplingBatch.objects.get(pk=b1.pk).status, "failed")
        again = FrameSamplingTask.objects.filter(pk__in=[t.pk for t in tasks])
        self.assertTrue(all(t.status == "queued" and t.attempts == 1 and t.params["solo"] for t in again))
        sr.dispatch()                                            # solo -> one batch per clip
        solo = SamplingBatch.objects.exclude(pk=b1.pk)
        self.assertEqual(solo.count(), 2)
        for b in solo:
            self.aws.objects[("out", f"failures/{b.batch_id}.failure")] = b"ModelError: again"
        sr.collect()
        self.assertTrue(all(t.status == "failed" for t in FrameSamplingTask.objects.all()))

    def test_a_batch_claimed_but_never_invoked_is_requeued(self):
        tasks = self.queue(self.videos[:1])
        with mock.patch.object(sr, "_invoke", side_effect=RuntimeError("network")):
            sr.dispatch()
        batch = SamplingBatch.objects.get()
        self.assertEqual(batch.status, "claimed")
        SamplingBatch.objects.filter(pk=batch.pk).update(claimed_at=timezone.now() - timedelta(minutes=11))
        self.assertEqual(sr.recover(), 1)
        self.assertEqual(FrameSamplingTask.objects.get(pk=tasks[0].pk).status, "queued")

    def test_the_local_pool_never_runs_gpu_tasks(self):
        self.queue(self.videos[:1])
        with mock.patch.object(sampling, "spawn_sampling_async") as spawn:
            sampling.poll_frame_sampling_tasks()
        spawn.assert_not_called()
        self.assertEqual(sampling.run_sampling_task(FrameSamplingTask.objects.get().pk), 0)
        self.assertEqual(FrameSamplingTask.objects.get().status, "queued")

    def test_adding_clips_queues_gpu_tasks_for_the_picked_classes(self):
        v = Video.objects.create(user=self.user, title="new", storage_key="u/new.mp4",
                                 file_size_bytes=1, status=Video.Status.READY)
        self.client.force_login(self.user)
        with mock.patch.object(sampling, "spawn_sampling_async") as spawn:
            self.client.post(reverse("annotations:add_videos", args=[self.project.pk]),
                             {"video_ids": [v.pk], "s_classes": ["bee", "wasp"]})
        spawn.assert_not_called()
        t = FrameSamplingTask.objects.get(video=v)
        self.assertEqual((t.status, t.params["method"], t.params["classes"]), ("queued", "sample_label", ["bee", "wasp"]))

    def test_resampling_needs_classes_and_supersedes_queued_work(self):
        old = self.queue(self.videos[:1])[0]
        self.client.force_login(self.user)
        url = reverse("annotations:sample_frames", args=[self.project.pk])
        self.client.post(url, {"video_ids": [self.videos[0].pk]})
        self.assertEqual(FrameSamplingTask.objects.count(), 1)   # no classes -> nothing queued
        self.client.post(url, {"video_ids": [self.videos[0].pk], "s_classes": ["wasp"], "s_max_frames": "7"})
        old.refresh_from_db()
        self.assertEqual(old.status, "cancelled")
        new = FrameSamplingTask.objects.exclude(pk=old.pk).get()
        self.assertEqual((new.params["classes"], new.params["max_frames"]), (["wasp"], 7))

    def test_project_page_shows_gpu_mode_and_status(self):
        self.queue(self.videos[:1])
        self.client.force_login(self.user)
        html = self.client.get(reverse("annotations:detail", args=[self.project.pk]),
                               {"tab": "clips"}).content.decode()
        self.assertIn('name="s_classes"', html)
        self.assertIn("1 queued", html)
        self.assertIn("Cancel sampling", html)


class NoActivityStageTests(TestCase):
    """A clip whose sampling finished with no frames is done — "No activity" —
    not "Not sampled" (which would invite sampling it again and again)."""

    def test_sampled_but_empty_clips_are_not_counted_as_unsampled(self):
        from apps.annotations import progress
        user = User.objects.create_user("n", password="x")
        project = AnnotationProject.objects.create(user=user, name="P", classes=["bee"])
        vids = []
        for i in range(3):
            v = Video.objects.create(user=user, title=f"c{i}", storage_key=f"u/{i}.mp4",
                                     file_size_bytes=1, status=Video.Status.READY)
            project.videos.add(v)
            vids.append(v)
        Annotation.objects.create(project=project, video=vids[0], frame_number=1, boxes=[])
        FrameSamplingTask.objects.create(user=user, project=project, video=vids[1],
                                         status="completed", frames_written=0)
        states = progress.per_video(project)
        summary = progress.summary(project, states, 3)
        self.assertEqual((summary["clips_unsampled"], summary["clips_empty"]), (1, 1))
        self.assertEqual(summary["stage_counts"]["new"], 1)
        self.assertEqual(states[vids[1].pk]["stage_label"], "No activity")

        self.client.force_login(user)
        html = self.client.get(reverse("annotations:detail", args=[project.pk]) + "?tab=clips&stage=new").content.decode()
        self.assertIn("c2", html)
        self.assertNotIn(">c1<", html)
        self.assertIn("1 no activity", self.client.get(reverse("annotations:detail", args=[project.pk])).content.decode())

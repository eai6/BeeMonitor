"""A model's card describes the data the model actually saw.

The source project keeps growing. If the card read it live, every model would
silently restate its coverage each time someone added a clip — a model trained
on two hotels would start claiming four. So the provenance is a snapshot taken
at training time and never touched again.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.annotations.models import Annotation, AnnotationProject
from apps.devices.models import Device
from apps.training.models import TrainingJob
from apps.training.views import _training_snapshot
from apps.videos.models import Video

User = get_user_model()


class SnapshotTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("tp", password="x")
        self.device = Device.objects.create(owner=self.user, name="A",
                                            key_hash="tp", prefix="bmk_tp")
        self.project = AnnotationProject.objects.create(
            user=self.user, name="Flower UV", classes=["bee", "flower"])
        self.n = 0

    def clip(self, frames=2, labelled=2, hour=9, device=None):
        self.n += 1
        v = Video.objects.create(user=self.user, device=device or self.device,
                                 title=f"c{self.n}", storage_key=f"tp/{self.n}.mp4",
                                 file_size_bytes=1, status=Video.Status.READY)
        v.hour = hour
        v.save(update_fields=["hour"])
        self.project.videos.add(v)
        for i in range(frames):
            Annotation.objects.create(project=self.project, video=v, frame_number=i,
                                      boxes=[{"label": "bee"}] if i < labelled else [])
        return v

    def job(self, subset=None):
        return TrainingJob.objects.create(
            user=self.user, project=self.project, name="run",
            class_subset=subset or [])

    def test_it_records_what_the_project_held(self):
        self.clip(frames=3, labelled=2)
        self.clip(frames=2, labelled=2)

        snap = _training_snapshot(self.job())

        self.assertEqual(snap["project"], "Flower UV")
        self.assertEqual(snap["frames"], 4)
        self.assertEqual(snap["clips"], 2)
        self.assertEqual(snap["devices"], 1)

    def test_it_records_how_varied_the_data_was(self):
        second = Device.objects.create(owner=self.user, name="B",
                                       key_hash="tp2", prefix="bmk_tp2")
        self.clip(hour=9)
        self.clip(hour=15, device=second)

        snap = _training_snapshot(self.job())

        self.assertEqual(snap["devices"], 2)
        self.assertEqual(snap["hours"], [9, 15])

    def test_a_subset_job_records_the_classes_it_trained_on(self):
        self.clip()

        snap = _training_snapshot(self.job(subset=["bee"]))

        self.assertEqual(snap["classes"], ["bee"])

    def test_unlabelled_frames_are_not_counted_as_training_data(self):
        self.clip(frames=4, labelled=1)

        self.assertEqual(_training_snapshot(self.job())["frames"], 1)

    def test_the_snapshot_does_not_change_when_the_project_grows(self):
        """The whole reason it is a snapshot."""
        self.clip(frames=2, labelled=2)
        snap = _training_snapshot(self.job())

        self.clip(frames=5, labelled=5)

        self.assertEqual(snap["frames"], 2)
        self.assertEqual(_training_snapshot(self.job())["frames"], 7)

    def test_a_job_with_no_project_yields_nothing_rather_than_raising(self):
        """TrainingJob.project is non-nullable, so this state cannot be built in
        the database — but the finaliser runs on whatever it is handed, and
        returning {} beats raising inside a completion hook."""
        class Orphan:
            project = None

        self.assertEqual(_training_snapshot(Orphan()), {})


class PublicModelUseTests(TestCase):
    """Publishing a model is only meaningful if someone else can select it."""

    def setUp(self):
        from apps.training.models import CustomModel

        self.owner = User.objects.create_user("mo", password="x")
        self.other = User.objects.create_user("mu", password="x")
        self.model = CustomModel.objects.create(
            user=self.owner, name="Bee detector v3", model_type="bee_tracking",
            base_model="yolov8n", storage_key="models/bee-v3.pt",
            classes=["bee"], status=CustomModel.Status.READY, is_active=True)

    def test_your_own_model_is_usable(self):
        from apps.training.models import CustomModel

        self.assertIn(self.model, CustomModel.usable(self.owner))

    def test_someone_elses_private_model_is_not(self):
        from apps.training.models import CustomModel

        self.assertNotIn(self.model, CustomModel.usable(self.other))

    def test_a_published_model_is_usable_by_anyone(self):
        from apps.training.models import CustomModel

        self.model.visibility = CustomModel.Visibility.PUBLIC
        self.model.save(update_fields=["visibility"])

        self.assertIn(self.model, CustomModel.usable(self.other))

    def test_a_deactivated_model_is_not_offered_even_when_public(self):
        """Deactivating is how you withdraw a model that turned out bad."""
        from apps.training.models import CustomModel

        self.model.visibility = CustomModel.Visibility.PUBLIC
        self.model.is_active = False
        self.model.save(update_fields=["visibility", "is_active"])

        self.assertNotIn(self.model, CustomModel.usable(self.other))

    def test_a_model_with_no_weights_is_never_offered(self):
        from apps.training.models import CustomModel

        self.model.visibility = CustomModel.Visibility.PUBLIC
        self.model.storage_key = ""
        self.model.save(update_fields=["visibility", "storage_key"])

        self.assertNotIn(self.model, CustomModel.usable(self.other))


class PublishModelViewTests(TestCase):
    def setUp(self):
        from apps.training.models import CustomModel

        self.owner = User.objects.create_user("pm", password="x")
        self.other = User.objects.create_user("pn", password="x")
        self.model = CustomModel.objects.create(
            user=self.owner, name="v3", model_type="bee_tracking",
            base_model="yolov8n", storage_key="m/v3.pt", classes=["bee"],
            status=CustomModel.Status.READY, is_active=True)

    def url(self):
        from django.urls import reverse
        return reverse("training:model_publish", args=[self.model.pk])

    def test_the_owner_can_publish_and_withdraw(self):
        from apps.training.models import CustomModel

        self.client.force_login(self.owner)

        self.client.post(self.url(), {"visibility": "public"})
        self.assertEqual(CustomModel.objects.get(pk=self.model.pk).visibility,
                         "public")

        self.client.post(self.url(), {"visibility": "private"})
        self.assertEqual(CustomModel.objects.get(pk=self.model.pk).visibility,
                         "private")

    def test_nobody_else_can(self):
        from apps.training.models import CustomModel

        self.client.force_login(self.other)

        resp = self.client.post(self.url(), {"visibility": "public"})

        self.assertEqual(resp.status_code, 404)
        self.assertEqual(CustomModel.objects.get(pk=self.model.pk).visibility,
                         "private")

    def test_withdrawing_keeps_the_publication_date(self):
        """Republishing later should not claim the model is newer than it is."""
        from apps.training.models import CustomModel

        self.client.force_login(self.owner)
        self.client.post(self.url(), {"visibility": "public"})
        first = CustomModel.objects.get(pk=self.model.pk).published_at

        self.client.post(self.url(), {"visibility": "private"})
        self.client.post(self.url(), {"visibility": "public"})

        self.assertEqual(CustomModel.objects.get(pk=self.model.pk).published_at,
                         first)

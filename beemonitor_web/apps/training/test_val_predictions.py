"""A model page must not render a gallery of images that are no longer there.

The training container dropped its rendered val-set predictions in the
SageMaker output bucket, and that bucket expires EVERYTHING after 7 days — it
exists for transient request/result JSON. The keys live in TrainingJob.metrics
forever, so any model older than a week rendered tiles whose every image 404s,
and the browser drew the alt text: each frame appeared as its filename twice.
"""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from apps.annotations.models import AnnotationProject
from apps.training.models import TrainingJob
from apps.training.views import _val_prediction_images

User = get_user_model()

MODELS = "bm-models"
OUTPUT = "bm-sm-output"
NEW = [f"custom/7/12/val_preds/f{i}.jpg" for i in range(3)]
OLD = [f"training/12/val_preds/f{i}.jpg" for i in range(3)]


class FakeS3:
    """Lists only the keys it was told exist, and signs anything."""

    def __init__(self, present=()):
        self.present = list(present)
        self.listed = []

    def list_objects_v2(self, **kw):
        self.listed.append((kw["Bucket"], kw["Prefix"]))
        contents = [{"Key": k} for (b, k) in self.present
                    if b == kw["Bucket"] and k.startswith(kw["Prefix"])]
        return {"Contents": contents}

    def generate_presigned_url(self, op, Params, ExpiresIn):  # noqa: N803
        return f"https://signed.test/{Params['Bucket']}/{Params['Key']}"


@override_settings(AWS_S3_BUCKET_MODELS=MODELS, SAGEMAKER_OUTPUT_BUCKET=OUTPUT)
class ValPredictionResolutionTests(TestCase):
    def _run(self, keys, present):
        fake = FakeS3(present)
        with patch("apps.training.views._boto3", return_value=fake):
            return _val_prediction_images(keys, job_pk=12) + (fake,)

    def test_frames_that_exist_are_presigned_from_the_models_bucket(self):
        images, expired, _ = self._run(NEW, [(MODELS, k) for k in NEW])

        self.assertFalse(expired)
        self.assertEqual([i["name"] for i in images], ["f0.jpg", "f1.jpg", "f2.jpg"])
        self.assertTrue(images[0]["url"].startswith(f"https://signed.test/{MODELS}/"))

    def test_legacy_keys_are_read_from_the_output_bucket(self):
        """Jobs trained before the move still render, while their week lasts."""
        images, expired, _ = self._run(OLD, [(OUTPUT, k) for k in OLD])

        self.assertFalse(expired)
        self.assertTrue(images[0]["url"].startswith(f"https://signed.test/{OUTPUT}/"))

    def test_expired_frames_yield_no_tiles_and_say_so(self):
        images, expired, _ = self._run(OLD, present=[])

        self.assertEqual(images, [])
        self.assertTrue(expired)

    def test_a_partly_swept_prefix_shows_only_what_survived(self):
        images, expired, _ = self._run(OLD, [(OUTPUT, OLD[1])])

        self.assertEqual([i["name"] for i in images], ["f1.jpg"])
        self.assertFalse(expired)

    def test_it_lists_once_per_folder_not_once_per_frame(self):
        """60 frames per job share one prefix; 60 HEADs would be absurd."""
        _images, _expired, fake = self._run(NEW, [(MODELS, k) for k in NEW])

        self.assertEqual(fake.listed, [(MODELS, "custom/7/12/val_preds/")])

    def test_an_s3_failure_is_unknown_rather_than_expired(self):
        """Claiming the frames expired because S3 hiccuped would be a lie."""
        with patch("apps.training.views._boto3", side_effect=RuntimeError("boom")):
            images, expired = _val_prediction_images(NEW, job_pk=12)

        self.assertEqual(images, [])
        self.assertFalse(expired)


@override_settings(AWS_S3_BUCKET_MODELS=MODELS, SAGEMAKER_OUTPUT_BUCKET=OUTPUT)
class ValPredictionPageTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("vp", password="x")
        self.client.force_login(self.user)
        project = AnnotationProject.objects.create(
            user=self.user, name="P", classes=["bee"])
        self.job = TrainingJob.objects.create(
            user=self.user, project=project, name="m",
            status=TrainingJob.Status.COMPLETED,
            metrics={"mAP50": 0.8, "val_predictions": OLD})

    def _html(self, present):
        with patch("apps.training.views._boto3", return_value=FakeS3(present)):
            return self.client.get(
                reverse("training:detail", kwargs={"pk": self.job.pk})
            ).content.decode()

    def test_a_swept_gallery_explains_itself_instead_of_breaking(self):
        html = self._html(present=[])

        self.assertIn("no longer", html)
        self.assertNotIn("f0.jpg", html)

    def test_a_live_gallery_still_renders_the_frames(self):
        html = self._html([(OUTPUT, k) for k in OLD])

        self.assertIn("f0.jpg", html)
        self.assertNotIn("no longer", html)

"""The training page's numbers must agree with each other.

The cards show best.pt re-tested after training; the chart legend used to
show the last epoch and the best epoch was picked by mAP50, while YOLO keeps
the epoch with the best 0.1 * mAP50 + 0.9 * mAP50-95. So "best epoch 31" and
the legend could both describe a different model than the one kept.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.annotations.models import AnnotationProject
from apps.training.models import TrainingJob
from apps.training.views import _build_epoch_charts, best_epoch_index

User = get_user_model()

EM = [
    {"epoch": 1, "mAP50": 0.70, "mAP50_95": 0.30, "precision": 0.6, "recall": 0.6},
    # Highest mAP50, but not the kept model.
    {"epoch": 2, "mAP50": 0.82, "mAP50_95": 0.38, "precision": 0.7, "recall": 0.8},
    # Highest fitness: this is best.pt.
    {"epoch": 3, "mAP50": 0.80, "mAP50_95": 0.42, "precision": 0.77, "recall": 0.81},
    {"epoch": 4, "mAP50": 0.78, "mAP50_95": 0.39, "precision": 0.76, "recall": 0.79,
     "box_loss": 1.2, "cls_loss": 0.6},
]


class BestEpochTests(TestCase):
    def test_best_epoch_follows_yolo_fitness_not_map50(self):
        self.assertEqual(EM[best_epoch_index(EM)]["epoch"], 3)

    def test_legend_shows_the_best_epoch_and_losses_the_last(self):
        c = _build_epoch_charts(EM)
        values = {s["label"]: s["value"] for s in c["detection"]}
        self.assertEqual(values["mAP50"], 0.80)
        self.assertEqual(values["mAP50-95"], 0.42)
        self.assertEqual({s["label"]: s["value"] for s in c["loss"]}["Box loss"], 1.2)
        self.assertEqual(c["best_epoch"], 3)

    def test_no_map_values_means_no_best_epoch(self):
        self.assertIsNone(best_epoch_index([{"epoch": 1, "box_loss": 2.0}]))


class TrainingPageTests(TestCase):
    def test_page_shows_kept_epoch_and_gpu_hours(self):
        user = User.objects.create_user("ec", password="x")
        self.client.force_login(user)
        project = AnnotationProject.objects.create(user=user, name="P", classes=["insect"])
        job = TrainingJob.objects.create(
            user=user, project=project, name="m", status=TrainingJob.Status.COMPLETED,
            execution_seconds=6712.5,
            metrics={"mAP50": 0.815, "best_epoch": 2, "epoch_metrics": EM})

        html = self.client.get(reverse("training:detail", kwargs={"pk": job.pk})).content.decode()

        self.assertIn("1.86 GPU-hours", html)
        self.assertNotIn("6712.5s", html)
        self.assertIn("best · epoch 3", html)
        self.assertIn("values at epoch 3", html)

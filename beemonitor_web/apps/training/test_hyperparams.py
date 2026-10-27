"""Advanced training settings: optional, stored only when set, sent to the trainer."""

import ast
from pathlib import Path

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.annotations.models import AnnotationProject
from apps.training import hyperparams as hp
from apps.training.forms import TrainingCreateForm
from apps.training.models import TrainingJob

User = get_user_model()
TRAINER = Path(__file__).resolve().parents[3] / "sagemaker_backend" / "train.py"


class HyperparamFormTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("t", password="x")
        self.project = AnnotationProject.objects.create(user=self.user, name="P", classes=["bee"])

    def form(self, **extra):
        data = {"project": self.project.pk, "name": "j", "base_model": "default",
                "frame_filter": "all", "epochs": 50, "image_size": 640, "batch_size": 16,
                "val_percent": 20, "class_subset": ["bee"], **extra}
        from django.http import QueryDict
        qd = QueryDict(mutable=True)
        for k, v in data.items():
            qd.setlist(k, v if isinstance(v, list) else [str(v)])
        return TrainingCreateForm(qd, user=self.user)

    def test_blank_advanced_settings_store_nothing(self):
        f = self.form()
        self.assertTrue(f.is_valid(), f.errors)
        self.assertEqual(f.instance.hyperparams, {})

    def test_set_values_are_kept(self):
        f = self.form(hp_optimizer="AdamW", hp_lr0="0.002", hp_cos_lr="on", hp_patience="20",
                      hp_mosaic="0")
        self.assertTrue(f.is_valid(), f.errors)
        self.assertEqual(f.instance.hyperparams, {"optimizer": "AdamW", "lr0": 0.002,
                                                  "cos_lr": True, "patience": 20, "mosaic": 0.0})

    def test_a_learning_rate_needs_an_optimizer(self):
        f = self.form(hp_lr0="0.002")
        self.assertFalse(f.is_valid())
        self.assertIn("hp_lr0", f.errors)

    def test_out_of_range_is_refused(self):
        f = self.form(hp_optimizer="SGD", hp_lr0="5")
        self.assertFalse(f.is_valid())
        self.assertIn("hp_lr0", f.errors)

    def test_the_form_page_hides_them_under_advanced(self):
        self.client.force_login(self.user)
        html = self.client.get(reverse("training:create")).content.decode()
        self.assertIn("Advanced settings", html)
        self.assertIn('name="hp_lr0"', html)
        self.assertNotIn("GPU seconds", html)

    def test_the_job_page_lists_what_was_set(self):
        job = TrainingJob.objects.create(user=self.user, project=self.project, name="j",
                                         base_model="yolov8n",
                                         hyperparams={"optimizer": "SGD", "lr0": 0.005})
        self.client.force_login(self.user)
        html = self.client.get(reverse("training:detail", args=[job.pk])).content.decode()
        self.assertIn("Learning rate", html)
        self.assertIn("0.005", html)


class TrainerAgreesTests(TestCase):
    def test_the_trainer_accepts_every_setting_the_form_offers(self):
        tree = ast.parse(TRAINER.read_text())
        trainer_keys = next(
            {k.value for k in node.value.keys}
            for node in tree.body
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "HYPERPARAMS")
        self.assertEqual(set(hp.FIELDS), trainer_keys)

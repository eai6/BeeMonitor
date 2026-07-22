from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from apps.annotations.models import AnnotationProject
from apps.devices.models import Device
from apps.pipelines.models import Pipeline
from apps.videos.models import Video

User = get_user_model()

LEGACY = [
    {"id": "v", "block_type": "input.video", "config": {}},
    {"id": "r", "block_type": "roi.nest_layout", "config": {"source": "device"}, "inputs": {"in": "v"}},
    {"id": "t", "block_type": "track.bee", "config": {"confidence": 0.4}, "inputs": {"video": "v", "rois": "r"}},
    {"id": "f", "block_type": "analyze.foraging_trips", "config": {}, "inputs": {"tracks": "t"}},
    {"id": "o", "block_type": "output.table", "config": {}, "inputs": {"in": "f"}},
]

class SmokeTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.client.force_login(self.user)
        self.device = Device.objects.create(owner=self.user, name="D", key_hash="h", prefix="p")

    def test_device_page_renders_without_removed_cards(self):
        html = self.client.get(f"/devices/{self.device.pk}/").content.decode()
        self.assertNotIn("Bee confirmation", html)
        self.assertNotIn("Review crops over cellular", html)
        self.assertIn("Scheduled processing", html)

    # The editor pulls the vendored Drawflow assets through {% static %}, which
    # needs a collectstatic manifest the test run doesn't build.
    @override_settings(STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    })
    def test_editor_renders_legacy_pipeline_with_palette_hidden(self):
        p = Pipeline.objects.create(user=self.user, title="Legacy", steps=LEGACY)
        html = self.client.get(f"/pipelines/{p.pk}/").content.decode()
        # Legacy blocks must be renderable (in BLOCKS json) but not draggable.
        self.assertIn("track.bee", html)
        self.assertNotIn('data-block="track.bee"', html)
        self.assertIn('data-block="detect.objects"', html)
        self.assertIn('data-block="track.mot"', html)

    def test_annotation_page_offers_sampling_and_labels_gpu_cost(self):
        proj = AnnotationProject.objects.create(user=self.user, name="P")
        v = Video.objects.create(user=self.user, title="c", storage_key="a/c.mp4",
                                 file_size_bytes=1, status=Video.Status.READY)
        proj.videos.add(v)
        html = self.client.get(f"/annotations/{proj.pk}/").content.decode()
        # Step numbers now live in the headings, not the button labels.
        self.assertIn(">Sample frames<", html)
        self.assertIn("Auto-label all (GPU)", html)
        # The cheap and expensive paths must stay distinguishable at a glance.
        self.assertIn("No GPU", html)
        self.assertIn("Uses GPU", html)

    def test_lessons_pages_render(self):
        self.assertEqual(self.client.get("/pipelines/lessons/").status_code, 200)
        self.assertEqual(self.client.get("/pipelines/lessons/interactions/").status_code, 200)

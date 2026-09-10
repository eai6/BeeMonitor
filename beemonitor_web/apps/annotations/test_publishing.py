"""Publishing a dataset, and taking a copy.

A public project is readable and copyable, never writable. "Improve upon it"
means copy and diverge — shared write with strangers is a moderation problem,
and a dataset that changes underneath its users cannot be cited.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.annotations import publishing
from apps.annotations.models import Annotation, AnnotationProject, ProjectShare
from apps.devices.models import Device
from apps.videos.models import Video

User = get_user_model()


class PublishingTestCase(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("owner", password="x")
        self.other = User.objects.create_user("other", password="x")
        self.device = Device.objects.create(owner=self.owner, name="Hotel A",
                                            key_hash="pub", prefix="bmk_pub")
        self.project = AnnotationProject.objects.create(
            user=self.owner, name="Flower UV", classes=["bee", "flower"])
        self.n = 0

    def clip(self, frames=2, boxes=1, hour=9):
        self.n += 1
        v = Video.objects.create(user=self.owner, device=self.device,
                                 title=f"c{self.n}", storage_key=f"pb/{self.n}.mp4",
                                 file_size_bytes=1, status=Video.Status.READY,
                                 site_name="Meadow A")
        v.hour = hour
        v.save(update_fields=["hour"])
        self.project.videos.add(v)
        for i in range(frames):
            Annotation.objects.create(
                project=self.project, video=v, frame_number=i,
                boxes=[{"label": "bee"}] * boxes if i < frames else [],
                frame_image_path=f"frames/{self.n}/{i}.jpg",
                reviewed=True)
        return v


class PublishTests(PublishingTestCase):
    def test_publishing_marks_it_and_stamps_the_date(self):
        publishing.publish(self.project)

        self.project.refresh_from_db()
        self.assertTrue(self.project.is_public)
        self.assertIsNotNone(self.project.published_at)

    def test_field_metadata_is_withheld_unless_asked_for(self):
        """For a bee hotel, recording times and sites are a movement log."""
        publishing.publish(self.project)

        self.project.refresh_from_db()
        self.assertFalse(self.project.publish_metadata)

    def test_it_can_be_included_deliberately(self):
        publishing.publish(self.project, include_metadata=True)

        self.project.refresh_from_db()
        self.assertTrue(self.project.publish_metadata)

    def test_republishing_keeps_the_original_date(self):
        publishing.publish(self.project)
        first = AnnotationProject.objects.get(pk=self.project.pk).published_at

        publishing.publish(self.project, include_metadata=True)

        self.assertEqual(
            AnnotationProject.objects.get(pk=self.project.pk).published_at, first)

    def test_unpublishing_takes_it_out_of_the_listing(self):
        publishing.publish(self.project)
        publishing.unpublish(self.project)

        self.assertNotIn(self.project, AnnotationProject.public())

    def test_the_public_listing_is_only_published_work(self):
        self.clip()
        publishing.publish(self.project)
        AnnotationProject.objects.create(user=self.other, name="Private")

        self.assertEqual(list(AnnotationProject.public()), [self.project])


class SummaryTests(PublishingTestCase):
    def test_a_card_reports_frames_boxes_and_clips(self):
        self.clip(frames=2, boxes=3)
        self.clip(frames=2, boxes=1)

        s = publishing.summary(self.project)

        self.assertEqual(s["clips"], 2)
        self.assertEqual(s["frames"], 4)
        self.assertEqual(s["boxes"], 8)

    def test_it_reports_how_varied_the_data_is_without_saying_where(self):
        """"Four hotels" says how varied it is; it does not say where any is."""
        self.clip(hour=9)
        second = Device.objects.create(owner=self.owner, name="Hotel B",
                                       key_hash="pub2", prefix="bmk_pub2")
        v = self.clip(hour=15)
        v.device = second
        v.save(update_fields=["device"])

        s = publishing.summary(self.project)

        self.assertEqual(s["devices"], 2)
        self.assertEqual(s["hours"], [9, 15])
        self.assertNotIn("site_name", s)


class CopyTests(PublishingTestCase):
    def test_a_copy_lands_in_the_copier_s_account(self):
        self.clip()
        publishing.publish(self.project)

        copy = publishing.copy_for(self.project, self.other)

        self.assertEqual(copy.user, self.other)
        self.assertEqual(copy.classes, ["bee", "flower"])

    def test_a_copy_carries_the_frames_and_the_boxes(self):
        self.clip(frames=3, boxes=2)

        copy = publishing.copy_for(self.project, self.other)

        self.assertEqual(copy.annotations.count(), 3)
        self.assertEqual(sum(len(a.boxes) for a in copy.annotations.all()), 6)

    def test_frames_are_referenced_rather_than_duplicated(self):
        """An Annotation carries a path into the processed bucket, and nothing
        deletes those objects — so a copy cannot break, and a few thousand
        JPEGs are not duplicated per copy."""
        self.clip(frames=2)

        copy = publishing.copy_for(self.project, self.other)

        originals = set(self.project.annotations.values_list("frame_image_path", flat=True))
        copied = set(copy.annotations.values_list("frame_image_path", flat=True))
        self.assertEqual(originals, copied)

    def test_a_copy_does_not_carry_the_footage(self):
        """Attaching the source clips would hand over footage the copier was
        never shared — the property the whole sharing model protects."""
        self.clip()

        copy = publishing.copy_for(self.project, self.other)

        self.assertEqual(copy.videos.count(), 0)
        self.assertFalse(Video.accessible(self.other).exists())

    def test_a_copy_is_not_marked_as_reviewed_by_the_copier(self):
        """Carrying the flag over would claim someone signed off work they have
        never seen."""
        self.clip(frames=2)

        copy = publishing.copy_for(self.project, self.other)

        self.assertFalse(copy.annotations.filter(reviewed=True).exists())
        self.assertTrue(self.project.annotations.filter(reviewed=True).exists())

    def test_a_copy_says_where_it_came_from(self):
        copy = publishing.copy_for(self.project, self.other)

        self.assertEqual(copy.copied_from, self.project)
        self.assertEqual(copy.copied_from_name, "Flower UV")

    def test_the_attribution_survives_the_original_being_deleted(self):
        copy = publishing.copy_for(self.project, self.other)
        self.project.delete()

        copy.refresh_from_db()
        self.assertIsNone(copy.copied_from)
        self.assertEqual(copy.copied_from_name, "Flower UV")

    def test_a_copy_starts_private(self):
        copy = publishing.copy_for(self.project, self.other)

        self.assertFalse(copy.is_public)

    def test_editing_a_copy_does_not_touch_the_original(self):
        self.clip(frames=1)
        copy = publishing.copy_for(self.project, self.other)

        ann = copy.annotations.first()
        ann.boxes = [{"label": "wasp"}]
        ann.save(update_fields=["boxes"])

        self.assertEqual(self.project.annotations.first().boxes,
                         [{"label": "bee"}])


class ReadableScopeTests(PublishingTestCase):
    def test_readable_includes_published_work(self):
        publishing.publish(self.project)

        self.assertIn(self.project, AnnotationProject.readable(self.other))

    def test_accessible_stays_private_only(self):
        """The private scope must not widen by accident — every existing write
        path is built on it."""
        publishing.publish(self.project)

        self.assertNotIn(self.project, AnnotationProject.accessible(self.other))

    def test_publishing_grants_no_write_of_any_kind(self):
        publishing.publish(self.project)

        for scope in (AnnotationProject.annotatable, AnnotationProject.reviewable,
                      AnnotationProject.manageable, AnnotationProject.owned):
            self.assertNotIn(self.project, scope(self.other), scope.__name__)


class PublishViewTests(PublishingTestCase):
    def setUp(self):
        super().setUp()
        self.clip()

    def test_only_the_owner_may_publish(self):
        ProjectShare.objects.create(project=self.project, user=self.other,
                                    role="manager")
        self.client.force_login(self.other)

        resp = self.client.post(
            reverse("annotations:publish", args=[self.project.pk]),
            {"visibility": "public"})

        self.assertEqual(resp.status_code, 404)
        self.project.refresh_from_db()
        self.assertFalse(self.project.is_public)

    def test_the_owner_can_publish_and_unpublish(self):
        self.client.force_login(self.owner)
        url = reverse("annotations:publish", args=[self.project.pk])

        self.client.post(url, {"visibility": "public"})
        self.assertTrue(AnnotationProject.objects.get(pk=self.project.pk).is_public)

        self.client.post(url, {"visibility": "private"})
        self.assertFalse(AnnotationProject.objects.get(pk=self.project.pk).is_public)

    def test_the_metadata_choice_is_reported_back(self):
        self.client.force_login(self.owner)

        resp = self.client.post(
            reverse("annotations:publish", args=[self.project.pk]),
            {"visibility": "public"}, follow=True)

        self.assertIn("withheld", " ".join(str(m) for m in resp.context["messages"]))


class BrowseAndCopyViewTests(PublishingTestCase):
    def setUp(self):
        super().setUp()
        self.clip(frames=2)
        self.client.force_login(self.other)

    def test_a_private_project_is_not_listed(self):
        html = self.client.get(reverse("annotations:browse")).content.decode()

        self.assertNotIn("Flower UV", html)

    def test_a_published_project_is(self):
        publishing.publish(self.project)

        html = self.client.get(reverse("annotations:browse")).content.decode()

        self.assertIn("Flower UV", html)
        self.assertIn("no site or time metadata", html)

    def test_copying_a_private_project_is_refused(self):
        """Otherwise the browse page is decoration and the URL is the door."""
        resp = self.client.post(reverse("annotations:copy", args=[self.project.pk]))

        self.assertEqual(resp.status_code, 404)
        self.assertEqual(AnnotationProject.objects.filter(user=self.other).count(), 0)

    def test_copying_a_published_project_works(self):
        publishing.publish(self.project)

        resp = self.client.post(reverse("annotations:copy", args=[self.project.pk]))

        copy = AnnotationProject.objects.get(user=self.other)
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(copy.annotations.count(), 2)
        self.assertEqual(copy.copied_from, self.project)

    def test_the_copy_gains_no_access_to_the_original(self):
        publishing.publish(self.project)
        self.client.post(reverse("annotations:copy", args=[self.project.pk]))

        for scope in (AnnotationProject.annotatable, AnnotationProject.manageable,
                      AnnotationProject.owned):
            self.assertNotIn(self.project, scope(self.other), scope.__name__)

    def test_a_named_copy_keeps_the_name(self):
        publishing.publish(self.project)

        self.client.post(reverse("annotations:copy", args=[self.project.pk]),
                         {"name": "My fork"})

        self.assertTrue(AnnotationProject.objects.filter(
            user=self.other, name="My fork").exists())

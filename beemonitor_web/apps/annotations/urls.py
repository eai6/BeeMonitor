from django.urls import path

from . import views

app_name = "annotations"

urlpatterns = [
    path("", views.ProjectListView.as_view(), name="list"),
    path("new/", views.ProjectCreateView.as_view(), name="create"),
    path("<int:pk>/", views.ProjectDetailView.as_view(), name="detail"),
    path("<int:pk>/settings/", views.ProjectUpdateView.as_view(), name="settings"),
    path("browse/", views.PublicBrowseView.as_view(), name="browse"),
    path("<int:pk>/people/", views.ProjectPeopleView.as_view(), name="people"),
    path("<int:pk>/publish/", views.PublishProjectView.as_view(), name="publish"),
    path("<int:pk>/copy/", views.CopyProjectView.as_view(), name="copy"),
    path("<int:pk>/people/invite/", views.ShareInviteView.as_view(), name="share_invite"),
    path("<int:pk>/people/update/", views.ShareUpdateView.as_view(), name="share_update"),
    path("<int:pk>/assign/", views.AssignFramesView.as_view(), name="assign_frames"),
    path("<int:pk>/take/", views.TakeFramesView.as_view(), name="take_frames"),
    path("<int:pk>/delete/", views.ProjectDeleteView.as_view(), name="delete"),
    path("<int:pk>/add-videos/", views.AddVideosView.as_view(), name="add_videos"),
    path("<int:pk>/add/", views.AddVideosWorkspaceView.as_view(), name="add_videos_page"),
    path("<int:pk>/add/draft/", views.AddVideosDraftView.as_view(), name="add_videos_draft"),
    path("<int:pk>/add/grid/", views.AddVideosGridView.as_view(), name="add_videos_grid"),
    path("<int:pk>/remove-video/", views.RemoveVideoView.as_view(), name="remove_video"),
    path("<int:pk>/edit/", views.AnnotationEditorView.as_view(), name="editor"),
    path("<int:pk>/transfer/", views.TransferVideoView.as_view(), name="transfer_video"),
    path("<int:pk>/save/", views.SaveAnnotationView.as_view(), name="save"),
    path("<int:pk>/sample-frames/", views.SampleFramesView.as_view(), name="sample_frames"),
    path("<int:pk>/sample-frames/cancel/", views.CancelSamplingView.as_view(), name="sample_frames_cancel"),
    path("<int:pk>/pre-annotate/", views.PreAnnotateView.as_view(), name="pre_annotate"),
    path("<int:pk>/pre-annotate/cancel/", views.CancelPreAnnotationView.as_view(), name="pre_annotate_cancel"),
    path("<int:pk>/export/", views.ExportProjectView.as_view(), name="export"),
    path("<int:pk>/frame/", views.FrameImageView.as_view(), name="frame_image"),
    # The frame grid is the project page's first tab now; old links land there.
    path("<int:pk>/review/", views.ReviewRedirectView.as_view(), name="review"),
]

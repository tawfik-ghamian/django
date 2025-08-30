# from django.urls import path
# # from . import views
# from .views import VideoUploadView, ProcessedVideoView, FeedBackView

# urlpatterns = [
#     # path('upload/', views.upload_video, name='upload_video'),
#     # path('view_videos/', views.video_list, name='vew_videos'),
#     path('upload/', VideoUploadView.as_view(), name='video-upload'),
#     path('processed/<int:pk>/', ProcessedVideoView.as_view(), name='processed-video'),
#     path('feedback/<int:pk>/', FeedBackView.as_view(), name='feedback'),

#     # path('compare/<int:video1_id>/<int:video2_id>/', views.compare_videos, name='compare_videos'),
# ]


# urls.py - Updated with new endpoint
from django.urls import path
from .views import VideoUploadView, ProcessedVideoView, FeedBackView, VideoStatusView

urlpatterns = [
    path('upload/', VideoUploadView.as_view(), name='video-upload'),
    path('processed/<int:pk>/', ProcessedVideoView.as_view(), name='processed-video'),
    path('feedback/<int:pk>/', FeedBackView.as_view(), name='feedback'),
    path('status/<int:pk>/', VideoStatusView.as_view(), name='video-status'),  # New endpoint
]
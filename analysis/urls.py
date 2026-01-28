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
from .views import (
    VideoUploadView, 
    ProcessedVideoView, 
    FeedBackView, 
    VideoListView,
    VideoStatusView,
    video_status, 
    task_status,
    retry_video_processing
)
# from .views import VideoUploadView, ProcessedVideoView, FeedBackView, VideoStatusView, video_status, retry_video_processing

urlpatterns = [
    path('upload/', VideoUploadView.as_view(), name='video-upload'),
    path('all/', VideoListView.as_view(), name='video-list'),
    path('processed/<int:pk>/', ProcessedVideoView.as_view(), name='processed-video'),
    path('feedback/<int:pk>/', FeedBackView.as_view(), name='feedback'),
    # Check task status by task ID (for real-time progress)
    path('task-status/<str:task_id>/', task_status, name='task-status'),
    path('status/<int:pk>/', video_status, name='video-status'),
    path('retry/<int:pk>/', retry_video_processing, name='retry-processing'),
    path('video/<int:pk>/', VideoStatusView.as_view(), name='video-detail'),
]
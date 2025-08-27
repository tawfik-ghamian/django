# from rest_framework import serializers
# from .models import Video

# class VideoSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Video
#         fields = '__all__'


# # Enhanced serializer.py
# from rest_framework import serializers
# from .models import Video

# class VideoSerializer(serializers.ModelSerializer):
#     is_tennis_complete = serializers.ReadOnlyField()
#     is_under_training = serializers.ReadOnlyField()
#     processing_status_display = serializers.ReadOnlyField()
#     sport_detected_display = serializers.CharField(source='get_sport_detected_display', read_only=True)
#     analysis_status_display = serializers.CharField(source='get_analysis_status_display', read_only=True)
    
#     class Meta:
#         model = Video
#         fields = [
#             'id', 'video_file', 'processed_video', 'video_data_json',
#             'sport_detected', 'sport_detected_display', 'sport_confidence', 
#             'analysis_status', 'analysis_status_display', 'processing_status_display',
#             'overall_score', 'detailed_scores', 'frames_analyzed',
#             'feedback', 'shot_types_detected',
#             'training_progress', 'estimated_completion', 'basic_metrics',
#             'uploaded_at', 'processed_at',
#             'is_tennis_complete', 'is_under_training'
#         ]
#         read_only_fields = [
#             'sport_detected', 'sport_confidence', 'analysis_status',
#             'overall_score', 'detailed_scores', 'frames_analyzed',
#             'feedback', 'shot_types_detected', 'training_progress',
#             'estimated_completion', 'basic_metrics', 'processed_at'
#         ]

# class VideoListSerializer(serializers.ModelSerializer):
#     """Simplified serializer for listing videos"""
#     sport_detected_display = serializers.CharField(source='get_sport_detected_display', read_only=True)
#     processing_status_display = serializers.ReadOnlyField()
    
#     class Meta:
#         model = Video
#         fields = [
#             'id', 'sport_detected', 'sport_detected_display', 
#             'analysis_status', 'processing_status_display',
#             'overall_score', 'frames_analyzed', 'uploaded_at'
#         ]

from rest_framework import serializers
from .models import Video

class VideoSerializer(serializers.ModelSerializer):
    is_tennis_complete = serializers.ReadOnlyField()
    is_under_training = serializers.ReadOnlyField()
    processing_status_display = serializers.ReadOnlyField()
    sport_type_display = serializers.CharField(source='get_sport_type_display', read_only=True)
    analysis_status_display = serializers.CharField(source='get_analysis_status_display', read_only=True)
    
    class Meta:
        model = Video
        fields = [
            'id', 'video_file', 'processed_video', 'video_data_json',
            'sport_type', 'sport_type_display', 
            'analysis_status', 'analysis_status_display', 'processing_status_display',
            'overall_score', 'detailed_scores', 'frames_analyzed',
            'feedback', 'shot_types_detected',
            'training_progress', 'estimated_completion', 'basic_metrics',
            'uploaded_at', 'processed_at',
            'is_tennis_complete', 'is_under_training'
        ]
        read_only_fields = [
            'analysis_status', 'overall_score', 'detailed_scores', 'frames_analyzed',
            'feedback', 'shot_types_detected', 'training_progress',
            'estimated_completion', 'basic_metrics', 'processed_at'
        ]

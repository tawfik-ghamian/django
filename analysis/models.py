# from django.db import models

# class Video(models.Model): 
#     file = models.FileField(upload_to='videos/') 
#     uploaded_at = models.DateTimeField(auto_now_add=True) 
    
# class PoseComparison(models.Model): 
#     video = models.ForeignKey(Video, on_delete=models.CASCADE) 
#     toe_angles = models.JSONField() # Store angles as JSON 
#     comparison_results = models.JSONField() 
#     created_at = models.DateTimeField(auto_now_add=True)

# class AngleData(models.Model):
#     processed_video = models.ForeignKey(ProcessedVideo, on_delete=models.CASCADE)
#     frame_number = models.IntegerField()
#     neck_shoulder_angle = models.FloatField()
#     wrist_waist_angle = models.FloatField()


# class UploadedVideo(models.Model):
#     id = models.AutoField(primary_key=True)
#     video_file1 = models.FileField(upload_to='uploaded_videos/')
#     video_file2 = models.FileField(upload_to='uploaded_videos/', null=True, blank=True)
#     uploaded_at = models.DateTimeField(auto_now_add=True)

# class ProcessedVideo(models.Model):
#     id = models.AutoField(primary_key=True)
#     uploaded_video = models.ForeignKey(UploadedVideo, on_delete=models.CASCADE)
#     processed_video_file = models.FileField(upload_to='processed_videos/')
#     processed_at = models.DateTimeField(auto_now_add=True)

# class AngleData(models.Model):
#     id = models.AutoField(primary_key=True)
#     uploaded_video = models.ForeignKey(UploadedVideo, on_delete=models.CASCADE)
#     frame_number = models.IntegerField()
#     neck_shoulder_angle = models.FloatField()
#     knee_hip_angle = models.FloatField()
# models.py
# from django.db import models

# class Video(models.Model):
#     id = models.AutoField(primary_key=True)
#     video_file = models.FileField(upload_to='videos/')
#     processed_video = models.FileField(upload_to='processed_videos/', null=True, blank=True)
#     video_data_json = models.FileField(upload_to='video_data/', null=True, blank=True)
#     feedback = models.TextField(null=True, blank=True)

#     def __str__(self):
#         return self.video_file.name


# # Enhanced models.py - Updated with more fields
# from django.db import models
# from django.utils import timezone

# class Video(models.Model):
#     SPORT_CHOICES = [
#         ('tennis', 'Tennis'),
#         ('running', 'Running'),
#         ('soccer', 'Soccer'),
#         ('unknown', 'Unknown'),
#     ]
    
#     ANALYSIS_STATUS_CHOICES = [
#         ('pending', 'Pending'),
#         ('processing', 'Processing'),
#         ('complete', 'Complete Analysis Available'),  # Tennis
#         ('under_training', 'Sport Under Training'),     # Running/Soccer
#         ('general_processing', 'General Processing'),   # Unknown sports
#         ('failed', 'Analysis Failed'),
#     ]
    
#     id = models.AutoField(primary_key=True)
#     video_file = models.FileField(upload_to='videos/')
#     processed_video = models.FileField(upload_to='processed_videos/', null=True, blank=True)
#     video_data_json = models.FileField(upload_to='video_data/', null=True, blank=True)
    
#     # Multi-sport detection fields
#     sport_detected = models.CharField(max_length=20, choices=SPORT_CHOICES, default='unknown')
#     sport_confidence = models.FloatField(null=True, blank=True, help_text="Confidence score for sport detection (0-1)")
#     analysis_status = models.CharField(max_length=20, choices=ANALYSIS_STATUS_CHOICES, default='pending')
    
#     # Analysis results (primarily for tennis)
#     overall_score = models.FloatField(null=True, blank=True, help_text="Overall technique score (0-10)")
#     detailed_scores = models.JSONField(null=True, blank=True, help_text="Detailed scores for different aspects")
#     frames_analyzed = models.IntegerField(null=True, blank=True, help_text="Number of frames with pose data")
    
#     # Feedback and metadata
#     feedback = models.TextField(null=True, blank=True, help_text="AI-generated coaching feedback")
#     shot_types_detected = models.JSONField(null=True, blank=True, help_text="List of detected shots/movements")
    
#     # Training progress (for sports under development)
#     training_progress = models.CharField(max_length=10, null=True, blank=True, help_text="Training progress percentage")
#     estimated_completion = models.CharField(max_length=20, null=True, blank=True, help_text="Estimated completion date")
#     basic_metrics = models.JSONField(null=True, blank=True, help_text="Basic movement metrics captured")
    
#     # Timestamps
#     uploaded_at = models.DateTimeField(auto_now_add=True)
#     processed_at = models.DateTimeField(null=True, blank=True)

#     class Meta:
#         ordering = ['-uploaded_at']
#         indexes = [
#             models.Index(fields=['sport_detected']),
#             models.Index(fields=['analysis_status']),
#             models.Index(fields=['uploaded_at']),
#         ]

#     def __str__(self):
#         return f"{self.get_sport_detected_display()} - {self.video_file.name}"
    
#     @property
#     def is_tennis_complete(self):
#         return self.sport_detected == 'tennis' and self.analysis_status == 'complete'
    
#     @property
#     def is_under_training(self):
#         return self.analysis_status == 'under_training'
    
#     @property
#     def processing_status_display(self):
#         status_messages = {
#             'pending': 'Waiting for processing...',
#             'processing': 'Analyzing video and extracting poses...',
#             'complete': 'Complete professional analysis available!',
#             'under_training': f'Sport detected! AI training {self.training_progress or "in progress"}',
#             'general_processing': 'Movement patterns captured, sport-specific training needed',
#             'failed': 'Processing encountered an error'
#         }
#         return status_messages.get(self.analysis_status, 'Unknown status')

#     def save(self, *args, **kwargs):
#         if self.analysis_status in ['complete', 'under_training', 'general_processing'] and not self.processed_at:
#             self.processed_at = timezone.now()
#         super().save(*args, **kwargs)


from django.db import models
from django.utils import timezone

class Video(models.Model):
    SPORT_CHOICES = [
        ('tennis', 'Tennis'),
        ('running', 'Running'),
        ('soccer', 'Soccer'),
    ]
    
    ANALYSIS_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('complete', 'Complete Analysis Available'),  # Tennis
        ('under_training', 'Sport Under Training'),     # Running/Soccer
        ('failed', 'Analysis Failed'),
    ]
    
    id = models.AutoField(primary_key=True)
    video_file = models.FileField(upload_to='videos/')
    processed_video = models.FileField(upload_to='processed_videos/', null=True, blank=True)
    video_data_json = models.FileField(upload_to='video_data/', null=True, blank=True)
    
    # User-specified sport (no auto-detection)
    sport_type = models.CharField(max_length=20, choices=SPORT_CHOICES, help_text="Sport type selected by user")
    analysis_status = models.CharField(max_length=20, choices=ANALYSIS_STATUS_CHOICES, default='pending')
    
    # Analysis results (primarily for tennis)
    overall_score = models.FloatField(null=True, blank=True, help_text="Overall technique score (0-10)")
    detailed_scores = models.JSONField(null=True, blank=True, help_text="Detailed scores for different aspects")
    frames_analyzed = models.IntegerField(null=True, blank=True, help_text="Number of frames with pose data")
    
    # Feedback and metadata
    feedback = models.TextField(null=True, blank=True, help_text="AI-generated coaching feedback")
    shot_types_detected = models.JSONField(null=True, blank=True, help_text="List of detected shots/movements")
    
    # Training progress (for sports under development)
    training_progress = models.CharField(max_length=10, null=True, blank=True, help_text="Training progress percentage")
    estimated_completion = models.CharField(max_length=20, null=True, blank=True, help_text="Estimated completion date")
    basic_metrics = models.JSONField(null=True, blank=True, help_text="Basic movement metrics captured")
    
    # Timestamps
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    task_id = models.CharField(
        max_length=255, 
        blank=True, 
        null=True,
        help_text="Celery task ID for async processing"
    )

    class Meta:
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['sport_type']),
            models.Index(fields=['analysis_status']),
            models.Index(fields=['uploaded_at']),
        ]

    def __str__(self):
        return f"Video {self.id} - {self.get_sport_type_display()} - {self.video_file.name}"
    
    @property
    def is_tennis_complete(self):
        return self.sport_type == 'tennis' and self.analysis_status == 'complete'
    
    @property
    def is_under_training(self):
        return self.analysis_status == 'under_training'
    
    @property
    def processing_status_display(self):
        status_messages = {
            'pending': 'Waiting for processing...',
            'processing': 'Analyzing video and extracting poses...',
            'complete': 'Complete professional analysis available!',
            'under_training': f'Sport detected! AI training {self.training_progress or "in progress"}',
            'failed': 'Processing encountered an error'
        }
        return status_messages.get(self.analysis_status, 'Unknown status')

    def save(self, *args, **kwargs):
        if self.analysis_status in ['complete', 'under_training'] and not self.processed_at:
            self.processed_at = timezone.now()
        super().save(*args, **kwargs)
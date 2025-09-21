# analysis/tasks.py
from celery import shared_task
from django.utils import timezone
import logging
import os

logger = logging.getLogger(__name__)

@shared_task(bind=True, soft_time_limit=600, time_limit=900)
def process_video_async(self, video_id, video_name, sport_type):
    """Async video processing with proper error handling"""
    try:
        from .models import Video
        from .video_processing import process_video
        from .multi_sport_coach import SimplifiedMultiSportCoachAnalyzer
        
        # Get video instance
        video = Video.objects.get(id=video_id)
        video.analysis_status = 'processing'
        video.save()
        
        logger.info(f"Starting video processing for {video_id}")
        
        # Process video (heavy AI inference)
        video_data, video_output_path, video_data_path = process_video(video_id, video_name)
        
        # Save processed files
        video.processed_video.save(f'{video_name}_processed.mp4', open(video_output_path, 'rb'))
        video.video_data_json.save(f'{video_name}_data.json', open(video_data_path, 'rb'))
        
        # Multi-sport analysis
        multi_sport_analyzer = SimplifiedMultiSportCoachAnalyzer()
        analysis_result = multi_sport_analyzer.analyze_video(sport_type, video_data)
        
        # Update video instance with results
        video.analysis_status = analysis_result['analysis_status']
        video.frames_analyzed = analysis_result['frames_analyzed']
        video.feedback = analysis_result['feedback']
        video.processed_at = timezone.now()
        
        # Add sport-specific data
        if analysis_result['analysis_status'] == 'complete':
            video.overall_score = analysis_result['overall_score']
            video.detailed_scores = analysis_result['detailed_scores']
            video.shot_types_detected = analysis_result.get('shot_types_detected', [])
        else:
            video.training_progress = analysis_result.get('training_progress')
            video.estimated_completion = analysis_result.get('estimated_completion')
            video.basic_metrics = analysis_result.get('basic_metrics')
        
        video.save()
        
        # Clean up temporary files
        os.remove(video_output_path)
        os.remove(video_data_path)
        
        logger.info(f"Completed video processing for {video_id}")
        return {
            "status": "success", 
            "video_id": video_id,
            "analysis_status": analysis_result['analysis_status']
        }
        
    except Exception as exc:
        logger.error(f"Video processing failed for {video_id}: {str(exc)}")
        
        # Update status on failure
        try:
            video = Video.objects.get(id=video_id)
            video.analysis_status = 'failed'
            video.feedback = f"Processing failed: {str(exc)}"
            video.save()
        except:
            pass
            
        # Retry logic
        if self.request.retries < 2:
            raise self.retry(countdown=60, exc=exc)
        raise
# # analysis/tasks.py
# from celery import shared_task
# from django.utils import timezone
# import logging
# import os

# logger = logging.getLogger(__name__)

# @shared_task(bind=True, soft_time_limit=600, time_limit=900)
# def process_video_async(self, video_id, video_name, sport_type):
#     """Async video processing with proper error handling"""
#     try:
#         from .models import Video
#         from .video_processing import process_video
#         from .multi_sport_coach import SimplifiedMultiSportCoachAnalyzer
        
#         # Get video instance
#         video = Video.objects.get(id=video_id)
#         video.analysis_status = 'processing'
#         video.save()
        
#         logger.info(f"Starting video processing for {video_id}")
        
#         # Process video (heavy AI inference)
#         video_data, video_output_path, video_data_path = process_video(video_id, video_name)
        
#         # Save processed files
#         video.processed_video.save(f'{video_name}_processed.mp4', open(video_output_path, 'rb'))
#         video.video_data_json.save(f'{video_name}_data.json', open(video_data_path, 'rb'))
        
#         # Multi-sport analysis
#         multi_sport_analyzer = SimplifiedMultiSportCoachAnalyzer()
#         analysis_result = multi_sport_analyzer.analyze_video(sport_type, video_data)
        
#         # Update video instance with results
#         video.analysis_status = analysis_result['analysis_status']
#         video.frames_analyzed = analysis_result['frames_analyzed']
#         video.feedback = analysis_result['feedback']
#         video.processed_at = timezone.now()
        
#         # Add sport-specific data
#         if analysis_result['analysis_status'] == 'complete':
#             video.overall_score = analysis_result['overall_score']
#             video.detailed_scores = analysis_result['detailed_scores']
#             video.shot_types_detected = analysis_result.get('shot_types_detected', [])
#         else:
#             video.training_progress = analysis_result.get('training_progress')
#             video.estimated_completion = analysis_result.get('estimated_completion')
#             video.basic_metrics = analysis_result.get('basic_metrics')
        
#         video.save()
        
#         # Clean up temporary files
#         os.remove(video_output_path)
#         os.remove(video_data_path)
        
#         logger.info(f"Completed video processing for {video_id}")
#         return {
#             "status": "success", 
#             "video_id": video_id,
#             "analysis_status": analysis_result['analysis_status']
#         }
        
#     except Exception as exc:
#         logger.error(f"Video processing failed for {video_id}: {str(exc)}")
        
#         # Update status on failure
#         try:
#             video = Video.objects.get(id=video_id)
#             video.analysis_status = 'failed'
#             video.feedback = f"Processing failed: {str(exc)}"
#             video.save()
#         except:
#             pass
            
#         # Retry logic
#         if self.request.retries < 2:
#             raise self.retry(countdown=60, exc=exc)
#         raise


# analysis/tasks.py
from celery import shared_task
from celery.utils.log import get_task_logger
from django.utils import timezone
import logging
from django.core.files import File
import os

logger = get_task_logger(__name__)

@shared_task(bind=True, soft_time_limit=7200, time_limit=7500)  # 2 hour limits
def process_video_async(self, video_id, video_name, sport_type):
    """
    Async video processing with proper error handling.
    This runs in the background, so the API request returns immediately.
    """
    try:
        from .models import Video
        from .video_processing import process_video
        from .multi_sport_coach import SimplifiedMultiSportCoachAnalyzer
        
        logger.info(f"🎬 Starting async processing for video {video_id}, sport: {sport_type}")
        
        # Get video instance
        video = Video.objects.get(id=video_id)
        video.analysis_status = 'processing'
        video.save()
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 10, 'total': 100, 'status': 'Processing video frames...'}
        )
        
        logger.info(f"Processing video frames for {video_id}")
        
        # Process video (heavy AI inference) - THIS IS THE LONG PART
        video_data, video_output_path, video_data_path = process_video(video_id, video_name)
        
        logger.info(f"Video processing completed. Frames: {len(video_data)}")
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 60, 'total': 100, 'status': 'Saving processed files...'}
        )
        
        # Save processed files
        with open(video_output_path, 'rb') as f:
            video.processed_video.save(f'{video_name}_processed.mp4', File(f), save=False)
        
        with open(video_data_path, 'rb') as f:
            video.video_data_json.save(f'{video_name}_data.json', File(f), save=False)
        
        logger.info(f"Processed files saved for video {video_id}")
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 80, 'total': 100, 'status': 'Running AI analysis...'}
        )
        
        # Multi-sport analysis
        logger.info(f"Starting AI analysis for sport: {sport_type}")
        multi_sport_analyzer = SimplifiedMultiSportCoachAnalyzer()
        analysis_result = multi_sport_analyzer.analyze_video(sport_type, video_data)
        
        logger.info(f"AI analysis completed. Status: {analysis_result['analysis_status']}")
        
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
            logger.info(f"✅ Tennis analysis complete. Score: {analysis_result['overall_score']}/10")
        else:
            video.training_progress = analysis_result.get('training_progress')
            video.estimated_completion = analysis_result.get('estimated_completion')
            video.basic_metrics = analysis_result.get('basic_metrics')
            logger.info(f"✅ Under-training sport processed: {sport_type}")
        
        video.save()
        
        # Clean up temporary files
        try:
            os.remove(video_output_path)
            os.remove(video_data_path)
            logger.info(f"Temporary files cleaned up for video {video_id}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temp files: {cleanup_error}")
        
        logger.info(f"✅ Video processing completed successfully for {video_id}")
        
        # Return result
        return {
            "status": "success", 
            "video_id": video_id,
            "analysis_status": analysis_result['analysis_status'],
            "overall_score": analysis_result.get('overall_score'),
            "frames_analyzed": analysis_result['frames_analyzed']
        }
        
    except Exception as exc:
        logger.error(f"❌ Video processing failed for {video_id}: {str(exc)}", exc_info=True)
        
        # Update status on failure
        try:
            video = Video.objects.get(id=video_id)
            video.analysis_status = 'failed'
            video.feedback = f"Processing failed: {str(exc)}"
            video.save()
        except Exception as db_error:
            logger.error(f"Failed to update video status: {db_error}")
            
        # Retry logic (max 2 retries)
        if self.request.retries < 2:
            logger.info(f"Retrying task for video {video_id}, attempt {self.request.retries + 1}")
            raise self.retry(countdown=60, exc=exc, max_retries=2)
        
        raise
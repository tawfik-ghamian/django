# from django.shortcuts import render
# from django.http import JsonResponse
# from .models import UploadedVideo, AngleData, ProcessedVideo
# from .video_processing import process_video, compare_angles

# def upload_video(request):
#     if request.method == 'POST':
#         video = UploadedVideo.objects.create(file=request.FILES['file']) 
#         processed_video_path = process_video(video.file.path,video.id) 
#         angles = extract_angles_from_processed_video(processed_video_path) # Implement this function 
#         comparison = PoseComparison.objects.create(video=video, toe_angles=angles) 
#         return JsonResponse({'video_id': video.id})

# def compare_videos(request, video1_id, video2_id):
#     video1 = UploadedVideo.objects.get(id=video1_id)
#     video2 = UploadedVideo.objects.get(id=video2_id)
    
#     angles1 = PoseComparison.objects.get(video=video1).toe_angles
#     angles2 = PoseComparison.objects.get(video=video2).toe_angles
    
#     comparison_results = compare_angles(angles1, angles2)
#     PoseComparison.objects.filter(video=video1).update(comparison_results=comparison_results)
    
#     return JsonResponse({'comparison_results': comparison_results})


# from .forms import VideoUploadForm
# from django.shortcuts import render, redirect
# from .video_processing import process_video
# from rest_framework.decorators import api_view,parser_classes
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.parsers import FileUploadParser
# from .serializer import VideoSerializer
# from .models import Video
# from .forms import VideoUploadForm

# @api_view(['POST'])
# # @parser_classes([FileUploadParser])
# def upload_video(request):
#     # if request.method == 'POST':
#     #     # form = VideoUploadForm(request.POST, request.FILES)
#     #     if form.is_valid():
#     #         video = form.save()
#     #         print(video.id)
#     #         print(video.video_file)
#     #         process_video(video.id) 

#     #         return redirect('video_list.html')
#     # else:
#     #     # form = VideoUploadForm()
#     # return render(request, 'upload_video.html', {'form': form})
    
#     if request.method == 'POST':
#         print("hi")
#         print(request.FILES)
#         form = VideoUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             uploaded_file = request.FILES.get("video_file")
#             print(uploaded_file)
#             video = Video.objects.create(video_file=uploaded_file)
#             print(video.id)
#             # process_video(video.id)  # Trigger video processing asynchronously
#             return Response(uploaded_file, status=status.HTTP_201_CREATED)
#         else:
#             print(form.errors)
#             return Response(form.errors, status=status.HTTP_400_BAD_REQUEST)

#     return Response(status=status.HTTP_405_METHOD_NOT_ALLOWED)

# # def video_list(request):
# #     videos = Video.objects.all()
# #     return render(request, 'video_list.html', {'videos': videos})

# @api_view(['GET'])
# def video_list(request):
#     """
#     API endpoint to retrieve a list of all uploaded videos.

#     Returns a JSON response with details of each video.
#     """

#     videos = Video.objects.all()
#     serializer = VideoSerializer(videos, many=True)
#     return Response(serializer.data)


# from rest_framework import generics, status
# from rest_framework.response import Response
# from rest_framework.exceptions import ValidationError

# from .video_processing import process_video
# from .models import Video
# from .serializer import VideoSerializer

# class VideoUploadView(generics.CreateAPIView):
#     queryset = Video.objects.all()
#     serializer_class = VideoSerializer

#     def create(self, request, *args, **kwargs):
#         serializer = self.get_serializer(data=request.data)
#         serializer.is_valid(raise_exception=True)

#         # Custom validation: File size and type restrictions
#         video_file = request.FILES['video_file'] 
#         allowed_types = ['video/mp4', 'video/webm', 'video/avi'] 
        
#         if video_file.size > 30 * 1024 * 1024:  # 5MB limit
#             raise ValidationError("Video file size exceeds the limit.")

#         if not video_file.content_type in allowed_types:
#             raise ValidationError("Invalid file type. Allowed types: {}".format(', '.join(allowed_types)))

#         self.perform_create(serializer)
#         headers = self.get_success_headers(serializer.data)
#         video_name = video_file.name.split(".")[0]
#         print(video_name)
#         video_data, video_output=process_video(serializer.data["id"],video_name)
        
#         serializer["processed_video"] = video_output
#         serializer.save()
        
#         return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


# # views.py
# from rest_framework import generics, status
# from rest_framework.response import Response
# from django.http import FileResponse
# from django.shortcuts import get_object_or_404
# from rest_framework.exceptions import ValidationError
# from .video_processing import process_video
# from .models import Video
# from .serializer import VideoSerializer
# import os
# import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")

# class VideoUploadView(generics.CreateAPIView):
#     queryset = Video.objects.all()
#     serializer_class = VideoSerializer
    
#     def create(self, request, *args, **kwargs):
#         serializer = self.get_serializer(data=request.data)
#         serializer.is_valid(raise_exception=True)
        
#         # Custom validation: File size and type restrictions
#         video_file = request.FILES['video_file']
#         print(video_file)
#         allowed_types = ['video/mp4', 'video/webm', 'video/avi']
        
#         if video_file.size > 30 * 1024 * 1024:  # 30MB limit
#             raise ValidationError("Video file size exceeds the limit.")
#         if not video_file.content_type in allowed_types:
#             raise ValidationError("Invalid file type. Allowed types: {}".format(', '.join(allowed_types)))
        
#         self.perform_create(serializer)
#         headers = self.get_success_headers(serializer.data)
#         video_name = video_file.name.split(".")[0]
        
#         # Process the video
#         video_data, video_output_path, video_data_path = process_video(serializer.instance.id, video_name)
        
#         # Save the processed video and JSON file
#         serializer.instance.processed_video.save(f'{video_name}_processed.mp4', open(video_output_path, 'rb'))
#         serializer.instance.video_data_json.save(f'{video_name}_data.json', open(video_data_path, 'rb'))
#         print(video_data.get(6))
#         video_data = [data for data in video_data]
#         print(len(video_data))
        
#         # Generate feedback using OpenAI
#         system_prompt = f"""
#         You are a professional tennis coach. The following data are pose estimation keypoints for a tennis player with predicted shot type for each frame. 
#         Please provide a score out of 10 and informative keypoints about improving the player's shot technique.
#         Be more accurate with feedback giving. Don't talk information you didn't sure about it. Make your feedback short and more informative.
#         ignore the null records in data
#         """
        
#         # Generate feedback using OpenAI
#         prompt = f"""
#         Data:
#         "{video_data}"
#         """
#         response = openai.ChatCompletion.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}],
#             max_tokens=150,
#            temperature = 0.7,
            
#         )
#         print(response)
#         feedback = response.choices[0].message.content
#         serializer.instance.feedback = feedback
#         serializer.instance.save()
        
#         # Delete the temporary files
#         os.remove(video_output_path)
#         os.remove(video_data_path)
        
#         # Prepare the response
#         # processed_video_url = request.build_absolute_uri(serializer.instance.processed_video.url)
#         response_data = {
#             # "processed_video_url": processed_video_url,
#             "id":serializer.instance.id
#             # "feedback": feedback
#         }
        
#         return Response(response_data, status=status.HTTP_201_CREATED, headers=headers)
    
# class ProcessedVideoView(generics.RetrieveAPIView):
#     queryset = Video.objects.all()
#     serializer_class = VideoSerializer
    
#     def retrieve(self, request, *args, **kwargs):
#         video:Video = self.get_object()
#         if video.processed_video:
#             print(video.processed_video.path.split("/")[-1])
#             response = FileResponse(open(video.processed_video.path, 'rb'), as_attachment=True, filename=video.processed_video.path.split("/")[-1])
#             response['Content-Type'] = 'video/mp4'
            
#             return response
#         else:
#             return Response({"error": "Processed video not available"}, status=status.HTTP_404_NOT_FOUND)
        
        
# class FeedBackView(generics.RetrieveAPIView):
#     queryset = Video.objects.all()
#     serializer_class = VideoSerializer
    
#     def retrieve(self, request, *args, **kwargs):
#         video = self.get_object()
#         if video.feedback:
#             response = video.feedback
            
#             return Response({"res" : response},status=status.HTTP_200_OK)
#         else:
#             return Response({"error": "Processed video not available"}, status=status.HTTP_404_NOT_FOUND)


# # Updated views.py
# from rest_framework import generics, status
# from rest_framework.response import Response
# from django.http import FileResponse
# from rest_framework.exceptions import ValidationError
# from .video_processing import process_video
# from .models import Video
# from .serializer import VideoSerializer
# from .llm_coach import TennisCoachAnalyzer
# import os

# class VideoUploadView(generics.CreateAPIView):
#     queryset = Video.objects.all()
#     serializer_class = VideoSerializer
    
#     def create(self, request, *args, **kwargs):
#         serializer = self.get_serializer(data=request.data)
#         serializer.is_valid(raise_exception=True)
        
#         # Custom validation: File size and type restrictions
#         video_file = request.FILES['video_file']
#         allowed_types = ['video/mp4', 'video/webm', 'video/avi']
        
#         if video_file.size > 30 * 1024 * 1024:  # 30MB limit
#             raise ValidationError("Video file size exceeds the limit.")
#         if not video_file.content_type in allowed_types:
#             raise ValidationError("Invalid file type. Allowed types: {}".format(', '.join(allowed_types)))
        
#         self.perform_create(serializer)
#         headers = self.get_success_headers(serializer.data)
#         video_name = video_file.name.split(".")[0]
        
#         # Process the video
#         video_data, video_output_path, video_data_path = process_video(serializer.instance.id, video_name)
        
#         # Save the processed video and JSON file
#         serializer.instance.processed_video.save(f'{video_name}_processed.mp4', open(video_output_path, 'rb'))
#         serializer.instance.video_data_json.save(f'{video_name}_data.json', open(video_data_path, 'rb'))
        
#         # Extract shot types from video data
#         shot_types = [frame_data.get('class_name', 'unknown') for frame_data in video_data.values() if frame_data.get('class_name')]
        
#         # Initialize AI coach analyzer
#         coach_analyzer = TennisCoachAnalyzer()
        
#         # Generate comprehensive feedback
#         coaching_analysis = coach_analyzer.generate_comprehensive_feedback(video_data, shot_types)
        
#         # Save feedback to database
#         serializer.instance.feedback = coaching_analysis['feedback']
#         serializer.instance.save()
        
#         # Delete the temporary files
#         os.remove(video_output_path)
#         os.remove(video_data_path)
        
#         # Prepare the response
#         response_data = {
#             "id": serializer.instance.id,
#             "overall_score": coaching_analysis['overall_score'],
#             "detailed_scores": coaching_analysis['detailed_scores'],
#             "frames_analyzed": coaching_analysis['frames_analyzed'],
#             "shot_types_detected": coaching_analysis['shot_types_detected']
#         }
        
#         return Response(response_data, status=status.HTTP_201_CREATED, headers=headers)

# class ProcessedVideoView(generics.RetrieveAPIView):
#     queryset = Video.objects.all()
#     serializer_class = VideoSerializer
    
#     def retrieve(self, request, *args, **kwargs):
#         video: Video = self.get_object()
#         if video.processed_video:
#             response = FileResponse(
#                 open(video.processed_video.path, 'rb'), 
#                 as_attachment=True, 
#                 filename=video.processed_video.path.split("/")[-1]
#             )
#             response['Content-Type'] = 'video/mp4'
#             return response
#         else:
#             return Response({"error": "Processed video not available"}, status=status.HTTP_404_NOT_FOUND)

# class FeedBackView(generics.RetrieveAPIView):
#     queryset = Video.objects.all()
#     serializer_class = VideoSerializer
    
#     def retrieve(self, request, *args, **kwargs):
#         video = self.get_object()
#         if video.feedback:
#             return Response({"feedback": video.feedback}, status=status.HTTP_200_OK)
#         else:
#             return Response({"error": "Feedback not available"}, status=status.HTTP_404_NOT_FOUND)



# # Updated views.py with enhanced multi-sport processing
# from rest_framework import generics, status
# from rest_framework.response import Response
# from django.http import FileResponse
# from rest_framework.exceptions import ValidationError
# from .video_processing import process_video
# from .models import Video
# from .serializer import VideoSerializer
# from .multi_sport_coach import EnhancedMultiSportCoachAnalyzer
# import os
# from django.utils import timezone

# class VideoUploadView(generics.CreateAPIView):
#     queryset = Video.objects.all()
#     serializer_class = VideoSerializer
    
#     def create(self, request, *args, **kwargs):
#         serializer = self.get_serializer(data=request.data)
#         serializer.is_valid(raise_exception=True)
        
#         # Custom validation
#         video_file = request.FILES['video_file']
#         allowed_types = ['video/mp4', 'video/webm', 'video/avi']
        
#         if video_file.size > 50 * 1024 * 1024:  # 50MB limit for multi-sport
#             raise ValidationError("Video file size exceeds the limit.")
#         if not video_file.content_type in allowed_types:
#             raise ValidationError("Invalid file type. Allowed types: {}".format(', '.join(allowed_types)))
        
#         self.perform_create(serializer)
#         headers = self.get_success_headers(serializer.data)
#         video_name = video_file.name.split(".")[0]
        
#         # Update status to processing
#         serializer.instance.analysis_status = 'processing'
#         serializer.instance.save()
        
#         try:
#             # Process the video (this now works for ALL sports with pose estimation)
#             video_data, video_output_path, video_data_path = process_video(serializer.instance.id, video_name)
            
#             # Save processed files
#             serializer.instance.processed_video.save(f'{video_name}_processed.mp4', open(video_output_path, 'rb'))
#             serializer.instance.video_data_json.save(f'{video_name}_data.json', open(video_data_path, 'rb'))
            
#             # Enhanced multi-sport analysis
#             multi_sport_analyzer = EnhancedMultiSportCoachAnalyzer()
#             analysis_result = multi_sport_analyzer.analyze_video(serializer.instance.video_file.path, video_data)
            
#             # Update video instance with analysis results
#             serializer.instance.sport_detected = analysis_result['sport_detected']
#             serializer.instance.sport_confidence = analysis_result['sport_confidence']
#             serializer.instance.analysis_status = analysis_result['analysis_status']
#             serializer.instance.frames_analyzed = analysis_result['frames_analyzed']
#             serializer.instance.feedback = analysis_result['feedback']
#             serializer.instance.processed_at = timezone.now()
            
#             # Add tennis-specific data if available
#             if analysis_result['analysis_status'] == 'complete':
#                 serializer.instance.overall_score = analysis_result['overall_score']
#                 serializer.instance.detailed_scores = analysis_result['detailed_scores']
#                 serializer.instance.shot_types_detected = analysis_result.get('shot_types_detected', [])
            
#             serializer.instance.save()
            
#             # Clean up temporary files
#             os.remove(video_output_path)
#             os.remove(video_data_path)
            
#             # Prepare response based on analysis type
#             response_data = {
#                 "id": serializer.instance.id,
#                 "sport_detected": analysis_result['sport_detected'],
#                 "sport_confidence": analysis_result['sport_confidence'],
#                 "analysis_status": analysis_result['analysis_status'],
#                 "frames_analyzed": analysis_result['frames_analyzed'],
#                 "total_frames": analysis_result.get('total_frames', len(video_data)),
#                 "processing_complete": analysis_result['processing_complete']
#             }
            
#             # Add detailed analysis for tennis
#             if analysis_result['analysis_status'] == 'complete':
#                 response_data.update({
#                     "overall_score": analysis_result['overall_score'],
#                     "detailed_scores": analysis_result['detailed_scores'],
#                     "shot_types_detected": analysis_result.get('shot_types_detected', [])
#                 })
            
#             # Add training info for sports under development
#             elif analysis_result['analysis_status'] in ['under_training', 'general_processing']:
#                 response_data.update({
#                     "training_progress": analysis_result.get('training_progress'),
#                     "estimated_completion": analysis_result.get('estimated_completion'),
#                     "basic_metrics": analysis_result.get('basic_metrics', {})
#                 })
            
#             return Response(response_data, status=status.HTTP_201_CREATED, headers=headers)
            
#         except Exception as e:
#             # Update status to failed
#             serializer.instance.analysis_status = 'failed'
#             serializer.instance.feedback = f"Processing failed: {str(e)}"
#             serializer.instance.save()
            
#             return Response({
#                 "error": "Video processing failed",
#                 "details": str(e),
#                 "id": serializer.instance.id
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# class ProcessedVideoView(generics.RetrieveAPIView):
#     queryset = Video.objects.all()
#     serializer_class = VideoSerializer
    
#     def retrieve(self, request, *args, **kwargs):
#         video: Video = self.get_object()
#         if video.processed_video:
#             response = FileResponse(
#                 open(video.processed_video.path, 'rb'), 
#                 as_attachment=True, 
#                 filename=video.processed_video.path.split("/")[-1]
#             )
#             response['Content-Type'] = 'video/mp4'
#             return response
#         else:
#             return Response({"error": "Processed video not available"}, status=status.HTTP_404_NOT_FOUND)

# class FeedBackView(generics.RetrieveAPIView):
#     queryset = Video.objects.all()
#     serializer_class = VideoSerializer
    
#     def retrieve(self, request, *args, **kwargs):
#         video = self.get_object()
#         if video.feedback:
#             response_data = {
#                 "feedback": video.feedback,
#                 "sport_detected": video.sport_detected,
#                 "analysis_status": video.analysis_status,
#                 "frames_analyzed": video.frames_analyzed
#             }
            
#             # Add additional data based on analysis status
#             if video.analysis_status == 'complete':
#                 response_data.update({
#                     "overall_score": video.overall_score,
#                     "detailed_scores": video.detailed_scores,
#                     "shot_types_detected": video.shot_types_detected
#                 })
            
#             return Response(response_data, status=status.HTTP_200_OK)
#         else:
#             return Response({"error": "Feedback not available"}, status=status.HTTP_404_NOT_FOUND)

# # New endpoint to get video analysis status
# class VideoStatusView(generics.RetrieveAPIView):
#     queryset = Video.objects.all()
#     serializer_class = VideoSerializer
    
#     def retrieve(self, request, *args, **kwargs):
#         video = self.get_object()
        
#         response_data = {
#             "id": video.id,
#             "sport_detected": video.sport_detected,
#             "sport_confidence": video.sport_confidence,
#             "analysis_status": video.analysis_status,
#             "frames_analyzed": video.frames_analyzed,
#             "uploaded_at": video.uploaded_at,
#             "processed_at": video.processed_at,
#         }
        
#         if video.analysis_status == 'complete':
#             response_data.update({
#                 "overall_score": video.overall_score,
#                 "detailed_scores": video.detailed_scores,
#                 "shot_types_detected": video.shot_types_detected
#             })
        
#         return Response(response_data, status=status.HTTP_200_OK)


from rest_framework import generics, status
from rest_framework.response import Response
from .multi_sport_coach import SimplifiedMultiSportCoachAnalyzer
from django.http import FileResponse
from rest_framework.exceptions import ValidationError
from .video_processing import process_video
from rest_framework.decorators import api_view
from .models import Video
from .serializer import VideoSerializer
# from .tasks import process_video_async
# from celery.result import AsyncResult 
from django.utils import timezone
import os
import logging

logger = logging.getLogger(__name__)

class VideoUploadView(generics.CreateAPIView):
    queryset = Video.objects.all()
    serializer_class = VideoSerializer
    
    # def create(self, request, *args, **kwargs):
    #     # Validate sport_type is provided
    #     sport_type = request.data.get('sport_type')
    #     if not sport_type or sport_type not in ['tennis', 'running', 'soccer']:
    #         return Response({
    #             "error": "sport_type is required and must be one of: tennis, running, soccer"
    #         }, status=status.HTTP_400_BAD_REQUEST)
        
    #     serializer = self.get_serializer(data=request.data)
    #     serializer.is_valid(raise_exception=True)
        
    #     # Custom validation
    #     video_file = request.FILES['video_file']
    #     allowed_types = ['video/mp4', 'video/webm', 'video/avi']
        
    #     if video_file.size > 50 * 1024 * 1024:  # 50MB limit
    #         raise ValidationError("Video file size exceeds the limit.")
    #     if not video_file.content_type in allowed_types:
    #         raise ValidationError("Invalid file type. Allowed types: {}".format(', '.join(allowed_types)))
        
    #     # Save video instance with initial status
    #     self.perform_create(serializer)
    #     video_instance = serializer.instance
    #     video_name = video_file.name.split(".")[0]
        
    #     # Queue the processing task (non-blocking)
    #     task = process_video_async.delay(
    #         video_instance.id, 
    #         video_name, 
    #         sport_type
    #     )
        
    #     # Store task ID for status tracking
    #     video_instance.analysis_status = 'pending'
    #     video_instance.save()
        
    #     # Return immediate response
    #     return Response({
    #         'id': video_instance.id,
    #         'task_id': task.id,
    #         'sport_type': sport_type,
    #         'status': 'queued',
    #         'message': 'Video uploaded successfully. Processing started in background.',
    #         'status_check_url': f'/api/video/{video_instance.id}/status/'
    #     }, status=status.HTTP_202_ACCEPTED)
    
    def create(self, request, *args, **kwargs):
        # Validate sport_type is provided
        sport_type = request.data.get('sport_type')
        if not sport_type or sport_type not in ['tennis', 'running', 'soccer']:
            return Response({
                "error": "sport_type is required and must be one of: tennis, running, soccer"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Custom validation
        video_file = request.FILES['video_file']
        allowed_types = ['video/mp4', 'video/webm', 'video/avi']
        
        if video_file.size > 50 * 1024 * 1024:  # 50MB limit
            raise ValidationError("Video file size exceeds the limit.")
        if not video_file.content_type in allowed_types:
            raise ValidationError("Invalid file type. Allowed types: {}".format(', '.join(allowed_types)))
        
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        video_name = video_file.name.split(".")[0]
        
        # Update status to processing
        serializer.instance.analysis_status = 'processing'
        serializer.instance.save()
        
        try:
            # Process the video (pose estimation works for all sports)
            video_data, video_output_path, video_data_path = process_video(serializer.instance.id, video_name)
            
            # Save processed files
            serializer.instance.processed_video.save(f'{video_name}_processed.mp4', open(video_output_path, 'rb'))
            serializer.instance.video_data_json.save(f'{video_name}_data.json', open(video_data_path, 'rb'))
            
            # Multi-sport analysis using user-specified sport type
            multi_sport_analyzer = SimplifiedMultiSportCoachAnalyzer()
            analysis_result = multi_sport_analyzer.analyze_video(sport_type, video_data)
            print("video analysis done")
            # Update video instance with analysis results
            serializer.instance.analysis_status = analysis_result['analysis_status']
            serializer.instance.frames_analyzed = analysis_result['frames_analyzed']
            serializer.instance.feedback = analysis_result['feedback']
            serializer.instance.processed_at = timezone.now()
            
            # Add tennis-specific data if available
            if analysis_result['analysis_status'] == 'complete':
                serializer.instance.overall_score = analysis_result['overall_score']
                serializer.instance.detailed_scores = analysis_result['detailed_scores']
                serializer.instance.shot_types_detected = analysis_result.get('shot_types_detected', [])
            else:
                # Add training data for other sports
                serializer.instance.training_progress = analysis_result.get('training_progress')
                serializer.instance.estimated_completion = analysis_result.get('estimated_completion')
                serializer.instance.basic_metrics = analysis_result.get('basic_metrics')
            
            serializer.instance.save()
            print("instance saved")
            # Clean up temporary files
            os.remove(video_output_path)
            os.remove(video_data_path)
            
            # Prepare response based on analysis type
            response_data = {
                "id": serializer.instance.id,
                "sport_type": sport_type,
                "analysis_status": analysis_result['analysis_status'],
                "frames_analyzed": analysis_result['frames_analyzed'],
                "total_frames": analysis_result.get('total_frames', len(video_data)),
                "processing_complete": analysis_result['processing_complete']
            }
            
            # Add detailed analysis for tennis
            if analysis_result['analysis_status'] == 'complete':
                response_data.update({
                    "overall_score": analysis_result['overall_score'],
                    "detailed_scores": analysis_result['detailed_scores'],
                    "shot_types_detected": analysis_result.get('shot_types_detected', [])
                })
            
            # Add training info for sports under development
            elif analysis_result['analysis_status'] == 'under_training':
                response_data.update({
                    "training_progress": analysis_result.get('training_progress'),
                    "estimated_completion": analysis_result.get('estimated_completion'),
                    "basic_metrics": analysis_result.get('basic_metrics', {})
                })
            
            return Response(response_data, status=status.HTTP_201_CREATED, headers=headers)
            
        except Exception as e:
            # Update status to failed
            serializer.instance.analysis_status = 'failed'
            serializer.instance.feedback = f"Processing failed: {str(e)}"
            serializer.instance.save()
            
            return Response({
                "error": "Video processing failed",
                "details": str(e),
                "id": serializer.instance.id
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ProcessedVideoView(generics.RetrieveAPIView):
    queryset = Video.objects.all()
    serializer_class = VideoSerializer
    
    def retrieve(self, request, *args, **kwargs):
        video: Video = self.get_object()
        if video.processed_video:
            response = FileResponse(
                open(video.processed_video.path, 'rb'), 
                as_attachment=True, 
                filename=video.processed_video.path.split("/")[-1]
            )
            response['Content-Type'] = 'video/mp4'
            return response
        else:
            return Response({"error": "Processed video not available"}, status=status.HTTP_404_NOT_FOUND)

class FeedBackView(generics.RetrieveAPIView):
    queryset = Video.objects.all()
    serializer_class = VideoSerializer
    
    def retrieve(self, request, *args, **kwargs):
        video = self.get_object()
        if video.feedback:
            response_data = {
                "feedback": video.feedback,
                "sport_type": video.sport_type,
                "analysis_status": video.analysis_status,
                "frames_analyzed": video.frames_analyzed
            }
            
            # Add additional data based on analysis status
            if video.analysis_status == 'complete':
                response_data.update({
                    "overall_score": video.overall_score,
                    "detailed_scores": video.detailed_scores,
                    "shot_types_detected": video.shot_types_detected
                })
            
            return Response(response_data, status=status.HTTP_200_OK)
        else:
            return Response({"error": "Feedback not available"}, status=status.HTTP_404_NOT_FOUND)
        
@api_view(['GET'])
def video_status(request, pk):
    """Check video processing status"""
    try:
        video = Video.objects.get(id=pk)
        
        response_data = {
            'id': video.id,
            'sport_type': video.sport_type,
            'analysis_status': video.analysis_status,
            'frames_analyzed': video.frames_analyzed,
            'uploaded_at': video.uploaded_at,
            'processed_at': video.processed_at,
            'processing_status_display': video.processing_status_display
        }
        
        # If completed, return full results
        if video.analysis_status == 'complete':
            response_data.update({
                'overall_score': video.overall_score,
                'detailed_scores': video.detailed_scores,
                'shot_types_detected': video.shot_types_detected,
                'feedback_available': bool(video.feedback)
            })
        elif video.analysis_status == 'under_training':
            response_data.update({
                'training_progress': video.training_progress,
                'estimated_completion': video.estimated_completion,
                'basic_metrics': video.basic_metrics
            })
        
        return Response(response_data)
        
    except Video.DoesNotExist:
        return Response({'error': 'Video not found'}, 
                       status=status.HTTP_404_NOT_FOUND)
        
# class VideoStatusView(generics.RetrieveAPIView):
#     queryset = Video.objects.all()
#     serializer_class = VideoSerializer
    
#     def retrieve(self, request, *args, **kwargs):
#         video = self.get_object()
        
#         response_data = {
#             "id": video.id,
#             "sport_type": video.sport_type,
#             "analysis_status": video.analysis_status,
#             "frames_analyzed": video.frames_analyzed,
#             "uploaded_at": video.uploaded_at,
#             "processed_at": video.processed_at,
#         }
        
#         if video.analysis_status == 'complete':
#             response_data.update({
#                 "overall_score": video.overall_score,
#                 "detailed_scores": video.detailed_scores,
#                 "shot_types_detected": video.shot_types_detected
#             })
        
#         return Response(response_data, status=status.HTTP_200_OK)
#         # return Response("checked",status=status.HTTP_200_OK)


# Legacy endpoint for compatibility
class VideoStatusView(generics.RetrieveAPIView):
    queryset = Video.objects.all()
    serializer_class = VideoSerializer
    
    def retrieve(self, request, *args, **kwargs):
        return video_status(request, kwargs['pk'])
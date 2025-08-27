# sport_detector.py
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from dataclasses import dataclass

@dataclass
class SportDetectionResult:
    sport_type: str
    confidence: float
    reasoning: str
    detected_objects: List[str]

class SportDetector:
    """Detects sport type from video analysis"""
    
    def __init__(self):
        self.sport_keywords = {
            'tennis': ['racket', 'court', 'net', 'serve', 'forehand', 'backhand'],
            'running': ['track', 'road', 'trail', 'marathon', 'sprint'],
            'soccer': ['ball', 'goal', 'field', 'pitch', 'kick', 'dribble']
        }
        
    def detect_sport_from_video(self, video_path: str, pose_data: Dict) -> SportDetectionResult:
        """
        Detect sport type from video analysis and pose data
        """
        # Analyze video frames for sport-specific elements
        sport_indicators = self._analyze_video_content(video_path)
        
        # Analyze pose patterns
        pose_indicators = self._analyze_pose_patterns(pose_data)
        
        # Combine indicators to determine sport
        sport_scores = self._calculate_sport_scores(sport_indicators, pose_indicators)
        
        # Determine the most likely sport
        best_sport = max(sport_scores.items(), key=lambda x: x[1])
        sport_type, confidence = best_sport
        
        return SportDetectionResult(
            sport_type=sport_type,
            confidence=confidence,
            reasoning=self._generate_reasoning(sport_indicators, pose_indicators, sport_type),
            detected_objects=sport_indicators.get('objects', [])
        )
    
    def _analyze_video_content(self, video_path: str) -> Dict:
        """Analyze video content for sport-specific indicators"""
        cap = cv2.VideoCapture(video_path)
        
        # Sample frames for analysis
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = min(10, frame_count // 10) if frame_count > 10 else frame_count
        
        indicators = {
            'dominant_colors': [],
            'motion_patterns': [],
            'objects': []
        }
        
        for i in range(sample_frames):
            frame_pos = (frame_count // sample_frames) * i
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            
            if ret:
                # Analyze dominant colors (tennis courts are usually green/blue)
                dominant_color = self._get_dominant_color(frame)
                indicators['dominant_colors'].append(dominant_color)
                
                # Basic motion analysis
                if i > 0:
                    motion = self._analyze_motion(prev_frame, frame)
                    indicators['motion_patterns'].append(motion)
                
                prev_frame = frame
        
        cap.release()
        return indicators
    
    def _analyze_pose_patterns(self, pose_data: Dict) -> Dict:
        """Analyze pose patterns to identify sport characteristics"""
        if not pose_data:
            return {}
        
        patterns = {
            'arm_movements': [],
            'leg_movements': [],
            'body_positions': []
        }
        
        for frame_idx, frame_data in pose_data.items():
            if not frame_data.get('keypoints'):
                continue
                
            keypoints = frame_data['keypoints']
            
            # Analyze typical tennis movements
            tennis_score = self._score_tennis_patterns(keypoints)
            running_score = self._score_running_patterns(keypoints)
            soccer_score = self._score_soccer_patterns(keypoints)
            
            patterns['arm_movements'].append(tennis_score)
            patterns['leg_movements'].append(running_score)
            patterns['body_positions'].append(soccer_score)
        
        return patterns
    
    def _score_tennis_patterns(self, keypoints: List[Dict]) -> float:
        """Score likelihood of tennis-specific movements"""
        score = 0.0
        
        # Look for tennis-specific arm positions
        right_wrist = next((kp for kp in keypoints if kp['class_name'] == 'right_wrist'), None)
        right_elbow = next((kp for kp in keypoints if kp['class_name'] == 'right_elbow'), None)
        right_shoulder = next((kp for kp in keypoints if kp['class_name'] == 'right_shoulder'), None)
        
        if all([right_wrist, right_elbow, right_shoulder]):
            # Tennis players often have extended arm positions
            arm_extension = abs(right_wrist['y'] - right_shoulder['y'])
            if arm_extension > 0.15:  # Extended arm position
                score += 0.3
            
            # Check for racket-holding position
            wrist_height = right_wrist['y']
            shoulder_height = right_shoulder['y']
            if wrist_height < shoulder_height:  # Wrist above shoulder
                score += 0.2
        
        return min(score, 1.0)
    
    def _score_running_patterns(self, keypoints: List[Dict]) -> float:
        """Score likelihood of running-specific movements"""
        score = 0.0
        
        # Look for running-specific leg patterns
        left_knee = next((kp for kp in keypoints if kp['class_name'] == 'left_knee'), None)
        right_knee = next((kp for kp in keypoints if kp['class_name'] == 'right_knee'), None)
        left_ankle = next((kp for kp in keypoints if kp['class_name'] == 'left_ankle'), None)
        right_ankle = next((kp for kp in keypoints if kp['class_name'] == 'right_ankle'), None)
        
        if all([left_knee, right_knee, left_ankle, right_ankle]):
            # Running typically shows alternating leg patterns
            knee_separation = abs(left_knee['y'] - right_knee['y'])
            if knee_separation > 0.1:  # Legs at different heights
                score += 0.4
            
            # Forward lean is common in running
            avg_knee_y = (left_knee['y'] + right_knee['y']) / 2
            avg_ankle_y = (left_ankle['y'] + right_ankle['y']) / 2
            if avg_knee_y < avg_ankle_y:  # Knees higher than ankles
                score += 0.2
        
        return min(score, 1.0)
    
    def _score_soccer_patterns(self, keypoints: List[Dict]) -> float:
        """Score likelihood of soccer-specific movements"""
        score = 0.0
        
        # Soccer involves a lot of leg work, less arm movement
        left_ankle = next((kp for kp in keypoints if kp['class_name'] == 'left_ankle'), None)
        right_ankle = next((kp for kp in keypoints if kp['class_name'] == 'right_ankle'), None)
        left_knee = next((kp for kp in keypoints if kp['class_name'] == 'left_knee'), None)
        right_knee = next((kp for kp in keypoints if kp['class_name'] == 'right_knee'), None)
        
        if all([left_ankle, right_ankle, left_knee, right_knee]):
            # Wide stance is common in soccer
            ankle_separation = abs(left_ankle['x'] - right_ankle['x'])
            if ankle_separation > 0.2:
                score += 0.3
            
            # Check for kicking position (one leg forward)
            if abs(left_ankle['x'] - right_ankle['x']) > 0.15:
                score += 0.2
        
        return min(score, 1.0)
    
    def _get_dominant_color(self, frame) -> str:
        """Get dominant color from frame"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        
        # Count pixels in each range
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        green_pixels = cv2.countNonZero(green_mask)
        blue_pixels = cv2.countNonZero(blue_mask)
        
        total_pixels = frame.shape[0] * frame.shape[1]
        
        if green_pixels > total_pixels * 0.3:
            return 'green'
        elif blue_pixels > total_pixels * 0.2:
            return 'blue'
        else:
            return 'other'
    
    def _analyze_motion(self, prev_frame, current_frame) -> Dict:
        """Basic motion analysis between frames"""
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, None, None)
        
        return {'intensity': 'medium'}  # Simplified for now
    
    def _calculate_sport_scores(self, video_indicators: Dict, pose_indicators: Dict) -> Dict[str, float]:
        """Calculate confidence scores for each sport"""
        scores = {'tennis': 0.0, 'running': 0.0, 'soccer': 0.0}
        
        # Video-based scoring
        dominant_colors = video_indicators.get('dominant_colors', [])
        if 'green' in dominant_colors:
            scores['tennis'] += 0.3  # Tennis courts are often green
            scores['soccer'] += 0.2   # Soccer fields are green
        
        # Pose-based scoring
        if pose_indicators:
            tennis_scores = pose_indicators.get('arm_movements', [])
            running_scores = pose_indicators.get('leg_movements', [])
            soccer_scores = pose_indicators.get('body_positions', [])
            
            if tennis_scores:
                scores['tennis'] += np.mean(tennis_scores) * 0.7
            if running_scores:
                scores['running'] += np.mean(running_scores) * 0.7
            if soccer_scores:
                scores['soccer'] += np.mean(soccer_scores) * 0.7
        
        # Normalize scores
        max_score = max(scores.values()) if any(scores.values()) else 1.0
        if max_score > 0:
            scores = {sport: score / max_score for sport, score in scores.items()}
        
        return scores
    
    def _generate_reasoning(self, video_indicators: Dict, pose_indicators: Dict, sport_type: str) -> str:
        """Generate reasoning for sport detection"""
        reasons = []
        
        if sport_type == 'tennis':
            reasons.append("Detected tennis-specific arm movements and court colors")
        elif sport_type == 'running':
            reasons.append("Identified running gait patterns and leg movements")
        elif sport_type == 'soccer':
            reasons.append("Observed soccer-typical body positions and field characteristics")
        
        dominant_colors = video_indicators.get('dominant_colors', [])
        if 'green' in dominant_colors:
            reasons.append("Green surface detected (typical of tennis courts/soccer fields)")
        
        return "; ".join(reasons) if reasons else "Based on pose and movement analysis"


# # multi_sport_coach.py
# from .llm_coach import TennisCoachAnalyzer
# from .sport_detector import SportDetector, SportDetectionResult
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# import os

# class MultiSportCoachAnalyzer:
#     def __init__(self):
#         self.sport_detector = SportDetector()
#         self.tennis_analyzer = TennisCoachAnalyzer()
#         self.llm = ChatGroq(
#             model_name="llama3-70b-8192",
#             groq_api_key=os.getenv("GROQ_API_KEY"),
#             temperature=0.3,
#             max_tokens=300
#         )
    
#     def analyze_video(self, video_path: str, video_data: Dict) -> Dict:
#         """Main entry point for multi-sport analysis"""
        
#         # Step 1: Detect sport type
#         sport_detection = self.sport_detector.detect_sport_from_video(video_path, video_data)
        
#         # Step 2: Route to appropriate analyzer based on sport
#         if sport_detection.sport_type == 'tennis' and sport_detection.confidence > 0.6:
#             return self._analyze_tennis(video_data, sport_detection)
#         else:
#             return self._analyze_coming_soon_sport(sport_detection, video_data)
    
#     def _analyze_tennis(self, video_data: Dict, sport_detection: SportDetectionResult) -> Dict:
#         """Full tennis analysis using the detailed tennis analyzer"""
#         shot_types = [frame_data.get('class_name', 'unknown') for frame_data in video_data.values() if frame_data.get('class_name')]
        
#         # Use the comprehensive tennis analyzer
#         tennis_analysis = self.tennis_analyzer.generate_comprehensive_feedback(video_data, shot_types)
        
#         return {
#             "sport_detected": sport_detection.sport_type,
#             "sport_confidence": sport_detection.confidence,
#             "sport_reasoning": sport_detection.reasoning,
#             "analysis_status": "complete",
#             "overall_score": tennis_analysis['overall_score'],
#             "detailed_scores": tennis_analysis['detailed_scores'],
#             "feedback": tennis_analysis['feedback'],
#             "frames_analyzed": tennis_analysis['frames_analyzed'],
#             "shot_types_detected": tennis_analysis['shot_types_detected'],
#             "detected_objects": sport_detection.detected_objects
#         }
    
#     def _analyze_coming_soon_sport(self, sport_detection: SportDetectionResult, video_data: Dict) -> Dict:
#         """Basic analysis for sports not yet fully implemented"""
        
#         # Count frames with pose data
#         frames_with_pose = sum(1 for frame_data in video_data.values() if frame_data.get('keypoints'))
#         total_frames = len(video_data)
        
#         # Generate basic feedback
#         basic_feedback = self._generate_basic_feedback(sport_detection.sport_type, frames_with_pose, total_frames)
        
#         return {
#             "sport_detected": sport_detection.sport_type,
#             "sport_confidence": sport_detection.confidence,
#             "sport_reasoning": sport_detection.reasoning,
#             "analysis_status": "basic_detection",
#             "overall_score": None,
#             "detailed_scores": None,
#             "feedback": basic_feedback,
#             "frames_analyzed": frames_with_pose,
#             "total_frames": total_frames,
#             "detected_objects": sport_detection.detected_objects,
#             "coming_soon": True,
#             "available_features": [
#                 "Sport detection and classification",
#                 "Basic pose estimation",
#                 "Movement pattern recognition"
#             ],
#             "upcoming_features": self._get_upcoming_features(sport_detection.sport_type)
#         }
    
#     def _generate_basic_feedback(self, sport_type: str, frames_analyzed: int, total_frames: int) -> str:
#         """Generate basic feedback for non-tennis sports"""
        
#         detection_quality = "excellent" if frames_analyzed > total_frames * 0.8 else "good" if frames_analyzed > total_frames * 0.6 else "fair"
        
#         sport_messages = {
#             'running': f"""
# 🏃‍♂️ **RUNNING ANALYSIS DETECTED**

# Great news! We've successfully identified your running session and analyzed {frames_analyzed} frames with {detection_quality} pose detection quality.

# **Current Capabilities:**
# ✅ Runner detection and tracking
# ✅ Basic gait pattern recognition  
# ✅ Movement flow analysis

# **Coming Soon - Advanced Running Coach:**
# 🔜 Stride length optimization
# 🔜 Cadence analysis and recommendations
# 🔜 Running form efficiency scoring
# 🔜 Injury prevention insights
# 🔜 Personalized training recommendations

# Your running technique analysis will be available soon with our specialized running biomechanics AI coach!
#             """,
            
#             'soccer': f"""
# ⚽ **SOCCER ANALYSIS DETECTED**

# Excellent! We've identified your soccer training session and processed {frames_analyzed} frames with {detection_quality} player tracking.

# **Current Capabilities:**
# ✅ Player detection and positioning
# ✅ Basic movement pattern analysis
# ✅ Ball interaction recognition

# **Coming Soon - Advanced Soccer Coach:**
# 🔜 Touch technique analysis
# 🔜 Passing accuracy assessment  
# 🔜 Shooting form evaluation
# 🔜 Tactical positioning insights
# 🔜 Skills development recommendations

# Your complete soccer technique analysis will be available soon with our specialized soccer coaching AI!
#             """,
            
#             'unknown': f"""
# 🤖 **SPORT ANALYSIS IN PROGRESS**

# We've detected athletic movement in your video and analyzed {frames_analyzed} frames with {detection_quality} pose tracking quality.

# **Current Capabilities:**
# ✅ Athlete detection and tracking
# ✅ Movement pattern recognition
# ✅ Basic biomechanical analysis

# **Multi-Sport Platform Coming Soon:**
# 🔜 Automatic sport classification
# 🔜 Sport-specific technique analysis
# 🔜 Personalized coaching for multiple sports
# 🔜 Performance benchmarking

# Our AI coaching platform is expanding to support more sports. Stay tuned for comprehensive analysis!
#             """
#         }
        
#         return sport_messages.get(sport_type, sport_messages['unknown']).strip()
    
#     def _get_upcoming_features(self, sport_type: str) -> List[str]:
#         """Get upcoming features for each sport"""
#         features = {
#             'running': [
#                 "Stride length and cadence analysis",
#                 "Running form efficiency scoring", 
#                 "Injury risk assessment",
#                 "Training load recommendations",
#                 "Performance benchmarking"
#             ],
#             'soccer': [
#                 "Ball control technique analysis",
#                 "Shooting accuracy assessment",
#                 "Passing technique evaluation", 
#                 "Tactical positioning insights",
#                 "Skills progression tracking"
#             ],
#             'unknown': [
#                 "Automatic sport detection",
#                 "Multi-sport technique analysis",
#                 "Cross-sport performance insights",
#                 "Comprehensive movement analysis"
#             ]
#         }
        
#         return features.get(sport_type, features['unknown'])


# # Updated views.py
# from rest_framework import generics, status
# from rest_framework.response import Response
# from django.http import FileResponse
# from rest_framework.exceptions import ValidationError
# from .video_processing import process_video
# from .models import Video
# from .serializer import VideoSerializer
# from .multi_sport_coach import MultiSportCoachAnalyzer
# import os

# class VideoUploadView(generics.CreateAPIView):
#     queryset = Video.objects.all()
#     serializer_class = VideoSerializer
    
#     def create(self, request, *args, **kwargs):
#         serializer = self.get_serializer(data=request.data)
#         serializer.is_valid(raise_exception=True)
        
#         # Custom validation
#         video_file = request.FILES['video_file']
#         allowed_types = ['video/mp4', 'video/webm', 'video/avi']
        
#         if video_file.size > 50 * 1024 * 1024:  # Increased to 50MB for multi-sport
#             raise ValidationError("Video file size exceeds the limit.")
#         if not video_file.content_type in allowed_types:
#             raise ValidationError("Invalid file type. Allowed types: {}".format(', '.join(allowed_types)))
        
#         self.perform_create(serializer)
#         headers = self.get_success_headers(serializer.data)
#         video_name = video_file.name.split(".")[0]
        
#         # Process the video
#         video_data, video_output_path, video_data_path = process_video(serializer.instance.id, video_name)
        
#         # Save processed files
#         serializer.instance.processed_video.save(f'{video_name}_processed.mp4', open(video_output_path, 'rb'))
#         serializer.instance.video_data_json.save(f'{video_name}_data.json', open(video_data_path, 'rb'))
        
#         # Multi-sport analysis
#         multi_sport_analyzer = MultiSportCoachAnalyzer()
#         analysis_result = multi_sport_analyzer.analyze_video(serializer.instance.video_file.path, video_data)
        
#         # Save feedback
#         serializer.instance.feedback = analysis_result['feedback']
#         serializer.instance.save()
        
#         # Clean up temporary files
#         os.remove(video_output_path)
#         os.remove(video_data_path)
        
#         # Prepare response
#         response_data = {
#             "id": serializer.instance.id,
#             "sport_detected": analysis_result['sport_detected'],
#             "sport_confidence": analysis_result['sport_confidence'],
#             "analysis_status": analysis_result['analysis_status'],
#             "frames_analyzed": analysis_result['frames_analyzed']
#         }
        
#         # Add detailed analysis for tennis
#         if analysis_result['analysis_status'] == 'complete':
#             response_data.update({
#                 "overall_score": analysis_result['overall_score'],
#                 "detailed_scores": analysis_result['detailed_scores'],
#                 "shot_types_detected": analysis_result.get('shot_types_detected', [])
#             })
#         else:
#             # Add coming soon info
#             response_data.update({
#                 "coming_soon": True,
#                 "available_features": analysis_result['available_features'],
#                 "upcoming_features": analysis_result['upcoming_features']
#             })
        
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
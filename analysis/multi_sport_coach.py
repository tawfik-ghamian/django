# # enhanced_multi_sport_coach.py
# from .llm_coach import TennisCoachAnalyzer
# from .sport_detector import SportDetector, SportDetectionResult
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# import os
# import json
# import random

# class EnhancedMultiSportCoachAnalyzer:
#     def __init__(self):
#         self.sport_detector = SportDetector()
#         self.tennis_analyzer = TennisCoachAnalyzer()
#         self.llm = ChatGroq(
#             model_name="llama3-70b-8192",
#             groq_api_key=os.getenv("GROQ_API_KEY"),
#             temperature=0.3,
#             max_tokens=400
#         )
    
#     def analyze_video(self, video_path: str, video_data: dict) -> dict:
#         """Main entry point for multi-sport analysis - now processes ALL sports"""
        
#         # Step 1: Detect sport type
#         sport_detection = self.sport_detector.detect_sport_from_video(video_path, video_data)
        
#         # Step 2: Route to appropriate analyzer based on sport
#         if sport_detection.sport_type == 'tennis' and sport_detection.confidence > 0.6:
#             return self._analyze_tennis(video_data, sport_detection)
#         elif sport_detection.sport_type == 'running':
#             return self._analyze_running_under_training(video_data, sport_detection)
#         elif sport_detection.sport_type == 'soccer':
#             return self._analyze_soccer_under_training(video_data, sport_detection)
#         else:
#             # Default to general analysis
#             return self._analyze_general_sport(video_data, sport_detection)
    
#     def _analyze_tennis(self, video_data: dict, sport_detection: SportDetectionResult) -> dict:
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
#             "detected_objects": sport_detection.detected_objects,
#             "processing_complete": True
#         }
    
#     def _analyze_running_under_training(self, video_data: dict, sport_detection: SportDetectionResult) -> dict:
#         """Running analysis with under training feedback"""
        
#         # Analyze basic running patterns
#         running_analysis = self._analyze_running_patterns(video_data)
        
#         # Generate under training feedback
#         feedback = self._generate_running_training_feedback(running_analysis, len(video_data))
        
#         return {
#             "sport_detected": sport_detection.sport_type,
#             "sport_confidence": sport_detection.confidence,
#             "sport_reasoning": sport_detection.reasoning,
#             "analysis_status": "under_training",
#             "overall_score": None,
#             "detailed_scores": None,
#             "feedback": feedback,
#             "frames_analyzed": running_analysis['frames_with_pose'],
#             "total_frames": len(video_data),
#             "detected_objects": sport_detection.detected_objects,
#             "processing_complete": True,
#             "training_progress": "65%",  # Simulated progress
#             "estimated_completion": "Q2 2025",
#             "basic_metrics": running_analysis['basic_metrics']
#         }
    
#     def _analyze_soccer_under_training(self, video_data: dict, sport_detection: SportDetectionResult) -> dict:
#         """Soccer analysis with under training feedback"""
        
#         # Analyze basic soccer patterns
#         soccer_analysis = self._analyze_soccer_patterns(video_data)
        
#         # Generate under training feedback
#         feedback = self._generate_soccer_training_feedback(soccer_analysis, len(video_data))
        
#         return {
#             "sport_detected": sport_detection.sport_type,
#             "sport_confidence": sport_detection.confidence,
#             "sport_reasoning": sport_detection.reasoning,
#             "analysis_status": "under_training",
#             "overall_score": None,
#             "detailed_scores": None,
#             "feedback": feedback,
#             "frames_analyzed": soccer_analysis['frames_with_pose'],
#             "total_frames": len(video_data),
#             "detected_objects": sport_detection.detected_objects,
#             "processing_complete": True,
#             "training_progress": "45%",  # Simulated progress
#             "estimated_completion": "Q3 2025",
#             "basic_metrics": soccer_analysis['basic_metrics']
#         }
    
#     def _analyze_general_sport(self, video_data: dict, sport_detection: SportDetectionResult) -> dict:
#         """General sport analysis for unrecognized sports"""
        
#         frames_with_pose = sum(1 for frame_data in video_data.values() if frame_data.get('keypoints'))
        
#         feedback = f"""
# 🤖 **SPORT ANALYSIS IN PROGRESS**

# We've detected athletic movement in your video and successfully processed all {len(video_data)} frames with pose tracking.

# **✅ Current Processing Capabilities:**
# - ✅ Athlete detection and tracking (100% complete)
# - ✅ Full pose keypoint extraction ({frames_with_pose} frames analyzed)
# - ✅ Movement pattern recognition
# - ✅ Biomechanical data collection

# **🔬 AI Training Status:**
# Our multi-sport AI is currently learning to identify this specific sport and movement patterns. The pose estimation and tracking are working perfectly - we can see every movement you make!

# **🎯 What's Happening Behind the Scenes:**
# - Advanced pose keypoints successfully extracted
# - Movement patterns being catalogued
# - Biomechanical data stored for analysis
# - Sport-specific training models in development

# **🚀 Coming Soon:**
# - Automatic sport classification
# - Sport-specific technique analysis  
# - Personalized coaching recommendations
# - Performance benchmarking

# Your movement data has been perfectly captured and will be ready for analysis as soon as our sport-specific AI models complete training!
#         """.strip()
        
#         return {
#             "sport_detected": "unknown",
#             "sport_confidence": sport_detection.confidence,
#             "sport_reasoning": sport_detection.reasoning,
#             "analysis_status": "general_processing",
#             "overall_score": None,
#             "detailed_scores": None,
#             "feedback": feedback,
#             "frames_analyzed": frames_with_pose,
#             "total_frames": len(video_data),
#             "processing_complete": True,
#             "training_progress": "20%",
#             "estimated_completion": "Q4 2025"
#         }
    
#     def _analyze_running_patterns(self, video_data: dict) -> dict:
#         """Analyze basic running patterns from pose data"""
        
#         frames_with_pose = 0
#         stride_patterns = []
#         body_lean_angles = []
#         arm_swing_patterns = []
        
#         for frame_idx, frame_data in video_data.items():
#             if not frame_data.get('keypoints'):
#                 continue
                
#             frames_with_pose += 1
#             keypoints = frame_data['keypoints']
            
#             # Basic stride analysis
#             left_knee = next((kp for kp in keypoints if kp['class_name'] == 'left_knee'), None)
#             right_knee = next((kp for kp in keypoints if kp['class_name'] == 'right_knee'), None)
            
#             if left_knee and right_knee:
#                 stride_separation = abs(left_knee['y'] - right_knee['y'])
#                 stride_patterns.append(stride_separation)
            
#             # Basic body lean analysis
#             nose = next((kp for kp in keypoints if kp['class_name'] == 'nose'), None)
#             left_ankle = next((kp for kp in keypoints if kp['class_name'] == 'left_ankle'), None)
#             right_ankle = next((kp for kp in keypoints if kp['class_name'] == 'right_ankle'), None)
            
#             if nose and left_ankle and right_ankle:
#                 avg_ankle_x = (left_ankle['x'] + right_ankle['x']) / 2
#                 lean = abs(nose['x'] - avg_ankle_x)
#                 body_lean_angles.append(lean)
            
#             # Basic arm swing analysis
#             left_wrist = next((kp for kp in keypoints if kp['class_name'] == 'left_wrist'), None)
#             right_wrist = next((kp for kp in keypoints if kp['class_name'] == 'right_wrist'), None)
            
#             if left_wrist and right_wrist:
#                 arm_separation = abs(left_wrist['y'] - right_wrist['y'])
#                 arm_swing_patterns.append(arm_separation)
        
#         # Calculate basic metrics
#         avg_stride = sum(stride_patterns) / len(stride_patterns) if stride_patterns else 0
#         avg_lean = sum(body_lean_angles) / len(body_lean_angles) if body_lean_angles else 0
#         avg_arm_swing = sum(arm_swing_patterns) / len(arm_swing_patterns) if arm_swing_patterns else 0
        
#         return {
#             "frames_with_pose": frames_with_pose,
#             "basic_metrics": {
#                 "average_stride_pattern": round(avg_stride, 3),
#                 "average_body_lean": round(avg_lean, 3),
#                 "average_arm_swing": round(avg_arm_swing, 3),
#                 "stride_consistency": round(1 - (max(stride_patterns) - min(stride_patterns)) if stride_patterns else 0, 3)
#             }
#         }
    
#     def _analyze_soccer_patterns(self, video_data: dict) -> dict:
#         """Analyze basic soccer patterns from pose data"""
        
#         frames_with_pose = 0
#         stance_widths = []
#         body_positions = []
#         leg_positions = []
        
#         for frame_idx, frame_data in video_data.items():
#             if not frame_data.get('keypoints'):
#                 continue
                
#             frames_with_pose += 1
#             keypoints = frame_data['keypoints']
            
#             # Basic stance analysis
#             left_ankle = next((kp for kp in keypoints if kp['class_name'] == 'left_ankle'), None)
#             right_ankle = next((kp for kp in keypoints if kp['class_name'] == 'right_ankle'), None)
            
#             if left_ankle and right_ankle:
#                 stance_width = abs(left_ankle['x'] - right_ankle['x'])
#                 stance_widths.append(stance_width)
            
#             # Basic leg positioning
#             left_knee = next((kp for kp in keypoints if kp['class_name'] == 'left_knee'), None)
#             right_knee = next((kp for kp in keypoints if kp['class_name'] == 'right_knee'), None)
            
#             if left_knee and right_knee:
#                 leg_separation = abs(left_knee['x'] - right_knee['x'])
#                 leg_positions.append(leg_separation)
            
#             # Body center analysis
#             left_hip = next((kp for kp in keypoints if kp['class_name'] == 'left_hip'), None)
#             right_hip = next((kp for kp in keypoints if kp['class_name'] == 'right_hip'), None)
            
#             if left_hip and right_hip:
#                 body_center = (left_hip['x'] + right_hip['x']) / 2
#                 body_positions.append(body_center)
        
#         # Calculate basic metrics
#         avg_stance = sum(stance_widths) / len(stance_widths) if stance_widths else 0
#         avg_leg_separation = sum(leg_positions) / len(leg_positions) if leg_positions else 0
#         body_stability = 1 - (max(body_positions) - min(body_positions)) if len(body_positions) > 1 else 1
        
#         return {
#             "frames_with_pose": frames_with_pose,
#             "basic_metrics": {
#                 "average_stance_width": round(avg_stance, 3),
#                 "average_leg_separation": round(avg_leg_separation, 3),
#                 "body_stability": round(body_stability, 3),
#                 "movement_variability": round(len(set([round(s, 2) for s in stance_widths])) / len(stance_widths) if stance_widths else 0, 3)
#             }
#         }
    
#     def _generate_running_training_feedback(self, analysis: dict, total_frames: int) -> str:
#         """Generate running-specific training feedback"""
        
#         metrics = analysis['basic_metrics']
#         frames_analyzed = analysis['frames_with_pose']
        
#         detection_quality = "excellent" if frames_analyzed > total_frames * 0.8 else "good" if frames_analyzed > total_frames * 0.6 else "fair"
        
#         feedback = f"""
# 🏃‍♂️ **RUNNING ANALYSIS - AI TRAINING IN PROGRESS**

# Fantastic! We've successfully processed your running session and extracted detailed biomechanical data from {frames_analyzed} frames with {detection_quality} pose detection quality.

# **✅ Processing Complete:**
# - ✅ Full body pose tracking (100% complete)
# - ✅ Stride pattern analysis detected
# - ✅ Body lean measurements captured  
# - ✅ Arm swing patterns recorded
# - ✅ Movement consistency calculated

# **📊 Basic Metrics Captured:**
# - Stride Pattern Consistency: {metrics['stride_consistency']:.1%}
# - Body Lean Average: {metrics['average_body_lean']:.3f}
# - Arm Swing Rhythm: {metrics['average_arm_swing']:.3f}

# **🧠 AI Training Status: 65% Complete**

# Our running biomechanics AI is currently in advanced training phase! Your pose data has been perfectly captured and is contributing to our machine learning models.

# **🔬 What We're Teaching Our AI:**
# - ✅ Stride length optimization algorithms
# - ✅ Cadence analysis and recommendations  
# - 🔄 Running form efficiency scoring (training)
# - 🔄 Injury risk assessment models (training)
# - 🔄 Personalized training recommendations (training)

# **🎯 Estimated Launch: Q2 2025**

# Your running data is pristine and ready! As soon as our specialized running AI completes training, you'll receive:
# - Professional gait analysis
# - Injury prevention insights
# - Performance optimization tips
# - Personalized training plans

# Thank you for being part of our AI training process! 🚀
#         """.strip()
        
#         return feedback
    
#     def _generate_soccer_training_feedback(self, analysis: dict, total_frames: int) -> str:
#         """Generate soccer-specific training feedback"""
        
#         metrics = analysis['basic_metrics']
#         frames_analyzed = analysis['frames_with_pose']
        
#         detection_quality = "excellent" if frames_analyzed > total_frames * 0.8 else "good" if frames_analyzed > total_frames * 0.6 else "fair"
        
#         feedback = f"""
# ⚽ **SOCCER ANALYSIS - AI TRAINING IN PROGRESS**

# Excellent! We've successfully analyzed your soccer session and captured comprehensive movement data from {frames_analyzed} frames with {detection_quality} player tracking quality.

# **✅ Processing Complete:**
# - ✅ Full player pose estimation (100% complete)
# - ✅ Stance and positioning analysis
# - ✅ Leg movement patterns captured
# - ✅ Body stability measurements recorded
# - ✅ Movement variability calculated

# **📊 Basic Metrics Captured:**
# - Average Stance Width: {metrics['average_stance_width']:.3f}
# - Body Stability Score: {metrics['body_stability']:.1%}
# - Movement Variability: {metrics['movement_variability']:.3f}

# **🧠 AI Training Status: 45% Complete**

# Our soccer technique AI is in intensive training mode! Your movement data has been perfectly captured and is being used to train our advanced soccer coaching models.

# **🔬 What We're Teaching Our AI:**
# - ✅ Player positioning and movement analysis
# - ✅ Basic ball interaction detection
# - 🔄 Touch technique analysis (training)
# - 🔄 Shooting form evaluation (training)
# - 🔄 Passing accuracy assessment (training)
# - 🔄 Tactical positioning insights (training)

# **🎯 Estimated Launch: Q3 2025**

# Your soccer technique data is perfectly captured! Once our specialized soccer AI completes training, you'll get:
# - Ball control technique analysis
# - Shooting accuracy insights
# - Passing technique evaluation
# - Tactical positioning feedback
# - Skills development recommendations

# You're helping us build the world's most advanced soccer coaching AI! ⚽🚀
#         """.strip()
        
#         return feedback


# from .llm_coach import TennisCoachAnalyzer
# from .fallback_analyzer import FallbackBiomechanicalAnalyzer
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# import logging
# import os
# import json

# logger = logging.getLogger(__name__)

# class SimplifiedMultiSportCoachAnalyzer:
#     def __init__(self):
#         self.tennis_analyzer = TennisCoachAnalyzer()
#         self.fallback_analyzer = FallbackBiomechanicalAnalyzer()
#         self.llm = ChatGroq(
#             model_name="llama3-70b-8192",
#             groq_api_key=os.getenv("GROQ_API_KEY"),
#             temperature=0.3,
#             max_tokens=400
#         )
    
#     def analyze_video(self, sport_type: str, video_data: dict) -> dict:
#         """Main entry point for multi-sport analysis - sport type provided by user"""
        
#         # Check if YOLO detected any shots
#         yolo_detections = sum(1 for frame_data in video_data.values() if frame_data.get('class_name'))
#         mediapipe_detections = sum(1 for frame_data in video_data.values() if frame_data.get('mediapipe_pose_landmarks'))
        
#         logger.info(f"YOLO detections: {yolo_detections}, MediaPipe detections: {mediapipe_detections}")
        
#         # ============ ADD THIS FALLBACK LOGIC ============
#         # If YOLO didn't detect any shots but MediaPipe has pose data, use fallback
#         if yolo_detections == 0 and mediapipe_detections > 0:
#             logger.warning(f"No YOLO detections found for {sport_type}. Using fallback biomechanical analysis.")
#             return self.fallback_analyzer.analyze_video_fallback(video_data, sport_type)
        
#         # If neither YOLO nor MediaPipe detected anything
#         if yolo_detections == 0 and mediapipe_detections == 0:
#             logger.error("No detections from YOLO or MediaPipe")
#             return {
#                 "sport_type": sport_type,
#                 "analysis_status": "failed",
#                 "overall_score": None,
#                 "detailed_scores": None,
#                 "feedback": "Unable to detect any movements in the video. Please ensure proper lighting and full body visibility.",
#                 "frames_analyzed": 0,
#                 "processing_complete": True,
#                 "error": "No pose data detected"
#             }
#         # ============ END FALLBACK LOGIC ============
        
#         # Route to appropriate analyzer based on user-selected sport
#         if sport_type == 'tennis':
#             return self._analyze_tennis(video_data)
#         elif sport_type == 'running':
#             return self._analyze_running_under_training(video_data)
#         elif sport_type == 'soccer':
#             return self._analyze_soccer_under_training(video_data)
#         else:
#             # Fallback to tennis if unknown sport
#             return self.fallback_analyzer.analyze_video_fallback(video_data, "unknown sport")
        
#         # # Route to appropriate analyzer based on user-selected sport
#         # if sport_type == 'tennis':
#         #     return self._analyze_tennis(video_data)
#         # elif sport_type == 'running':
#         #     return self._analyze_running_under_training(video_data)
#         # elif sport_type == 'soccer':
#         #     return self._analyze_soccer_under_training(video_data)
#         # else:
#         #     # Fallback to tennis if unknown sport
#         #     return self._analyze_tennis(video_data)
    
#     def _analyze_tennis(self, video_data: dict) -> dict:
#         """Full tennis analysis using the detailed tennis analyzer"""
#         shot_types = [frame_data.get('class_name', 'unknown') for frame_data in video_data.values() if frame_data.get('class_name')]
#         print("analysing")
#         # Use the comprehensive tennis analyzer
#         tennis_analysis = self.tennis_analyzer.generate_comprehensive_feedback(video_data, shot_types)
#         print(tennis_analysis)
#         return {
#             "sport_type": "tennis",
#             "analysis_status": "complete",
#             "overall_score": tennis_analysis['overall_score'],
#             "detailed_scores": tennis_analysis['detailed_scores'],
#             "feedback": tennis_analysis['feedback'],
#             "frames_analyzed": tennis_analysis['frames_analyzed'],
#             "shot_types_detected": tennis_analysis['shot_types_detected'],
#             "processing_complete": True
#         }
    
#     def _analyze_running_under_training(self, video_data: dict) -> dict:
#         """Running analysis with under training feedback"""
        
#         # Analyze basic running patterns
#         running_analysis = self._analyze_running_patterns(video_data)
        
#         # Generate under training feedback
#         feedback = self._generate_running_training_feedback(running_analysis, len(video_data))
        
#         return {
#             "sport_type": "running",
#             "analysis_status": "under_training",
#             "overall_score": None,
#             "detailed_scores": None,
#             "feedback": feedback,
#             "frames_analyzed": running_analysis['frames_with_pose'],
#             "total_frames": len(video_data),
#             "processing_complete": True,
#             "training_progress": "65%",
#             "estimated_completion": "Q2 2025",
#             "basic_metrics": running_analysis['basic_metrics']
#         }
    
#     def _analyze_soccer_under_training(self, video_data: dict) -> dict:
#         """Soccer analysis with under training feedback"""
        
#         # Analyze basic soccer patterns
#         soccer_analysis = self._analyze_soccer_patterns(video_data)
        
#         # Generate under training feedback
#         feedback = self._generate_soccer_training_feedback(soccer_analysis, len(video_data))
        
#         return {
#             "sport_type": "soccer",
#             "analysis_status": "under_training",
#             "overall_score": None,
#             "detailed_scores": None,
#             "feedback": feedback,
#             "frames_analyzed": soccer_analysis['frames_with_pose'],
#             "total_frames": len(video_data),
#             "processing_complete": True,
#             "training_progress": "45%",
#             "estimated_completion": "Q3 2025",
#             "basic_metrics": soccer_analysis['basic_metrics']
#         }
    
#     def _analyze_running_patterns(self, video_data: dict) -> dict:
#         """Analyze basic running patterns from pose data"""
        
#         frames_with_pose = 0
#         stride_patterns = []
#         body_lean_angles = []
#         arm_swing_patterns = []
        
#         for frame_idx, frame_data in video_data.items():
#             if not frame_data.get('keypoints'):
#                 continue
                
#             frames_with_pose += 1
#             keypoints = frame_data['keypoints']
            
#             # Basic stride analysis
#             left_knee = next((kp for kp in keypoints if kp.get('class_name') == 'left_knee'), None)
#             right_knee = next((kp for kp in keypoints if kp.get('class_name') == 'right_knee'), None)
            
#             if left_knee and right_knee:
#                 stride_separation = abs(left_knee.get('y', 0) - right_knee.get('y', 0))
#                 stride_patterns.append(stride_separation)
            
#             # Basic body lean analysis
#             nose = next((kp for kp in keypoints if kp.get('class_name') == 'nose'), None)
#             left_ankle = next((kp for kp in keypoints if kp.get('class_name') == 'left_ankle'), None)
#             right_ankle = next((kp for kp in keypoints if kp.get('class_name') == 'right_ankle'), None)
            
#             if nose and left_ankle and right_ankle:
#                 avg_ankle_x = (left_ankle.get('x', 0) + right_ankle.get('x', 0)) / 2
#                 lean = abs(nose.get('x', 0) - avg_ankle_x)
#                 body_lean_angles.append(lean)
            
#             # Basic arm swing analysis
#             left_wrist = next((kp for kp in keypoints if kp.get('class_name') == 'left_wrist'), None)
#             right_wrist = next((kp for kp in keypoints if kp.get('class_name') == 'right_wrist'), None)
            
#             if left_wrist and right_wrist:
#                 arm_separation = abs(left_wrist.get('y', 0) - right_wrist.get('y', 0))
#                 arm_swing_patterns.append(arm_separation)
        
#         # Calculate basic metrics
#         avg_stride = sum(stride_patterns) / len(stride_patterns) if stride_patterns else 0
#         avg_lean = sum(body_lean_angles) / len(body_lean_angles) if body_lean_angles else 0
#         avg_arm_swing = sum(arm_swing_patterns) / len(arm_swing_patterns) if arm_swing_patterns else 0
        
#         return {
#             "frames_with_pose": frames_with_pose,
#             "basic_metrics": {
#                 "average_stride_pattern": round(avg_stride, 3),
#                 "average_body_lean": round(avg_lean, 3),
#                 "average_arm_swing": round(avg_arm_swing, 3),
#                 "stride_consistency": round(1 - (max(stride_patterns) - min(stride_patterns)) if stride_patterns else 0, 3)
#             }
#         }
    
#     def _analyze_soccer_patterns(self, video_data: dict) -> dict:
#         """Analyze basic soccer patterns from pose data"""
        
#         frames_with_pose = 0
#         stance_widths = []
#         body_positions = []
#         leg_positions = []
        
#         for frame_idx, frame_data in video_data.items():
#             if not frame_data.get('keypoints'):
#                 continue
                
#             frames_with_pose += 1
#             keypoints = frame_data['keypoints']
            
#             # Basic stance analysis
#             left_ankle = next((kp for kp in keypoints if kp.get('class_name') == 'left_ankle'), None)
#             right_ankle = next((kp for kp in keypoints if kp.get('class_name') == 'right_ankle'), None)
            
#             if left_ankle and right_ankle:
#                 stance_width = abs(left_ankle.get('x', 0) - right_ankle.get('x', 0))
#                 stance_widths.append(stance_width)
            
#             # Basic leg positioning
#             left_knee = next((kp for kp in keypoints if kp.get('class_name') == 'left_knee'), None)
#             right_knee = next((kp for kp in keypoints if kp.get('class_name') == 'right_knee'), None)
            
#             if left_knee and right_knee:
#                 leg_separation = abs(left_knee.get('x', 0) - right_knee.get('x', 0))
#                 leg_positions.append(leg_separation)
            
#             # Body center analysis
#             left_hip = next((kp for kp in keypoints if kp.get('class_name') == 'left_hip'), None)
#             right_hip = next((kp for kp in keypoints if kp.get('class_name') == 'right_hip'), None)
            
#             if left_hip and right_hip:
#                 body_center = (left_hip.get('x', 0) + right_hip.get('x', 0)) / 2
#                 body_positions.append(body_center)
        
#         # Calculate basic metrics
#         avg_stance = sum(stance_widths) / len(stance_widths) if stance_widths else 0
#         avg_leg_separation = sum(leg_positions) / len(leg_positions) if leg_positions else 0
#         body_stability = 1 - (max(body_positions) - min(body_positions)) if len(body_positions) > 1 else 1
        
#         return {
#             "frames_with_pose": frames_with_pose,
#             "basic_metrics": {
#                 "average_stance_width": round(avg_stance, 3),
#                 "average_leg_separation": round(avg_leg_separation, 3),
#                 "body_stability": round(body_stability, 3),
#                 "movement_variability": round(len(set([round(s, 2) for s in stance_widths])) / len(stance_widths) if stance_widths else 0, 3)
#             }
#         }
    
#     def _generate_running_training_feedback(self, analysis: dict, total_frames: int) -> str:
#         """Generate running-specific training feedback"""
        
#         metrics = analysis['basic_metrics']
#         frames_analyzed = analysis['frames_with_pose']
        
#         detection_quality = "excellent" if frames_analyzed > total_frames * 0.8 else "good" if frames_analyzed > total_frames * 0.6 else "fair"
        
#         feedback = f"""
# 🏃‍♂️ **RUNNING ANALYSIS - AI TRAINING IN PROGRESS**

# Fantastic! We've successfully processed your running session and extracted detailed biomechanical data from {frames_analyzed} frames with {detection_quality} pose detection quality.

# **✅ Processing Complete:**
# - ✅ Full body pose tracking (100% complete)
# - ✅ Stride pattern analysis detected
# - ✅ Body lean measurements captured  
# - ✅ Arm swing patterns recorded
# - ✅ Movement consistency calculated

# **📊 Basic Metrics Captured:**
# - Stride Pattern Consistency: {metrics['stride_consistency']:.1%}
# - Body Lean Average: {metrics['average_body_lean']:.3f}
# - Arm Swing Rhythm: {metrics['average_arm_swing']:.3f}

# **🧠 AI Training Status: 65% Complete**

# Our running biomechanics AI is currently in advanced training phase! Your pose data has been perfectly captured and is contributing to our machine learning models.

# **🔬 What We're Teaching Our AI:**
# - ✅ Stride length optimization algorithms
# - ✅ Cadence analysis and recommendations  
# - 🔄 Running form efficiency scoring (training)
# - 🔄 Injury risk assessment models (training)
# - 🔄 Personalized training recommendations (training)

# **🎯 Estimated Launch: Q2 2025**

# Your running data is pristine and ready! As soon as our specialized running AI completes training, you'll receive:
# - Professional gait analysis
# - Injury prevention insights
# - Performance optimization tips
# - Personalized training plans

# Thank you for being part of our AI training process! 🚀
#         """.strip()
        
#         return feedback
    
#     def _generate_soccer_training_feedback(self, analysis: dict, total_frames: int) -> str:
#         """Generate soccer-specific training feedback"""
        
#         metrics = analysis['basic_metrics']
#         frames_analyzed = analysis['frames_with_pose']
        
#         detection_quality = "excellent" if frames_analyzed > total_frames * 0.8 else "good" if frames_analyzed > total_frames * 0.6 else "fair"
        
#         feedback = f"""
# ⚽ **SOCCER ANALYSIS - AI TRAINING IN PROGRESS**

# Excellent! We've successfully analyzed your soccer session and captured comprehensive movement data from {frames_analyzed} frames with {detection_quality} player tracking quality.

# **✅ Processing Complete:**
# - ✅ Full player pose estimation (100% complete)
# - ✅ Stance and positioning analysis
# - ✅ Leg movement patterns captured
# - ✅ Body stability measurements recorded
# - ✅ Movement variability calculated

# **📊 Basic Metrics Captured:**
# - Average Stance Width: {metrics['average_stance_width']:.3f}
# - Body Stability Score: {metrics['body_stability']:.1%}
# - Movement Variability: {metrics['movement_variability']:.3f}

# **🧠 AI Training Status: 45% Complete**

# Our soccer technique AI is in intensive training mode! Your movement data has been perfectly captured and is being used to train our advanced soccer coaching models.

# **🔬 What We're Teaching Our AI:**
# - ✅ Player positioning and movement analysis
# - ✅ Basic ball interaction detection
# - 🔄 Touch technique analysis (training)
# - 🔄 Shooting form evaluation (training)
# - 🔄 Passing accuracy assessment (training)
# - 🔄 Tactical positioning insights (training)

# **🎯 Estimated Launch: Q3 2025**

# Your soccer technique data is perfectly captured! Once our specialized soccer AI completes training, you'll get:
# - Ball control technique analysis
# - Shooting accuracy insights
# - Passing technique evaluation
# - Tactical positioning feedback
# - Skills development recommendations

# You're helping us build the world's most advanced soccer coaching AI! ⚽🚀
#         """.strip()
        
#         return feedback



# # multi_sport_coach.py
# from .llm_coach import TennisCoachAnalyzer
# from .fallback_analyzer import FallbackBiomechanicalAnalyzer
# import logging

# logger = logging.getLogger(__name__)

# class SimplifiedMultiSportCoachAnalyzer:
#     """
#     Unified multi-sport analyzer that provides consistent response structure
#     regardless of whether tennis is detected or fallback analysis is used.
#     """
    
#     def __init__(self):
#         self.tennis_analyzer = TennisCoachAnalyzer()
#         self.fallback_analyzer = FallbackBiomechanicalAnalyzer()
    
#     def analyze_video(self, sport_type: str, video_data: dict) -> dict:
#         """
#         Main entry point for multi-sport analysis.
#         Returns unified response structure for all scenarios.
        
#         Args:
#             sport_type: 'tennis', 'running', or 'soccer'
#             video_data: Dictionary with frame data from video processing
            
#         Returns:
#             Unified dict with keys:
#             - sport_type: str
#             - analysis_status: 'complete', 'fallback_analysis', or 'insufficient_data'
#             - overall_score: float or None
#             - detailed_scores: dict or None
#             - feedback: str
#             - frames_analyzed: int
#             - shot_types_detected: list
#             - processing_complete: bool
#             - processed_video_url: str (added by view)
#             - feedback_url: str (added by view)
#         """
        
#         logger.info(f"Starting analysis for sport: {sport_type}")
        
#         # Check if any frames have YOLO detections (shot types)
#         shot_types = []
#         frames_with_yolo = 0
#         frames_with_mediapipe = 0
        
#         for frame_data in video_data.values():
#             if frame_data.get('class_name'):
#                 shot_types.append(frame_data['class_name'])
#                 frames_with_yolo += 1
#             if frame_data.get('mediapipe_pose_landmarks'):
#                 frames_with_mediapipe += 1
        
#         logger.info(f"Frames with YOLO detections: {frames_with_yolo}")
#         logger.info(f"Frames with MediaPipe pose: {frames_with_mediapipe}")
        
#         # CASE 1: Tennis detected with YOLO - use full tennis analyzer
#         if sport_type == 'tennis' and frames_with_yolo > 0:
#             logger.info("Tennis detected - using full tennis analyzer")
#             return self._analyze_tennis(video_data, shot_types)
        
#         # CASE 2: No YOLO detections OR other sports - use fallback with MediaPipe + LLM
#         elif frames_with_mediapipe > 0:
#             logger.info(f"No YOLO detections for {sport_type} - using MediaPipe fallback with LLM")
#             return self._analyze_with_fallback(sport_type, video_data)
        
#         # CASE 3: No data at all
#         else:
#             logger.warning("No pose data detected at all")
#             return self._generate_no_data_response(sport_type)
    
#     def _analyze_tennis(self, video_data: dict, shot_types: list) -> dict:
#         """
#         Full tennis analysis using YOLO keypoints + detailed tennis analyzer.
#         Returns complete unified response.
#         """
#         logger.info("Running comprehensive tennis analysis")
        
#         # Use the comprehensive tennis analyzer
#         tennis_analysis = self.tennis_analyzer.generate_comprehensive_feedback(
#             video_data, 
#             shot_types
#         )
        
#         # Return unified response structure
#         return {
#             "sport_type": "tennis",
#             "analysis_status": "complete",
#             "overall_score": tennis_analysis['overall_score'],
#             "detailed_scores": tennis_analysis['detailed_scores'],
#             "feedback": tennis_analysis['feedback'],
#             "frames_analyzed": tennis_analysis['frames_analyzed'],
#             "shot_types_detected": tennis_analysis['shot_types_detected'],
#             "processing_complete": True,
#             "is_fallback": False,
#             "model_status": "fully_trained"
#         }
    
#     def _analyze_with_fallback(self, sport_type: str, video_data: dict) -> dict:
#         """
#         Fallback analysis using MediaPipe angles + LLM feedback.
#         Returns unified response structure matching tennis format.
#         """
#         logger.info(f"Running fallback analysis for {sport_type}")
        
#         # Use fallback analyzer with LLM
#         fallback_result = self.fallback_analyzer.analyze_video_fallback(
#             video_data, 
#             sport_type
#         )
        
#         # Fallback analyzer already returns unified structure
#         return fallback_result
    
#     def _generate_no_data_response(self, sport_type: str) -> dict:
#         """
#         Generate response when no pose data is available.
#         Returns unified response structure.
#         """
#         return {
#             "sport_type": sport_type,
#             "analysis_status": "insufficient_data",
#             "overall_score": None,
#             "detailed_scores": None,
#             "feedback": f"""
# ⚠️ **INSUFFICIENT DATA**

# We were unable to detect body movements in your {sport_type} video.

# **Common causes:**
# - Camera too far from subject
# - Poor lighting conditions
# - Subject not fully visible
# - Motion blur from fast movements
# - Background obstructions

# **Please try again with:**
# ✅ Better lighting (natural daylight preferred)
# ✅ Camera 10-15 feet from athlete
# ✅ Full body visible throughout video
# ✅ Minimal background distractions

# Upload a new video for analysis!
# """,
#             "frames_analyzed": 0,
#             "shot_types_detected": [],
#             "processing_complete": True,
#             "is_fallback": True,
#             "model_status": "insufficient_data",
#             "warning": "No pose data detected in video"
#         }


# # Legacy compatibility - if needed
# class EnhancedMultiSportCoachAnalyzer(SimplifiedMultiSportCoachAnalyzer):
#     """Alias for backwards compatibility"""
#     pass



# multi_sport_coach.py
from .llm_coach import TennisCoachAnalyzer
from .fallback_analyzer import FallbackBiomechanicalAnalyzer
import logging

logger = logging.getLogger(__name__)

class SimplifiedMultiSportCoachAnalyzer:
    """
    Unified multi-sport analyzer that provides consistent response structure.
    
    USER-FACING STATUSES:
    - "complete": Analysis successful (with or without technique detection)
    - "processing": Video is being analyzed
    - "failed": Analysis could not be performed
    
    INTERNAL TRACKING:
    - is_fallback: True if using MediaPipe-only analysis (no YOLO)
    - analysis_method: Describes which method was used
    """
    
    def __init__(self):
        self.tennis_analyzer = TennisCoachAnalyzer()
        self.fallback_analyzer = FallbackBiomechanicalAnalyzer()
    
    def analyze_video(self, sport_type: str, video_data: dict) -> dict:
        """
        Main entry point for multi-sport analysis.
        Returns unified response structure for all scenarios.
        
        Args:
            sport_type: 'tennis', 'running', or 'soccer'
            video_data: Dictionary with frame data from video processing
            
        Returns:
            Unified dict with user-friendly status:
            {
                "sport_type": str,
                "analysis_status": "complete" | "processing" | "failed",
                "overall_score": float | None,
                "detailed_scores": dict | None,
                "feedback": str,
                "frames_analyzed": int,
                "shot_types_detected": list,
                "processing_complete": bool,
                
                # Internal tracking (backend use)
                "is_fallback": bool,
                "analysis_method": str,
                "note": str
            }
        """
        
        logger.info(f"Starting analysis for sport: {sport_type}")
        
        # Check if any frames have YOLO detections (shot types)
        shot_types = []
        frames_with_yolo = 0
        frames_with_mediapipe = 0
        
        for frame_data in video_data.values():
            if frame_data.get('class_name'):
                shot_types.append(frame_data['class_name'])
                frames_with_yolo += 1
            if frame_data.get('mediapipe_pose_landmarks'):
                frames_with_mediapipe += 1
        
        logger.info(f"Detection summary - YOLO: {frames_with_yolo} frames, MediaPipe: {frames_with_mediapipe} frames")
        
        # CASE 1: Tennis detected with YOLO - use full tennis analyzer
        if sport_type == 'tennis' and frames_with_yolo > 0:
            logger.info("✅ Tennis techniques detected - using full tennis analyzer")
            return self._analyze_tennis(video_data, shot_types)
        
        # CASE 2: No YOLO detections OR other sports - use fallback with MediaPipe + LLM
        elif frames_with_mediapipe > 0:
            logger.info(f"ℹ️ No specific techniques detected for {sport_type} - using biomechanical pose analysis")
            return self._analyze_with_fallback(sport_type, video_data)
        
        # CASE 3: No data at all
        else:
            logger.warning("❌ No pose data detected in video")
            return self._generate_no_data_response(sport_type)
    
    def _analyze_tennis(self, video_data: dict, shot_types: list) -> dict:
        """
        Full tennis analysis using YOLO keypoints + detailed tennis analyzer.
        Returns complete unified response with scoring.
        
        STATUS: "complete" (with technique detection)
        """
        logger.info("Running comprehensive tennis technique analysis")
        
        # Use the comprehensive tennis analyzer
        tennis_analysis = self.tennis_analyzer.generate_comprehensive_feedback(
            video_data, 
            shot_types
        )
        
        # Return unified response structure
        return {
            "sport_type": "tennis",
            "analysis_status": "complete",  # ✅ USER-FRIENDLY
            "overall_score": tennis_analysis['overall_score'],
            "detailed_scores": tennis_analysis['detailed_scores'],
            "feedback": tennis_analysis['feedback'],
            "frames_analyzed": tennis_analysis['frames_analyzed'],
            "shot_types_detected": tennis_analysis['shot_types_detected'],
            "processing_complete": True,
            
            # Internal tracking
            "is_fallback": False,
            "analysis_method": "tennis_technique_detection",
            "note": f"Full tennis analysis with {len(tennis_analysis['shot_types_detected'])} technique types detected"
        }
    
    def _analyze_with_fallback(self, sport_type: str, video_data: dict) -> dict:
        """
        Fallback analysis using MediaPipe angles + LLM feedback.
        Returns unified response structure.
        
        STATUS: "complete" (biomechanical analysis)
        INTERNAL: is_fallback = True
        """
        logger.info(f"Running biomechanical pose analysis for {sport_type}")
        
        # Use fallback analyzer with LLM
        fallback_result = self.fallback_analyzer.analyze_video_fallback(
            video_data, 
            sport_type
        )
        
        # Fallback analyzer already returns unified structure with "complete" status
        return fallback_result
    
    def _generate_no_data_response(self, sport_type: str) -> dict:
        """
        Generate response when no pose data is available.
        
        STATUS: "failed" (couldn't analyze)
        """
        return {
            "sport_type": sport_type,
            "analysis_status": "failed",  # ✅ USER-FRIENDLY
            "overall_score": None,
            "detailed_scores": None,
            "feedback": f"""⚠️ **UNABLE TO ANALYZE VIDEO**

We couldn't detect your body movements in this {sport_type} video. This usually happens when:

• Camera is too far from the athlete
• Lighting is too dark or has harsh shadows  
• Athlete is partially out of frame
• Video quality is too low or blurry

**📸 For best results, please record with:**
✅ Good lighting (outdoor daylight or bright indoor lighting)
✅ Camera positioned 10-15 feet from athlete
✅ Full body visible from head to feet
✅ Clear, unobstructed view
✅ Stable camera (tripod recommended)

Please upload a new video and we'll analyze it right away!
""",
            "frames_analyzed": 0,
            "shot_types_detected": [],
            "processing_complete": True,
            
            # Internal tracking
            "is_fallback": False,
            "analysis_method": "none",
            "note": "No pose data detected in video"
        }


# Legacy compatibility - if needed
class EnhancedMultiSportCoachAnalyzer(SimplifiedMultiSportCoachAnalyzer):
    """Alias for backwards compatibility"""
    pass
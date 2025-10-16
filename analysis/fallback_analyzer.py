# analysis/fallback_analyzer.py
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class FallbackBiomechanicalAnalyzer:
    """
    Fallback analyzer when YOLO doesn't detect any shots.
    Uses MediaPipe keypoints to calculate body angles and provide basic feedback.
    """
    
    def __init__(self):
        # Define MediaPipe landmark indices
        self.landmarks = {
            'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
    
    def analyze_video_fallback(self, video_data: Dict, sport_type: str) -> Dict[str, Any]:
        """
        Analyze video using only MediaPipe keypoints when YOLO fails.
        """
        logger.info(f"Starting fallback analysis for sport: {sport_type}")
        
        frames_analyzed = 0
        angle_measurements = {
            'left_knee_angles': [],
            'right_knee_angles': [],
            'left_hip_angles': [],
            'right_hip_angles': [],
            'left_elbow_angles': [],
            'right_elbow_angles': [],
            'left_shoulder_angles': [],
            'right_shoulder_angles': [],
            'spine_angles': [],
            'neck_angles': []
        }
        
        for frame_idx, frame_data in video_data.items():
            mediapipe_landmarks = frame_data.get('mediapipe_pose_landmarks', [])
            
            if not mediapipe_landmarks or len(mediapipe_landmarks) < 33:
                continue
            
            frames_analyzed += 1
            
            # Convert landmarks to dict for easy access
            landmarks_dict = {lm['landmark_name']: lm for lm in mediapipe_landmarks}
            
            # Calculate all angles
            angles = self._calculate_all_angles(landmarks_dict)
            
            # Store angles
            for angle_name, angle_value in angles.items():
                if angle_value is not None:
                    angle_measurements[angle_name].append(angle_value)
        
        if frames_analyzed == 0:
            logger.warning("No MediaPipe landmarks found in any frame")
            return self._generate_no_data_response(sport_type)
        
        logger.info(f"Fallback analysis processed {frames_analyzed} frames")
        
        # Calculate statistics
        angle_stats = self._calculate_angle_statistics(angle_measurements)
        
        # Generate feedback based on sport and angles
        feedback = self._generate_fallback_feedback(sport_type, angle_stats, frames_analyzed)
        
        return {
            "sport_type": sport_type,
            "analysis_status": "fallback_analysis",
            "overall_score": None,
            "detailed_scores": None,
            "feedback": feedback,
            "frames_analyzed": frames_analyzed,
            "angle_statistics": angle_stats,
            "processing_complete": True,
            "is_fallback": True,
            "warning": "Shot detection model did not identify specific techniques. Analysis based on general biomechanics."
        }
    
    def _calculate_all_angles(self, landmarks: Dict) -> Dict[str, float]:
        """Calculate all relevant body angles from landmarks."""
        angles = {}
        
        # Knee angles (thigh-shin angle)
        angles['left_knee_angles'] = self._calculate_angle_from_landmarks(
            landmarks.get('LEFT_HIP'),
            landmarks.get('LEFT_KNEE'),
            landmarks.get('LEFT_ANKLE')
        )
        angles['right_knee_angles'] = self._calculate_angle_from_landmarks(
            landmarks.get('RIGHT_HIP'),
            landmarks.get('RIGHT_KNEE'),
            landmarks.get('RIGHT_ANKLE')
        )
        
        # Hip angles (torso-thigh angle)
        angles['left_hip_angles'] = self._calculate_angle_from_landmarks(
            landmarks.get('LEFT_SHOULDER'),
            landmarks.get('LEFT_HIP'),
            landmarks.get('LEFT_KNEE')
        )
        angles['right_hip_angles'] = self._calculate_angle_from_landmarks(
            landmarks.get('RIGHT_SHOULDER'),
            landmarks.get('RIGHT_HIP'),
            landmarks.get('RIGHT_KNEE')
        )
        
        # Elbow angles (upper arm-forearm angle)
        angles['left_elbow_angles'] = self._calculate_angle_from_landmarks(
            landmarks.get('LEFT_SHOULDER'),
            landmarks.get('LEFT_ELBOW'),
            landmarks.get('LEFT_WRIST')
        )
        angles['right_elbow_angles'] = self._calculate_angle_from_landmarks(
            landmarks.get('RIGHT_SHOULDER'),
            landmarks.get('RIGHT_ELBOW'),
            landmarks.get('RIGHT_WRIST')
        )
        
        # Shoulder angles (torso-upper arm angle)
        angles['left_shoulder_angles'] = self._calculate_angle_from_landmarks(
            landmarks.get('LEFT_HIP'),
            landmarks.get('LEFT_SHOULDER'),
            landmarks.get('LEFT_ELBOW')
        )
        angles['right_shoulder_angles'] = self._calculate_angle_from_landmarks(
            landmarks.get('RIGHT_HIP'),
            landmarks.get('RIGHT_SHOULDER'),
            landmarks.get('RIGHT_ELBOW')
        )
        
        # Spine angle (vertical alignment)
        angles['spine_angles'] = self._calculate_spine_angle(
            landmarks.get('NOSE'),
            landmarks.get('LEFT_SHOULDER'),
            landmarks.get('RIGHT_SHOULDER'),
            landmarks.get('LEFT_HIP'),
            landmarks.get('RIGHT_HIP')
        )
        
        # Neck angle
        angles['neck_angles'] = self._calculate_angle_from_landmarks(
            landmarks.get('NOSE'),
            self._midpoint(landmarks.get('LEFT_SHOULDER'), landmarks.get('RIGHT_SHOULDER')),
            self._midpoint(landmarks.get('LEFT_HIP'), landmarks.get('RIGHT_HIP'))
        )
        
        return angles
    
    def _calculate_angle_from_landmarks(self, point_a, point_b, point_c) -> float:
        """Calculate angle between three points (in degrees)."""
        if not all([point_a, point_b, point_c]):
            return None
        
        try:
            a = np.array([point_a['x'], point_a['y']])
            b = np.array([point_b['x'], point_b['y']])
            c = np.array([point_c['x'], point_c['y']])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
        except Exception as e:
            logger.warning(f"Error calculating angle: {e}")
            return None
    
    def _calculate_spine_angle(self, nose, left_shoulder, right_shoulder, left_hip, right_hip) -> float:
        """Calculate spine deviation from vertical."""
        if not all([nose, left_shoulder, right_shoulder, left_hip, right_hip]):
            return None
        
        try:
            shoulder_mid = self._midpoint(left_shoulder, right_shoulder)
            hip_mid = self._midpoint(left_hip, right_hip)
            
            if not shoulder_mid or not hip_mid:
                return None
            
            # Calculate angle from vertical
            dx = shoulder_mid['x'] - hip_mid['x']
            dy = shoulder_mid['y'] - hip_mid['y']
            
            angle_from_vertical = np.degrees(np.arctan2(abs(dx), abs(dy)))
            return angle_from_vertical
        except Exception as e:
            logger.warning(f"Error calculating spine angle: {e}")
            return None
    
    def _midpoint(self, point_a, point_b):
        """Calculate midpoint between two landmarks."""
        if not point_a or not point_b:
            return None
        return {
            'x': (point_a['x'] + point_b['x']) / 2,
            'y': (point_a['y'] + point_b['y']) / 2
        }
    
    def _calculate_angle_statistics(self, angle_measurements: Dict) -> Dict:
        """Calculate mean, std, min, max for each angle."""
        stats = {}
        
        for angle_name, values in angle_measurements.items():
            if values:
                stats[angle_name] = {
                    'mean': round(np.mean(values), 1),
                    'std': round(np.std(values), 1),
                    'min': round(np.min(values), 1),
                    'max': round(np.max(values), 1),
                    'range': round(np.max(values) - np.min(values), 1)
                }
            else:
                stats[angle_name] = None
        
        return stats
    
    def _generate_fallback_feedback(self, sport_type: str, angle_stats: Dict, frames_analyzed: int) -> str:
        """Generate feedback based on biomechanical analysis."""
        
        sport_emoji = {
            'tennis': '🎾',
            'running': '🏃',
            'soccer': '⚽'
        }
        
        feedback = f"""
{sport_emoji.get(sport_type, '🏋️')} **BIOMECHANICAL ANALYSIS - {sport_type.upper()}**

⚠️ **IMPORTANT NOTICE**
Our sport-specific shot detection model is currently under training and did not identify specific techniques in your video. However, we've successfully analyzed your body movements and biomechanics using advanced pose estimation.

**✅ Analysis Completed:**
- ✅ Full body pose tracking ({frames_analyzed} frames analyzed)
- ✅ Joint angle measurements calculated
- ✅ Movement pattern assessment
- ✅ Biomechanical posture evaluation

**📊 BIOMECHANICAL MEASUREMENTS:**

"""
        
        # Knee analysis
        left_knee = angle_stats.get('left_knee_angles')
        right_knee = angle_stats.get('right_knee_angles')
        if left_knee or right_knee:
            feedback += "**Knee Angles:**\n"
            if left_knee:
                feedback += f"- Left Knee: {left_knee['mean']}° (range: {left_knee['min']}° - {left_knee['max']}°)\n"
                if left_knee['mean'] < 140:
                    feedback += "  💡 Tip: Your left knee shows significant bending. Ensure proper flexion for shock absorption.\n"
            if right_knee:
                feedback += f"- Right Knee: {right_knee['mean']}° (range: {right_knee['min']}° - {right_knee['max']}°)\n"
                if right_knee['mean'] < 140:
                    feedback += "  💡 Tip: Your right knee shows significant bending. This is good for athletic stance.\n"
            feedback += "\n"
        
        # Hip analysis
        left_hip = angle_stats.get('left_hip_angles')
        right_hip = angle_stats.get('right_hip_angles')
        if left_hip or right_hip:
            feedback += "**Hip Angles:**\n"
            if left_hip:
                feedback += f"- Left Hip: {left_hip['mean']}° (range: {left_hip['min']}° - {left_hip['max']}°)\n"
            if right_hip:
                feedback += f"- Right Hip: {right_hip['mean']}° (range: {right_hip['min']}° - {right_hip['max']}°)\n"
            feedback += "\n"
        
        # Elbow analysis (important for tennis/throwing sports)
        left_elbow = angle_stats.get('left_elbow_angles')
        right_elbow = angle_stats.get('right_elbow_angles')
        if left_elbow or right_elbow:
            feedback += "**Arm Angles:**\n"
            if left_elbow:
                feedback += f"- Left Elbow: {left_elbow['mean']}° (variability: {left_elbow['std']}°)\n"
            if right_elbow:
                feedback += f"- Right Elbow: {right_elbow['mean']}° (variability: {right_elbow['std']}°)\n"
            feedback += "\n"
        
        # Spine/posture analysis
        spine = angle_stats.get('spine_angles')
        if spine:
            feedback += f"**Posture Analysis:**\n"
            feedback += f"- Spine Deviation: {spine['mean']}° from vertical\n"
            if spine['mean'] > 15:
                feedback += "  ⚠️ Warning: Significant forward/backward lean detected. Check your posture.\n"
            elif spine['mean'] < 5:
                feedback += "  ✅ Excellent: Maintaining upright posture throughout movement.\n"
            feedback += "\n"
        
        # Sport-specific recommendations
        feedback += f"""
**🎯 SPORT-SPECIFIC OBSERVATIONS ({sport_type.upper()}):**

"""
        
        if sport_type == 'tennis':
            feedback += """
- Your body movements were captured, but specific shot types (forehand, backhand, serve) were not identified
- Knee flexion and hip rotation angles suggest athletic readiness position
- Consider recording closer to the camera with better lighting for shot detection
"""
        elif sport_type == 'running':
            feedback += """
- Your running gait pattern was captured at the joint level
- Knee flexion angles indicate your stride mechanics
- For detailed gait analysis, ensure full body visibility in frame
"""
        elif sport_type == 'soccer':
            feedback += """
- Your movement patterns were captured, but specific ball interactions were not detected
- Lower body joint angles show dynamic movement patterns
- For technique analysis, ensure clear view of ball contact moments
"""
        
        feedback += f"""

**📈 WHAT'S NEXT:**
Our AI models are continuously improving! Once our {sport_type} technique detection model completes training, you'll receive:
- ✅ Specific technique identification
- ✅ Professional-level scoring (0-10 scale)
- ✅ Detailed form corrections
- ✅ Comparison with professional athletes

**💡 IMPROVEMENT TIPS FOR BETTER ANALYSIS:**
1. Record with good lighting (natural daylight is best)
2. Ensure full body is visible in frame (head to feet)
3. Position camera 10-15 feet away for optimal angle
4. Minimize background clutter for clearer detection
5. Record during active movements (not warm-up/rest periods)

---
*This analysis was generated using advanced pose estimation. For most accurate sport-specific feedback, our AI models need to detect specific techniques. Keep training and upload more videos!* 🚀
"""
        
        return feedback.strip()
    
    def _generate_no_data_response(self, sport_type: str) -> Dict:
        """Generate response when no pose data is available."""
        return {
            "sport_type": sport_type,
            "analysis_status": "insufficient_data",
            "overall_score": None,
            "detailed_scores": None,
            "feedback": f"""
⚠️ **INSUFFICIENT POSE DATA**

We were unable to detect body movements in your {sport_type} video. This can happen due to:

- Camera too far from the subject
- Poor lighting conditions
- Subject not fully visible in frame
- Very fast movements causing motion blur
- Obstructions blocking body visibility

**Please try again with:**
✅ Better lighting (natural daylight preferred)
✅ Camera 10-15 feet from athlete
✅ Full body visible throughout video
✅ Minimal background distractions

Upload a new video for analysis!
""",
            "frames_analyzed": 0,
            "processing_complete": True,
            "is_fallback": True,
            "warning": "No pose data detected in video"
        }
        
        
        
# **🔬 TECHNICAL DETAILS:**
# - Analysis Method: MediaPipe Pose Estimation (33-point skeleton tracking)
# - Confidence: Moderate (fallback analysis without sport-specific model)
# - Data Quality: {"Good" if frames_analyzed > 50 else "Limited"} ({frames_analyzed} frames)

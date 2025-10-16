# llm_coach.py
import os
import json
from typing import Dict, List, Any
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv

load_dotenv()

class TennisCoachAnalyzer:
    def __init__(self):
        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant",  # or "mixtral-8x7b-32768"
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
            max_tokens=500
        )
        
    def analyze_keypoints(self, frame_data: Dict) -> Dict[str, Any]:
        """Analyze keypoints from a single frame"""
        if not frame_data.get('keypoints'):
            return {"confidence": 0, "issues": ["No pose detected"]}
        
        keypoints = frame_data['keypoints']
        analysis = {
            "stance_quality": self._analyze_stance(keypoints),
            "arm_position": self._analyze_arm_position(keypoints),
            "body_alignment": self._analyze_body_alignment(keypoints),
            "balance": self._analyze_balance(keypoints)
        }
        return analysis
    
    def _analyze_stance(self, keypoints: List[Dict]) -> Dict:
        """Analyze player's stance based on keypoints"""
        # Find relevant keypoints
        left_ankle = next((kp for kp in keypoints if kp['class_name'] == 'left_ankle'), None)
        right_ankle = next((kp for kp in keypoints if kp['class_name'] == 'right_ankle'), None)
        left_knee = next((kp for kp in keypoints if kp['class_name'] == 'left_knee'), None)
        right_knee = next((kp for kp in keypoints if kp['class_name'] == 'right_knee'), None)
        
        if not all([left_ankle, right_ankle, left_knee, right_knee]):
            return {"score": 0, "issues": ["Incomplete leg detection"]}
        
        # Calculate stance width
        stance_width = abs(left_ankle['x'] - right_ankle['x'])
        
        # Analyze knee alignment
        knee_alignment = abs((left_knee['x'] - left_ankle['x']) - (right_knee['x'] - right_ankle['x']))
        
        score = 10
        issues = []
        
        if stance_width < 0.1:  # Too narrow
            score -= 3
            issues.append("Stance too narrow - widen feet for better balance")
        elif stance_width > 0.4:  # Too wide
            score -= 2
            issues.append("Stance too wide - may reduce mobility")
            
        if knee_alignment > 0.05:
            score -= 2
            issues.append("Knees not properly aligned over feet")
            
        return {"score": max(0, score), "issues": issues}
    
    def _analyze_arm_position(self, keypoints: List[Dict]) -> Dict:
        """Analyze arm positioning for tennis strokes"""
        shoulder_left = next((kp for kp in keypoints if kp['class_name'] == 'left_shoulder'), None)
        shoulder_right = next((kp for kp in keypoints if kp['class_name'] == 'right_shoulder'), None)
        elbow_left = next((kp for kp in keypoints if kp['class_name'] == 'left_elbow'), None)
        elbow_right = next((kp for kp in keypoints if kp['class_name'] == 'right_elbow'), None)
        wrist_left = next((kp for kp in keypoints if kp['class_name'] == 'left_wrist'), None)
        wrist_right = next ((kp for kp in keypoints if kp['class_name'] == 'right_wrist'), None)
        
        if not all([shoulder_left, shoulder_right, elbow_left, elbow_right]):
            return {"score": 0, "issues": ["Incomplete arm detection"]}
        
        score = 10
        issues = []
        
        # Check elbow extension
        if elbow_right and wrist_right and shoulder_right:
            # Calculate arm extension (simplified)
            arm_extension = abs(wrist_right['y'] - shoulder_right['y'])
            if arm_extension < 0.1:
                score -= 2
                issues.append("Extend your hitting arm more during stroke")
        
        # Check shoulder alignment
        shoulder_level = abs(shoulder_left['y'] - shoulder_right['y'])
        if shoulder_level > 0.05:
            score -= 1
            issues.append("Keep shoulders more level during stroke")
            
        return {"score": max(0, score), "issues": issues}
    
    def _analyze_body_alignment(self, keypoints: List[Dict]) -> Dict:
        """Analyze overall body alignment"""
        hip_left = next((kp for kp in keypoints if kp['class_name'] == 'left_hip'), None)
        hip_right = next((kp for kp in keypoints if kp['class_name'] == 'right_hip'), None)
        shoulder_left = next((kp for kp in keypoints if kp['class_name'] == 'left_shoulder'), None)
        shoulder_right = next((kp for kp in keypoints if kp['class_name'] == 'right_shoulder'), None)
        
        if not all([hip_left, hip_right, shoulder_left, shoulder_right]):
            return {"score": 0, "issues": ["Incomplete body detection"]}
        
        score = 10
        issues = []
        
        # Check hip alignment
        hip_alignment = abs(hip_left['y'] - hip_right['y'])
        if hip_alignment > 0.05:
            score -= 2
            issues.append("Maintain level hips throughout stroke")
        
        # Check torso rotation (simplified)
        hip_center = (hip_left['x'] + hip_right['x']) / 2
        shoulder_center = (shoulder_left['x'] + shoulder_right['x']) / 2
        rotation = abs(hip_center - shoulder_center)
        
        if rotation > 0.3:
            score -= 1
            issues.append("Excessive body rotation - maintain more stable core")
            
        return {"score": max(0, score), "issues": issues}
    
    def _analyze_balance(self, keypoints: List[Dict]) -> Dict:
        """Analyze player's balance"""
        left_ankle = next((kp for kp in keypoints if kp['class_name'] == 'left_ankle'), None)
        right_ankle = next((kp for kp in keypoints if kp['class_name'] == 'right_ankle'), None)
        nose = next((kp for kp in keypoints if kp['class_name'] == 'nose'), None)
        
        if not all([left_ankle, right_ankle, nose]):
            return {"score": 0, "issues": ["Cannot assess balance - incomplete detection"]}
        
        score = 10
        issues = []
        
        # Check if head is centered over base of support
        foot_center = (left_ankle['x'] + right_ankle['x']) / 2
        head_offset = abs(nose['x'] - foot_center)
        
        if head_offset > 0.2:
            score -= 3
            issues.append("Weight distribution off-center - focus on balance")
        elif head_offset > 0.1:
            score -= 1
            issues.append("Minor balance adjustment needed")
            
        return {"score": max(0, score), "issues": issues}

    def generate_comprehensive_feedback(self, video_data: Dict, shot_types: List[str]) -> Dict[str, Any]:
        """Generate comprehensive coaching feedback using LLM"""
        
        # Analyze keypoints across all frames
        frame_analyses = []
        valid_frames = 0
        print("getting feedback")
        for frame_idx, frame_data in video_data.items():
            print("keypoint:")
            if frame_data.get('keypoints'):
                analysis = self.analyze_keypoints(frame_data)
                frame_analyses.append({
                    "frame": frame_idx,
                    "shot_type": frame_data.get('class_name', 'unknown'),
                    "analysis": analysis
                })
                valid_frames += 1
        print(str(frame_analyses))
        
        if not frame_analyses:
            return {
                "overall_score": 0,
                "feedback": "Unable to analyze video - no pose data detected",
                "recommendations": []
            }
        
        # Calculate overall scores
        avg_scores = self._calculate_average_scores(frame_analyses)
        overall_score = round(sum(avg_scores.values()) / len(avg_scores), 1)
        print(str(overall_score))
        # Prepare detailed analysis for LLM
        analysis_summary = self._prepare_analysis_summary(frame_analyses, shot_types, avg_scores)
        print("prepared summary")
        # Generate LLM feedback
        llm_feedback = self._get_llm_feedback(analysis_summary, overall_score)
        print("finfish feedback")
        return {
            "overall_score": overall_score,
            "detailed_scores": avg_scores,
            "feedback": llm_feedback,
            "frames_analyzed": valid_frames,
            "shot_types_detected": list(set(shot_types))
        }
    
    def _calculate_average_scores(self, frame_analyses: List[Dict]) -> Dict[str, float]:
        """Calculate average scores across all frames"""
        scores = {
            "stance_quality": [],
            "arm_position": [],
            "body_alignment": [],
            "balance": []
        }
        print(str(scores))
        for frame_analysis in frame_analyses:
            analysis = frame_analysis["analysis"]
            for metric in scores.keys():
                if metric in analysis:
                    scores[metric].append(analysis[metric]["score"])
        print(str(scores))
        print(str({metric: round(sum(values) / len(values), 1) if values else 0 
            for metric, values in scores.items()}))
        return {
            metric: round(sum(values) / len(values), 1) if values else 0 
            for metric, values in scores.items()
        }
    
    def _prepare_analysis_summary(self, frame_analyses: List[Dict], shot_types: List[str], avg_scores: Dict) -> str:
        """Prepare analysis summary for LLM"""
        # Collect all issues
        all_issues = []
        shot_type_distribution = {}
        
        for frame_analysis in frame_analyses:
            shot_type = frame_analysis["shot_type"]
            if shot_type != "unknown":
                shot_type_distribution[shot_type] = shot_type_distribution.get(shot_type, 0) + 1
            
            analysis = frame_analysis["analysis"]
            for metric_analysis in analysis.values():
                if isinstance(metric_analysis, dict) and "issues" in metric_analysis:
                    all_issues.extend(metric_analysis["issues"])
        
        # Count common issues
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        summary = f"""
        TENNIS TECHNIQUE ANALYSIS SUMMARY:
        
        Frames Analyzed: {len(frame_analyses)}
        Shot Types Detected: {shot_type_distribution}
        
        Average Scores (out of 10):
        - Stance Quality: {avg_scores.get('stance_quality', 0)}
        - Arm Position: {avg_scores.get('arm_position', 0)}
        - Body Alignment: {avg_scores.get('body_alignment', 0)}
        - Balance: {avg_scores.get('balance', 0)}
        
        Most Common Issues:
        {chr(10).join([f"- {issue} (occurred {count} times)" for issue, count in common_issues])}
        """
        
        return summary
    
    def _get_llm_feedback(self, analysis_summary: str, overall_score: float) -> str:
        """Get coaching feedback from LLM"""
        
        system_template = """You are a professional tennis coach and biomechanics expert with 20+ years of experience. 
        You specialize in analyzing tennis technique using pose estimation data and providing actionable coaching advice.
        
        Your coaching style is:
        - Precise and technical when needed, but accessible to players of all levels
        - Focused on practical, actionable improvements
        - Encouraging while being honest about areas needing work
        - Prioritizes the most impactful changes first
        
        Based on the technical analysis provided, give coaching feedback that includes:
        1. Overall technique assessment
        2. Top 3 priority improvements (most impactful first)
        3. One specific drill or exercise recommendation
        4. Encouragement and next steps
        
        Keep your response concise but comprehensive (max 400 words)."""
        
        human_template = """Here's the technical analysis of the tennis player's performance:
        
        Overall Score: {overall_score}/10
        
        {analysis_summary}
        
        Please provide your professional coaching feedback and recommendations."""
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
        
        try:
            messages = chat_prompt.format_messages(
                overall_score=overall_score,
                analysis_summary=analysis_summary
            )
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Analysis complete with score {overall_score}/10. Technical analysis available, but detailed feedback generation encountered an issue: {str(e)}"
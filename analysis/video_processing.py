# # import cv2
# # import mediapipe as mp
# # import numpy as np

# # def process_video(file_path):
# #     cap = cv2.VideoCapture(file_path)
# #     mp_pose = mp.solutions.pose
# #     pose = mp_pose.Pose()
# #     angles = []

# #     while cap.isOpened():
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# #         if results.pose_landmarks:
# #             landmarks = results.pose_landmarks.landmark
# #             # Calculate the angle of the toe joint
# #             angle = calculate_angle(landmarks)
# #             angles.append(angle)

# #     cap.release()
# #     return angles

# # def calculate_angle(landmarks):
# #     # Assuming you have functions to get the coordinates of toe joints
# #     a = np.array([landmarks[foot_index1].x, landmarks[foot_index1].y])
# #     b = np.array([landmarks[foot_index2].x, landmarks[foot_index2].y])
# #     c = np.array([landmarks[foot_index3].x, landmarks[foot_index3].y])
    
# #     ba = a - b
# #     bc = c - b
    
# #     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
# #     angle = np.arccos(cosine_angle)
    
# #     return np.degrees(angle)

# # def compare_angles(angles1, angles2):
# #     # Compare angles of two videos
# #     diff = np.abs(np.array(angles1) - np.array(angles2))
# #     return diff
    
# # import cv2
# # import mediapipe as mp
# # import numpy as np

# # def process_videos(uploaded_video):
# #     mp_pose = mp.solutions.pose
# #     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
# #         cap = cv2.VideoCapture(uploaded_video)
# #         processed_video_path = 'path/to/processed_video.mp4'  # Replace with actual path
# #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# #         out = cv2.VideoWriter(processed_video_path, fourcc, 20.0, (640, 480))

# #         while cap.isOpened():
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break

# #             results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# #             if results.pose_landmarks:
# #                 # Your processing logic here (e.g., draw landmarks)
# #                 # For example, just draw the landmarks
# #                 for landmark in results.pose_landmarks.landmark:
# #                     cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 5, (0, 255, 0), -1)

# #             out.write(frame)
        
# #         cap.release()
# #         out.release()

# #         return processed_video_path
    
    
# # import cv2
# # import mediapipe as mp
# # import numpy as np

# # def extract_angles_from_processed_video(processed_video_path):
# #     mp_pose = mp.solutions.pose
# #     angles = []

# #     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
# #         cap = cv2.VideoCapture(processed_video_path)

# #         while cap.isOpened():
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break

# #             results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# #             if results.pose_landmarks:
# #                 landmarks = results.pose_landmarks.landmark
# #                 # Calculate the angle of the toe joint
# #                 angle = calculate_angle(landmarks)
# #                 angles.append(angle)
        
# #         cap.release()

# #     return angles

# # def calculate_angle(landmarks):
# #     # Replace with the correct indices for foot landmarks
# #     foot_index1 = 0  # For demonstration, replace with actual index
# #     foot_index2 = 1
# #     foot_index3 = 2

# #     a = np.array([landmarks[foot_index1].x, landmarks[foot_index1].y])
# #     b = np.array([landmarks[foot_index2].x, landmarks[foot_index2].y])
# #     c = np.array([landmarks[foot_index3].x, landmarks[foot_index3].y])
    
# #     ba = a - b
# #     bc = c - b
    
# #     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
# #     angle = np.arccos(cosine_angle)
    
# #     return np.degrees(angle)

# import cv2
# import mediapipe as mp
# import numpy as np
# from .models import AngleData

# mp_pose = mp.solutions.pose

# def process_videos(uploaded_video_path,id):
#     """
#     Processes uploaded videos using MediaPipe pose estimation, calculates neck-shoulder and knee-hip angles,
#     saves the processed video, and stores angle data in the database.

#     Args:
#         uploaded_video: An instance of the UploadedVideo model.

#     Returns:
#         The path to the processed video file.
#     """


#     processed_video_path = f"processed_videos/{id}.mp4"

#     mp_drawing = mp.solutions.drawing_utils

#     cap = cv2.VideoCapture(uploaded_video_path)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(processed_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
#                           (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break 


#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(frame_rgb)

#             if results.pose_landmarks:
#                 landmarks = results.pose_landmarks.landmark 


#                 # Calculate angles
#                 neck_shoulder_angle = calculate_neck_shoulder_angle(landmarks)
#                 knee_hip_angle = calculate_knee_hip_angle(landmarks)

#                 # Save angle data to the database
#                 angle_data = AngleData(
#                     processed_video=id,
#                     frame_number=frame_count,
#                     neck_shoulder_angle=neck_shoulder_angle,
#                     knee_hip_angle=knee_hip_angle
#                 )
#                 angle_data.save()

#                 # Draw landmarks and angles (optional)
#                 mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             out.write(frame)

#             cv2.imshow('Processed Video', frame)
#             if cv2.waitKey(1) == ord('q'):
#                 break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     return processed_video_path

# def calculate_neck_shoulder_angle(landmarks):
#     """
#     Calculates the angle between the neck (landmark 1), left shoulder (landmark 11), and right shoulder (landmark 12).

#     Args:
#         landmarks: A list of MediaPipe pose landmarks.

#     Returns:
#         The angle in degrees between the neck and shoulders.
#     """

#     neck = np.array([landmarks[mp_pose.PoseLandmark.NECK].x, landmarks[mp_pose.PoseLandmark.NECK].y])
#     left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
#     right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])

#     # Calculate vectors connecting neck to left and right shoulders
#     ba = neck - left_shoulder
#     bc = neck - right_shoulder

#     # Calculate cosine of the angle using dot product and norm
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     angle = np.arccos(cosine_angle)
    
#     return np.degrees(angle)

# def calculate_knee_hip_angle(landmarks):
#     """
#     Calculates the angle between the left knee, left hip, and right hip.

#     Args:
#         landmarks: A list of MediaPipe pose landmarks.

#     Returns:
#         The angle in degrees between the left knee and hips.
#     """

#     left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y])
#     left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])
#     right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y])

#     # Calculate vectors connecting left knee to left and right hips
#     ba = left_knee - left_hip
#     bc = left_knee - right_hip

#     # Calculate cosine of the angle using dot product and norm
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     angle = np.arccos(cosine_angle)

#     return np.degrees(angle)


# import cv2
# import math
# import numpy as np
# import mediapipe as mp
# from .models import Video

# # Define some colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# GREEN = (0, 200, 0)
# RED = (0, 0, 255)
# BLUE = (245, 117, 25)

# forcc = cv2.VideoWriter.fourcc(*'mp4v')


# # Video writer.

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
# stage = ""
# color = RED
# pro_elbow_angle_range = (60, 80)  # Example range for pro players
# pro_shoulder_angle_range = (40, 80)  # Example range for pro players
# pro_knee_angle_range = (130, 170)  # Example range for pro players
# pro_hip_angle_range = (100, 140)  # Example range for pro players


# def calculate_angle(a, b, c):
#     """
#     Function to calculate the angle between three points.
#     """
#     # Convert the points from list or tuple to numpy array
#     a = np.array(a)  # First joint coordinate (for example, shoulder)
#     b = np.array(b)  # Mid joint coordinate (for example, elbow)
#     c = np.array(c)  # End joint coordinate (for example, wrist)

#     # Calculate the angle using the atan2 function from the math library
#     # The atan2 function returns the angle in radians from the x-axis to the point (y, x),
#     # so it's used here to get the angles of the lines bc and ba from the x-axis.
#     radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])

#     # Convert the angle from radians to degrees. As we want to get the positive angle,
#     # we use the abs function to ensure that the angle is positive.
#     angle = np.abs(radians * 180.0 / math.pi)

#     # If the calculated angle is more than 180 degrees,
#     # then we calculate its supplementary angle by subtracting it from 360 degrees.
#     # This is done because the joints can move in any direction and we are interested in the smaller angle formed.
#     if angle > 180.0:
#         angle = 360 - angle

#     return angle



# def process_video(video_id):
#     video = Video.objects.get(pk=video_id)
#     print(video.video_file.path)
#     cap = cv2.VideoCapture(video.video_file.path)
    
#     if (cap.isOpened() == False):  
#         print("Error reading video file")
        
#     frame_width = int(cap.get(3)) 
#     frame_height = int(cap.get(4))

#     size = (frame_width, frame_height)
    
    

#     # cv2.CAP_PROP_FPS
#     video_output = cv2.VideoWriter('output1.mp4', cv2.VideoWriter_fourcc(*'MMP4'), 60, size)

#     mp_pose = mp.solutions.pose
#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while cap.isOpened():
#             success, image = cap.read()
#             if not success:
#                 break

#             # Convert the BGR image to RGB for MediaPipe
#             # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             # # Process the image with MediaPipe
#             # results = pose.process(image_rgb)

#             # if results.pose_landmarks:
#             #     # Extract landmarks
#             #     landmarks = results.pose_landmarks.landmark

#             #     # Calculate angles (adjust indices as needed)
#             #     shoulder_eye_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
#             #                                           landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER])
#             #     hip_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP],
#             #                                       landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
#             #     angle_difference = abs(shoulder_eye_angle - hip_knee_angle)

#             #     # Update the Video model
#             #     video.shoulder_eye_angle = shoulder_eye_angle
#             #     video.hip_knee_angle = hip_knee_angle
#             #     video.angle_difference = angle_difference
#             #     video.save()

#             #     # Optionally, draw landmarks and angles on the image
#             #     # ...

#             # # Process the frame and save it to the processed video
#             # # ...
            
            
#             # Convert the BGR image to RGB.
#             # Mediapipe Pose model uses RGB images for processing
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             # Make the image unwritable as mediapipe pose model requires the input images to be unwritable
#             image.flags.writeable = False

#             # Process the image to detect pose landmarks
#             results = pose.process(image)

#             # Make the image writable again so we can draw on it later
#             image.flags.writeable = True

#             # Convert the image back to BGR for further opencv processing and visualizations
#             # Because OpenCV uses BGR as its default color format
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
#             try:
#                 landmarks = results.pose_landmarks.landmark
#                 left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#                 right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#                 left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#                 right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
#                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
#                 left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
#                 right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
#                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
#                 left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#                 right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
#                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
#                 left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
#                 right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
#                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
#                 left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
#                 right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
#                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]            
#                 nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
#                             landmarks[mp_pose.PoseLandmark.NOSE.value].y]
#             except:
                    
#                     print("too far away")
#                     pass
            
#             # neck_angle = calculate_angle(nose, left_shoulder, left_shoulder)
#             left_elbow_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)
#             right_elbow_angle = calculate_angle(right_wrist, right_elbow, right_shoulder)
#             left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
#             right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
#             left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
#             right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
#             left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
#             right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            
#             # Check skill level based on multiple angles
#             if pro_elbow_angle_range[0] <= right_elbow_angle <= pro_elbow_angle_range[1] and \
#             pro_shoulder_angle_range[0] <= right_shoulder_angle <= pro_shoulder_angle_range[1] :
#             # pro_knee_angle_range[0] <= right_knee_angle <= pro_knee_angle_range[1] and \
#             # pro_hip_angle_range[0] <= right_hip_angle <= pro_hip_angle_range[1]:
                    
#                     stage = "Pro Player"
#                     color = GREEN
#             else:
#                     stage = "Beginner Player"
#                     color = RED
            
            
#             cv2.rectangle(image, (int(frame_width / 2) - 150, 0), (int(frame_width / 2) + 250, 73), BLUE, -1)
#             cv2.putText(image, 'AI Athletes Manager', (int(frame_width / 2) - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                 WHITE, 2, cv2.LINE_AA)
            
            
#             # Reps
#             cv2.rectangle(image, (0, 0), (260, 200), BLUE, -1)
#             cv2.putText(image, f"right_elbow_angle: {int(right_elbow_angle)}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
#             cv2.putText(image, f"left_elbow_angle: {int(left_elbow_angle)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
#             cv2.putText(image, f"right_shoulder_angle: {int(right_shoulder_angle)}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
#             cv2.putText(image, f"left_shoulder_angle: {int(left_shoulder_angle)}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
#             cv2.putText(image, f"left_knee_angle: {int(left_knee_angle)}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
#             cv2.putText(image, f"right_knee_angle: {int(right_knee_angle)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
            
            
#             cv2.rectangle(image, (0, 400), (180, 250), color, -1)
#             cv2.putText(image, 'stage:', (15, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
#             cv2.putText(image, stage, (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 1, cv2.LINE_AA)
            
            
#             # Render detections
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                         mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=3),
#                                         mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=3))
            
            
#             # Display the frame
#             cv2.imshow('AI Athletes Manager', image)
            
#             video_output.write(image)

#             # Quit when 'q' key is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break


#     # Release the VideoCapture object
#     cap.release()
#     video_output.release()
#     cv2.destroyAllWindows()
#     print(video_output.get(1))
#     print(video_output.get(2))
#     print(video_output.get(3))
#     print(video_output.get(4))
#     return video_output
    
# def calculate_angle(a, b, c):
#     """
#     Calculates the angle between two vectors defined by points a, b, and c.

#     Args:
#         a: The first point (tuple or numpy array).
#         b: The middle point (tuple or numpy array).
#         c: The second point (tuple or numpy array).

#     Returns:
#         The angle in degrees.
#     """

#     ba = np.array(b) - np.array(a)
#     bc = np.array(c) - np.array(b)

#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     angle_rad = np.arccos(cosine_angle)
#     angle_deg = np.degrees(angle_rad)

#     return angle_deg




# # video_processing.py
# from inference import get_model
# import supervision as sv
# from .models import Video
# import cv2
# import json
# import os
# from dotenv import load_dotenv

# load_dotenv()

# def process_video(video_id, video_name):
#     # Open the video file
#     video = Video.objects.get(pk=video_id)
#     cap = cv2.VideoCapture(video.video_file.path)
    
#     if not cap.isOpened():
#         print("Error reading video file")
#         return
    
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     size = (frame_width, frame_height)
    
#     # Load a pre-trained YOLOv8 model
#     model = get_model(model_id=os.getenv("MODEL_ID"))
    
#     # Initialize video writer for saving annotated video
#     video_output_path = f'{video_name}_processed.mp4'
#     video_output = cv2.VideoWriter(
#         video_output_path, 
#         cv2.VideoWriter_fourcc(*'mp4v'),  # Use 'mp4v' codec for compatibility
#         cap.get(cv2.CAP_PROP_FPS), 
#         size
#     )
    
#     # Initialize a dictionary to store all extracted information
#     video_data = {}
#     frame_idx = 0
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             break
        
#         # Run inference on the current frame
#         results = model.infer(image)[0]
        
#         # Extract class name and keypoints for the current frame
#         frame_data = {
#             "class_name": None,
#             "keypoints": []
#         }
#         if len(results.predictions) > 0:
#             # Extract class name
#             class_name = results.predictions[0].class_name
#             frame_data["class_name"] = class_name
#             print(class_name)
#             # Extract keypoints
#             for keypoint in results.predictions[0].keypoints:
#                 x = keypoint.x
#                 y = keypoint.y
#                 confidence = keypoint.confidence
#                 frame_data["keypoints"].append({
#                     "x": x,
#                     "y": y,
#                     "confidence": confidence,
#                     "keypoint_id": keypoint.class_id,
#                     "class_name": keypoint.class_name
#                 })
#         # Add the frame data to the video data dictionary
#         video_data[frame_idx] = frame_data
#         # Load the results into the supervision Detections API
#         detections = sv.Detections.from_inference(results)
        
#         # Create supervision annotators
#         bounding_box_annotator = sv.BoxAnnotator()
#         label_annotator = sv.LabelAnnotator()
        
#         # Annotate the image with our inference results
#         annotated_image = bounding_box_annotator.annotate(
#             scene=image, detections=detections)
#         annotated_image = label_annotator.annotate(
#             scene=annotated_image, detections=detections)
#         # Write the annotated frame to the output video
#         video_output.write(annotated_image)
        
#         # Quit when 'q' key is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         # Increment the frame index
#         frame_idx += 1
    
#     # Release the VideoCapture and VideoWriter objects
#     cap.release()
#     video_output.release()
#     cv2.destroyAllWindows()
    
#     # Save the extracted data to a JSON file
#     video_data_path = f"{video_name}_data.json"
#     with open(video_data_path, "w") as f:
#         json.dump(video_data, f, indent=4)
#     print("Video processing completed. Data saved to 'video_data.json'.")
    
#     return video_data, video_output_path, video_data_path


# video_processing.py
from inference import get_model
import supervision as sv
from .models import Video
import cv2
import json
import os
from dotenv import load_dotenv
import logging
import mediapipe as mp
import numpy as np

load_dotenv()

logger = logging.getLogger(__name__)

def verify_media_volume():
    """
    Verify that media volume is properly mounted and accessible.
    Call this when the worker starts.
    """
    media_root = os.environ.get('MEDIA_ROOT', '/app/data')
    
    logger.info("=" * 60)
    logger.info("🔍 MEDIA VOLUME VERIFICATION")
    logger.info("=" * 60)
    
    # Check if directory exists
    if not os.path.exists(media_root):
        logger.error(f"❌ Media root does not exist: {media_root}")
        logger.error("   This means the volume is not properly mounted!")
        logger.error("   Fix: Mount bucket-volume to /app/data in Railway")
        return False
    
    logger.info(f"✅ Media root exists: {media_root}")
    
    # Check if directory is writable
    test_file = os.path.join(media_root, '.volume_test')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        logger.info(f"✅ Media root is writable")
    except Exception as e:
        logger.error(f"❌ Media root is not writable: {e}")
        return False
    
    # List videos directory
    videos_dir = os.path.join(media_root, 'videos')
    if os.path.exists(videos_dir):
        files = os.listdir(videos_dir)
        logger.info(f"✅ Videos directory exists with {len(files)} files")
        if files:
            logger.info(f"   Sample files: {files[:5]}")
    else:
        logger.warning(f"⚠️  Videos directory does not exist yet: {videos_dir}")
        logger.info("   This is normal if no videos have been uploaded")
    
    logger.info("=" * 60)
    return True


def debug_file_access(video_id, video_path):
    """
    Debug helper to diagnose file access issues
    """
    logger.error("=" * 60)
    logger.error("🔍 FILE ACCESS DEBUG INFO")
    logger.error("=" * 60)
    logger.error(f"Video ID: {video_id}")
    logger.error(f"Expected path: {video_path}")
    logger.error(f"File exists: {os.path.exists(video_path)}")
    
    # Check directory
    directory = os.path.dirname(video_path)
    logger.error(f"Directory: {directory}")
    logger.error(f"Directory exists: {os.path.exists(directory)}")
    
    if os.path.exists(directory):
        files = os.listdir(directory)
        logger.error(f"Files in directory ({len(files)}): {files[:10]}")
    
    # Check media root
    from django.conf import settings
    media_root = settings.MEDIA_ROOT
    logger.error(f"MEDIA_ROOT setting: {media_root}")
    logger.error(f"MEDIA_ROOT exists: {os.path.exists(media_root)}")
    
    # Check environment
    logger.error(f"Current working directory: {os.getcwd()}")
    logger.error(f"User: {os.getenv('USER', 'unknown')}")
    
    # List mounted volumes
    logger.error("Checking /app for mounted volumes...")
    if os.path.exists('/app'):
        app_contents = os.listdir('/app')
        logger.error(f"Contents of /app: {app_contents}")
        if 'media' in app_contents:
            media_contents = os.listdir('/app/data')
            logger.error(f"Contents of /app/data: {media_contents}")
    
    logger.error("=" * 60)


def process_video(video_id, video_name):
    # Open the video file
    # video = Video.objects.get(pk=video_id)
    # cap = cv2.VideoCapture(video.video_file.path)
    
    # if not cap.isOpened():
    #     logger.error(f"Error reading video file for video ID: {video_id}")
    #     print("Error reading video file")
    #     return
    
    logger.info(f"🎬 Starting video processing for ID: {video_id}")
    
    if not verify_media_volume():
        raise RuntimeError(
            "Media volume not properly mounted. "
            "Please mount bucket-volume to /app/data in both Server and celery-worker services."
        )
    
    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        raise ValueError(f"Video with ID {video_id} not found in database")
        
    if not video.video_file:
        raise ValueError(f"No video file for video {video_id}")
    
    video_path = video.video_file.path
    logger.info(f"📹 Opening: {video_path}")
    
    if not os.path.exists(video_path):
        debug_file_access(video_id, video_path)
        raise FileNotFoundError(
            f"Video file not found: {video_path}\n"
            f"This usually means:\n"
            f"1. The bucket-volume is not mounted to the celery-worker service\n"
            f"2. The volume mount path doesn't match MEDIA_ROOT\n"
            f"3. The file was not properly saved by the Server service\n"
            f"\nFIX: Mount bucket-volume to /app/data in BOTH Server and celery-worker services in Railway"
        )
    
    logger.info(f"✅ Video file found at: {video_path}")
    logger.info(f"   File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")
    
    logger.info(f"✅ Video opened successfully")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (frame_width, frame_height)
    
    logger.info(f"Processing video {video_id}: {frame_width}x{frame_height}, {fps}fps, {total_frames} frames")
    
    # Load a pre-trained YOLOv8 model
    model = get_model(model_id=os.getenv("MODEL_ID"))
    logger.info(f"YOLO model loaded: {os.getenv('MODEL_ID')}")
    
    # Initialize MediaPipe pose estimation
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    pose = mp_pose.Pose(
        # static_image_mode=False,
        # model_complexity=1,
        # smooth_landmarks=True,
        # enable_segmentation=False,
        # smooth_segmentation=True,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2
    )
    logger.info("MediaPipe Pose initialized")
    
    # Initialize video writer for saving annotated video
    video_output_path = f'{video_name}_processed.mp4'
    video_output = cv2.VideoWriter(
        video_output_path, 
        cv2.VideoWriter_fourcc(*'mp4v'),  # Use 'mp4v' codec for compatibility
        fps, 
        size
    )
    
    # Initialize a dictionary to store all extracted information
    video_data = {}
    frame_idx = 0
    yolo_detections = 0
    mediapipe_detections = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("image not read ")
            break
        print("image read")
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Run MediaPipe pose estimation
        pose_results = pose.process(image_rgb)
        # print(pose_results)
        
        image_rgb.flags.writeable = True
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Run YOLOv8 inference on the current frame
        results = model.infer(image_rgb)[0]
        
        # Extract class name and keypoints for the current frame
        frame_data = {
            "class_name": None,
            "keypoints": [],
            "mediapipe_pose_landmarks": []
        }
        
        # Process YOLOv8 results
        if len(results.predictions) > 0:
            yolo_detections += 1
            # Extract class name
            class_name = results.predictions[0].class_name
            frame_data["class_name"] = class_name
            # print(f"YOLOv8 detected: {class_name}")
            
            # Extract YOLOv8 keypoints
            for keypoint in results.predictions[0].keypoints:
                x = keypoint.x
                y = keypoint.y
                confidence = keypoint.confidence
                frame_data["keypoints"].append({
                    "x": x,
                    "y": y,
                    "confidence": confidence,
                    "keypoint_id": keypoint.class_id,
                    "class_name": keypoint.class_name
                })
        
        # Process MediaPipe pose results
        if pose_results.pose_landmarks:
            mediapipe_detections += 1
            # print("MediaPipe pose detected")
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                # Convert normalized coordinates to pixel coordinates
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                z = landmark.z  # Depth coordinate
                visibility = landmark.visibility
                
                frame_data["mediapipe_pose_landmarks"].append({
                    "landmark_id": idx,
                    "landmark_name": mp_pose.PoseLandmark(idx).name,
                    "x": x,
                    "y": y,
                    "z": z,
                    "visibility": visibility
                })
        
        # Add the frame data to the video data dictionary
        video_data[frame_idx] = frame_data
        
        # Load the YOLOv8 results into the supervision Detections API
        detections = sv.Detections.from_inference(results)
        
        # Create supervision annotators for YOLOv8
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # Annotate the image with YOLOv8 inference results
        annotated_image = bounding_box_annotator.annotate(
            scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)
        
        # Draw MediaPipe pose landmarks on the annotated image
        if pose_results.pose_landmarks:
            # print("processing")
            mp_drawing.draw_landmarks(
                annotated_image,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Write the annotated frame to the output video
        video_output.write(annotated_image)
        
        # Quit when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Increment the frame index
        frame_idx += 1
        # Log progress every 30 frames
        if frame_idx % 30 == 0:
            logger.info(f"Processed {frame_idx}/{total_frames} frames")
    
    # Release resources
    cap.release()
    video_output.release()
    cv2.destroyAllWindows()
    pose.close()
    
    logger.info(f"Video processing complete: {frame_idx} frames, YOLO: {yolo_detections}, MediaPipe: {mediapipe_detections}")
    
    # Save the extracted data to a JSON file
    video_data_path = f"{video_name}_data.json"
    with open(video_data_path, "w") as f:
        json.dump(video_data, f, indent=4)
    print("Video processing completed. Data saved to 'video_data.json'.")
    
    return video_data, video_output_path, video_data_path
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import base64
import os

app = Flask(__name__)

@dataclass
class MemePose:
    name: str
    image_path: str
    description: str
    priority: int

class MemePoseDetector:
    def __init__(self):
        base_options_face = python.BaseOptions(model_asset_path='face_landmarker.task')
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_face)
        
        base_options_pose = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
        options_pose = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            running_mode=vision.RunningMode.VIDEO
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)
        
        self.meme_poses = [
            MemePose("Shaq Timeout", "shaq.jpeg", "Make a T/timeout pose", 4),
            MemePose("Raised Eyebrow Cat", "raised_eyebrow_cat.jpeg", "Raise one eyebrow", 3),
            MemePose("Screaming Cat", "screaming_cat.jpeg", "Open your mouth wide", 2),
            MemePose("Thinking Monkey", "thinking_monkey.jpeg", "Finger near lips", 1),
            MemePose("Staring Cat", "staring_cat.jpeg", "Default", 0)
        ]
        
        self.current_meme = None
        self.detection_frames = 0
        self.required_frames = 5
        self.frame_count = 0
        
        self.meme_images = {}
        for meme in self.meme_poses:
            try:
                img = cv2.imread(meme.image_path)
                if img is not None:
                    h, w = img.shape[:2]
                    scale = 400 / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    self.meme_images[meme.name] = cv2.resize(img, (new_w, new_h))
            except Exception as e:
                print(f"Error loading {meme.image_path}: {e}")
    
    def detect_timeout_pose(self, pose_landmarks, face_landmarks):
        """Detect T/timeout pose - check multiple hand points for robustness"""
        if not pose_landmarks:
            return False
        
        pose_lm = pose_landmarks[0]
        
        # Get multiple hand landmarks (wrists, thumbs, index fingers)
        left_wrist = pose_lm[15]
        right_wrist = pose_lm[16]
        left_thumb = pose_lm[21]
        right_thumb = pose_lm[22]
        left_index = pose_lm[19]
        right_index = pose_lm[20]
        
        # Create list of all hand points to check
        left_points = [left_wrist, left_thumb, left_index]
        right_points = [right_wrist, right_thumb, right_index]
        
        # Check all combinations - if ANY pair is close, trigger T-pose
        threshold = 0.35  # Very generous threshold
        
        for left_point in left_points:
            for right_point in right_points:
                distance = np.sqrt((left_point.x - right_point.x)**2 + 
                                 (left_point.y - right_point.y)**2)
                if distance < threshold:
                    return True
        
        return False
    
    def detect_raised_eyebrow(self, face_landmarks):
        left_eyebrow_indices = [70, 63, 105, 66, 107]
        right_eyebrow_indices = [300, 293, 334, 296, 336]
        landmarks = face_landmarks[0]
        
        left_eyebrow_y = np.mean([landmarks[i].y for i in left_eyebrow_indices])
        right_eyebrow_y = np.mean([landmarks[i].y for i in right_eyebrow_indices])
        left_eye_y = landmarks[133].y
        right_eye_y = landmarks[362].y
        
        left_raise = left_eye_y - left_eyebrow_y
        right_raise = right_eye_y - right_eyebrow_y
        eyebrow_diff = abs(left_raise - right_raise)
        avg_raise = (left_raise + right_raise) / 2
        
        return eyebrow_diff > 0.01 or avg_raise > 0.08
    
    def detect_mouth_open(self, face_landmarks):
        landmarks = face_landmarks[0]
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        mouth_opening = abs(upper_lip.y - lower_lip.y)
        return mouth_opening > 0.04
    
    def detect_finger_to_mouth(self, pose_landmarks, face_landmarks):
        if not pose_landmarks or not face_landmarks:
            return False
        
        face_lm = face_landmarks[0]
        pose_lm = pose_landmarks[0]
        
        lip_area_indices = [0, 37, 39, 267, 269, 13, 14, 17, 84, 86, 314, 316, 61, 291, 78, 308]
        lip_area_points = [face_lm[i] for i in lip_area_indices]
        
        finger_indices = [19, 17, 21, 15, 20, 18, 22, 16]
        finger_points = [pose_lm[i] for i in finger_indices]
        
        threshold = 0.18
        
        for finger in finger_points:
            for lip_point in lip_area_points:
                distance = np.sqrt((finger.x - lip_point.x)**2 + (finger.y - lip_point.y)**2)
                if distance < threshold:
                    return True
        return False
    
    def match_pose(self, face_landmarks, pose_landmarks):
        if self.detect_timeout_pose(pose_landmarks, face_landmarks):
            return next(m for m in self.meme_poses if m.name == "Shaq Timeout")
        if self.detect_finger_to_mouth(pose_landmarks, face_landmarks):
            return next(m for m in self.meme_poses if m.name == "Thinking Monkey")
        if self.detect_mouth_open(face_landmarks):
            return next(m for m in self.meme_poses if m.name == "Screaming Cat")
        if self.detect_raised_eyebrow(face_landmarks):
            return next(m for m in self.meme_poses if m.name == "Raised Eyebrow Cat")
        return next(m for m in self.meme_poses if m.name == "Staring Cat")
    
    def process_frame(self, frame):
        self.frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(self.frame_count * 33.33)
        
        face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        face_landmarks = face_result.face_landmarks if face_result.face_landmarks else None
        pose_landmarks = pose_result.pose_landmarks if pose_result.pose_landmarks else None
        
        display_meme = None
        
        if face_landmarks:
            matched_meme = self.match_pose(face_landmarks, pose_landmarks)
            
            if matched_meme:
                if matched_meme == self.current_meme:
                    self.detection_frames += 1
                else:
                    self.current_meme = matched_meme
                    self.detection_frames = 1
            else:
                if self.detection_frames > 0:
                    self.detection_frames -= 1
                else:
                    self.current_meme = None
            
            if self.detection_frames >= self.required_frames and self.current_meme:
                display_meme = self.current_meme
            else:
                display_meme = next(m for m in self.meme_poses if m.name == "Staring Cat")
        else:
            display_meme = next(m for m in self.meme_poses if m.name == "Staring Cat")
        
        if display_meme and display_meme.name in self.meme_images:
            meme_img = self.meme_images[display_meme.name]
            mh, mw = meme_img.shape[:2]
            x_offset = w - mw - 20
            y_offset = 20
            
            if y_offset + mh <= h and x_offset + mw <= w:
                frame[y_offset:y_offset+mh, x_offset:x_offset+mw] = meme_img
        
        return frame

detector = MemePoseDetector()

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = detector.process_frame(frame)
        ret, buffer = cv2.imencode('.jpeg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

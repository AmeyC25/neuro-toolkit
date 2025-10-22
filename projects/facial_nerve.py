import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Landmark indices for facial nerve assessment
FACIAL_LANDMARKS = {
    "mouth_left": 291,
    "mouth_right": 61,
    "eyebrow_left": 334,
    "eyebrow_right": 105,
    "eye_top_left": 386,
    "eye_bottom_left": 374,
    "eye_top_right": 159,
    "eye_bottom_right": 145,
    "nose_tip": 1
}

if 'baseline_coords' not in st.session_state:
    st.session_state.baseline_coords = None

class FacialNerveTransformer(VideoTransformerBase):
    def __init__(self):
        self.task = "None"

    def set_task(self, task):
        self.task = task
        if task == "Set Baseline":
            st.session_state.baseline_coords = None # Reset baseline

    def _get_coords(self, landmarks, idx, w, h):
        lm = landmarks.landmark[idx]
        return int(lm.x * w), int(lm.y * h)
    
    def _get_dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Get all key coordinates
            coords = {}
            for name, idx in FACIAL_LANDMARKS.items():
                coords[name] = self._get_coords(landmarks, idx, w, h)
                
            # Draw key points
            for name, coord in coords.items():
                color = (0, 255, 0)
                if 'mouth' in name: color = (0, 0, 255)
                elif 'eye' in name: color = (255, 0, 0)
                cv2.circle(img, coord, 3, color, -1)

            if self.task == "Set Baseline":
                st.session_state.baseline_coords = coords
                cv2.putText(img, "Baseline Set!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if st.session_state.baseline_coords is not None:
                baseline = st.session_state.baseline_coords
                
                if self.task == "Smile":
                    # Compare horizontal excursion of mouth corners from nose
                    left_dist = abs(coords["mouth_left"][0] - baseline["mouth_left"][0])
                    right_dist = abs(coords["mouth_right"][0] - baseline["mouth_right"][0])
                    
                    if (left_dist + right_dist) > 10: # Threshold to avoid div by zero
                        symmetry = min(left_dist, right_dist) / max(left_dist, right_dist) * 100
                        cv2.putText(img, f"Smile Symmetry: {symmetry:.1f}%", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                elif self.task == "Raise Eyebrows":
                    # Compare vertical excursion of eyebrows from baseline
                    left_raise = baseline["eyebrow_left"][1] - coords["eyebrow_left"][1]
                    right_raise = baseline["eyebrow_right"][1] - coords["eyebrow_right"][1]
                    
                    if (left_raise + right_raise) > 5:
                        symmetry = min(left_raise, right_raise) / max(left_raise, right_raise) * 100
                        cv2.putText(img, f"Eyebrow Symmetry: {symmetry:.1f}%", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                elif self.task == "Close Eyes Tightly":
                    # Compare distance between upper and lower eyelids
                    left_opening = self._get_dist(coords["eye_top_left"], coords["eye_bottom_left"])
                    right_opening = self._get_dist(coords["eye_top_right"], coords["eye_bottom_right"])
                    
                    if (left_opening + right_opening) > 0:
                        symmetry = min(left_opening, right_opening) / max(left_opening, right_opening) * 100
                        cv2.putText(img, f"Eye Closure (Smaller is better)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(img, f"Left: {left_opening:.1f} Right: {right_opening:.1f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return img

def run():
    st.header("Facial Nerve (C.N. VII) Analyzer")
    
    with st.expander("ℹ️ Instructions & Explanation", expanded=True):
        st.markdown("""
        This tool quantifies facial symmetry, a key part of assessing Cranial Nerve VII function (used in F.A.S.T. stroke checks or for Bell's Palsy).
        
        **How to Use:**
        1.  Sit facing the camera in a well-lit room.
        2.  Click "Start" to begin the webcam feed.
        3.  Select **"Set Baseline"** from the dropdown and hold a neutral, relaxed face.
        4.  Select **"Smile"** and smile as widely as you can. A symmetry score will appear.
        5.  Select **"Raise Eyebrows"** and raise them as high as you can.
        6.  Select **"Close Eyes Tightly"** and squeeze your eyes shut.
        
        The app provides real-time feedback. A score of 100% is perfectly symmetrical.
        """)

    task = st.selectbox("Choose Task", ["None", "Set Baseline", "Smile", "Raise Eyebrows", "Close Eyes Tightly"])

    ctx = webrtc_streamer(
        key="facial-nerve",
        video_transformer_factory=FacialNerveTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

    if ctx.video_transformer:
        ctx.video_transformer.set_task(task)
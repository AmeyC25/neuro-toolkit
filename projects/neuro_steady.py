import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, 
                       min_detection_confidence=0.7, 
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# This list will store the points for tremor analysis
if 'points' not in st.session_state:
    st.session_state.points = []

# The target for the "hold steady" task
TARGET_POS = (320, 240)  # Center of a 640x480 frame
TARGET_RADIUS = 20

class DexterityTransformer(VideoTransformerBase):
    def __init__(self):
        self.task = "None"
        self.start_time = None
        self.tracking = False

    def set_task(self, task):
        self.task = task
        self.tracking = (task == "Hold Steady")
        if self.tracking:
            st.session_state.points = [] # Clear points on new task
            self.start_time = time.time()
        else:
            self.start_time = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Flip horizontally for a "mirror" view
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if self.task == "Hold Steady":
            # Draw the target circle
            cv2.circle(img, TARGET_POS, TARGET_RADIUS, (0, 255, 0), 2)
            
            if self.start_time and (time.time() - self.start_time) > 10:
                # Stop tracking after 10 seconds
                self.tracking = False
                self.task = "None"
                self.start_time = None

        if results.multi_hand_landmarks and self.tracking:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get coords for index finger tip (Landmark 8)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            
            # Draw a circle at the fingertip
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
            
            # Add point to our list for analysis
            st.session_state.points.append((cx, cy))

            if self.task == "Hold Steady":
                # Check if tip is in the target
                dist = np.sqrt((cx - TARGET_POS[0])**2 + (cy - TARGET_POS[1])**2)
                if dist <= TARGET_RADIUS:
                    # In target: Green circle
                    cv2.circle(img, TARGET_POS, TARGET_RADIUS, (0, 255, 0), -1)
                else:
                    # Out of target: Red circle
                    cv2.circle(img, TARGET_POS, TARGET_RADIUS, (0, 0, 255), -1)
        
        elif results.multi_hand_landmarks:
             # Hand is detected, but tracking is off. Just draw the hand.
             mp_drawing.draw_landmarks(img, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        return img

def run():
    st.header("Neuro-Steady: Dexterity & Tremor Analyzer")
    
    with st.expander("ℹ️ Instructions & Explanation", expanded=True):
        st.markdown("""
        This tool analyzes fine-motor control and tremor by tracking your index finger.
        
        **How to Use:**
        1.  Select the "Hold Steady" task from the dropdown.
        2.  Click "Start" to begin the webcam feed.
        3.  Place your hand in the frame until you see the blue landmarks.
        4.  Click the "Start 10-Second Tracking" button.
        5.  **Task:** Try to hold your index finger (blue dot) inside the green target circle.
        6.  After 10 seconds, the test will stop and you can click "Analyze Results".
        """)

    task = st.selectbox("Choose Task", ["None", "Hold Steady"])

    ctx = webrtc_streamer(
        key="dexterity",
        video_transformer_factory=DexterityTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

    if ctx.video_transformer:
        ctx.video_transformer.set_task(task)
        
        if st.button("Start 10-Second Tracking"):
            ctx.video_transformer.tracking = True
            ctx.video_transformer.start_time = time.time()
            st.session_state.points = [] # Clear old points
            st.info("Tracking... Hold steady for 10 seconds.")
    
    st.markdown("---")
    
    if st.button("Analyze Results"):
        if 'points' not in st.session_state or not st.session_state.points:
            st.warning("No tracking data collected. Please run the 'Start Tracking' task first.")
        else:
            points = np.array(st.session_state.points)
            
            # 1. Steadiness Analysis (Standard Deviation)
            std_x = np.std(points[:, 0])
            std_y = np.std(points[:, 1])
            steadiness_score = np.mean([std_x, std_y])
            
            # 2. Precision Analysis (Mean Distance from Target)
            distances = [np.sqrt((p[0] - TARGET_POS[0])**2 + (p[1] - TARGET_POS[1])**2) for p in points]
            precision_score = np.mean(distances)
            
            st.header("Analysis Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Steadiness Score (Lower is Better)", 
                          value=f"{steadiness_score:.2f} px")
                st.markdown("*(Average pixel deviation from your mean position. Measures 'jitter'.)*")
            with col2:
                st.metric(label="Precision Score (Lower is Better)", 
                          value=f"{precision_score:.2f} px")
                st.markdown("*(Average pixel distance from the center of the target.)*")

            # Plot the path
            st.subheader("Your Hand's Path")
            path_img = np.full((480, 640, 3), 255, dtype=np.uint8) # White background
            cv2.circle(path_img, TARGET_POS, TARGET_RADIUS, (0, 255, 0), 2) # Draw target
            
            if len(points) > 1:
                cv2.polylines(path_img, [points.astype(np.int32)], isClosed=False, color=(0, 0, 0), thickness=1)
            cv2.circle(path_img, tuple(points[0].astype(int)), 5, (255, 0, 0), -1)
            cv2.circle(path_img, tuple(points[-1].astype(int)), 5, (0, 0, 255), -1)
            
            st.image(path_img, caption="Blue dot = Start, Red dot = End")
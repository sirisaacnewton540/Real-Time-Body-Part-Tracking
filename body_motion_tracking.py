import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import threading
from PIL import Image, ImageTk

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variables to hold the selected options
selected_options = {
    'Hand': False,
    'Face': False,
    'Body': False,
    'Iris': False
}

# Function to draw and track specific landmarks
def draw_and_track(image, landmarks, track_index, color, track_history):
    h, w, _ = image.shape
    if track_index >= len(landmarks.landmark):
        return
    lm = landmarks.landmark[track_index]
    cx, cy = int(lm.x * w), int(lm.y * h)
    if track_index not in track_history:
        track_history[track_index] = []
    track_history[track_index].append((cx, cy))
    for i in range(1, len(track_history[track_index])):
        cv2.line(image, track_history[track_index][i - 1], track_history[track_index][i], color, 2)

# Function to draw all landmarks
def draw_all_landmarks(image, landmarks, connections, color):
    h, w, _ = image.shape
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx < len(landmarks.landmark) and end_idx < len(landmarks.landmark):
            start_point = (int(landmarks.landmark[start_idx].x * w), int(landmarks.landmark[start_idx].y * h))
            end_point = (int(landmarks.landmark[end_idx].x * w), int(landmarks.landmark[end_idx].y * h))
            cv2.line(image, start_point, end_point, color, 2)
    for lm in landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 3, color, -1)

# Function to process video frames
def process_frames():
    cap = cv2.VideoCapture(1)
    track_history = {}
    blank_frame = None

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
         mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Initialize the blank frame with the same size as the webcam frame
            if blank_frame is None:
                blank_frame = np.zeros_like(frame)

            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            # Process the image for hands, face, and pose
            hand_results = hands.process(frame_rgb)
            face_results = face_mesh.process(frame_rgb)
            pose_results = pose.process(frame_rgb)

            # Convert the image color back to BGR
            frame_rgb.flags.writeable = True
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Copy blank frame for drawing
            skeleton_frame = blank_frame.copy()

            # Track and draw hand landmarks (prioritize right hand)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x > 0.5:  # Assuming right hand
                        draw_all_landmarks(skeleton_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, (0, 255, 0))
                        if selected_options['Hand']:
                            draw_and_track(skeleton_frame, hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, (0, 100, 0), track_history)
                        break
                else:
                    hand_landmarks = hand_results.multi_hand_landmarks[0]
                    draw_all_landmarks(skeleton_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, (0, 255, 0))
                    if selected_options['Hand']:
                        draw_and_track(skeleton_frame, hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, (0, 100, 0), track_history)

            # Track and draw face landmarks
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    draw_all_landmarks(skeleton_frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, (255, 0, 0))
                    if selected_options['Face']:
                        draw_and_track(skeleton_frame, face_landmarks, 1, (139, 0, 0), track_history)  # Nose tip

            # Track and draw iris landmarks
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    if selected_options['Iris']:
                        draw_and_track(skeleton_frame, face_landmarks, 468, (255, 140, 0), track_history)  # Left iris landmark
                    try:
                        for idx in range(468, 473):
                            if idx < len(face_landmarks.landmark) and (idx + 1) < len(face_landmarks.landmark):
                                start_idx = idx
                                end_idx = idx + 1
                                start_point = (int(face_landmarks.landmark[start_idx].x * w), int(face_landmarks.landmark[start_idx].y * h))
                                end_point = (int(face_landmarks.landmark[end_idx].x * w), int(face_landmarks.landmark[end_idx].y * h))
                                cv2.line(skeleton_frame, start_point, end_point, (255, 140, 0), 2)
                    except ValueError:
                        pass

            # Track and draw pose landmarks
            if pose_results.pose_landmarks:
                draw_all_landmarks(skeleton_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, (255, 255, 0))
                if selected_options['Body']:
                    if mp_pose.PoseLandmark.RIGHT_SHOULDER < len(pose_results.pose_landmarks.landmark):
                        draw_and_track(skeleton_frame, pose_results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, (0, 0, 139), track_history)

            # Display the skeleton frame
            cv2.imshow('Tracking', skeleton_frame)

            if cv2.waitKey(10) & 0xFF == 27:
                break

    # Save the final tracking image
    cv2.imwrite('tracking_result.png', skeleton_frame)
    cap.release()
    cv2.destroyAllWindows()

    # Show the final tracking image as a pop-up
    show_final_image('tracking_result.png')

# Function to show the final image as a pop-up
def show_final_image(image_path):
    final_image = Image.open(image_path)
    final_image.show()

# Function to start the video processing in a separate thread
def start_processing():
    threading.Thread(target=process_frames).start()

# Function to create the GUI
def create_gui():
    root = tk.Tk()
    root.title("Pose Estimation Options")

    def update_option(option):
        selected_options[option] = not selected_options[option]

    tk.Checkbutton(root, text="Hand", command=lambda: update_option('Hand')).pack(anchor=tk.W)
    tk.Checkbutton(root, text="Face", command=lambda: update_option('Face')).pack(anchor=tk.W)
    tk.Checkbutton(root, text="Body", command=lambda: update_option('Body')).pack(anchor=tk.W)
    tk.Checkbutton(root, text="Iris", command=lambda: update_option('Iris')).pack(anchor=tk.W)

    tk.Button(root, text="Start", command=start_processing).pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui()

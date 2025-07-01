import cv2
import numpy as np
import mediapipe as mp
import time
import os
import pygame

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load("alert.wav")

# Thresholds
EAR_THRESHOLD = 0.21
CONSECUTIVE_FRAMES = 30  # Adjust based on how many frames eye must be closed to be considered drowsy

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Drawing specs
mp_drawing = mp.solutions.drawing_utils
draw_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Eye landmark indices for EAR calculation
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Function to calculate EAR
def calculate_EAR(landmarks, eye_indices, image_width, image_height):
    coords = [(int(landmarks[idx].x * image_width), int(landmarks[idx].y * image_height)) for idx in eye_indices]
    
    # EAR formula
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    ear = (A + B) / (2.0 * C)
    
    return ear

# Video capture
cap = cv2.VideoCapture(0)
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            left_ear = calculate_EAR(landmarks.landmark, LEFT_EYE, w, h)
            right_ear = calculate_EAR(landmarks.landmark, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            cv2.putText(frame, f'EAR: {avg_ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CONSECUTIVE_FRAMES:
                    cv2.putText(frame, "⚠️ WAKE UP!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
            else:
                frame_counter = 0
                pygame.mixer.music.stop()

    cv2.imshow("TaniX Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import datetime
import tkinter as tk
from tkinter import messagebox
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def load_accessory(name):
    global accessory
    file_path = os.path.join('assets', f'{name}.png')
    if os.path.exists(file_path):
        accessory = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if accessory is None or accessory.shape[2] != 4:
            messagebox.showerror("Error", "Accessory must be a PNG image with an alpha channel.")
        else:
            print(f"Accessory '{name}' loaded successfully.")
    else:
        messagebox.showerror("Error", f"Accessory '{name}' not found.")

def create_interface():
    root = tk.Tk()
    root.title("Virtual Try-On Accessory Selector")

    tk.Label(root, text="Enter accessory name:").pack(pady=5)
    accessory_name = tk.Entry(root)
    accessory_name.pack(pady=5)

    def on_apply():
        name = accessory_name.get().strip()
        if name:
            load_accessory(name)

    tk.Button(root, text="Apply Accessory", command=on_apply).pack(pady=10)
    root.mainloop()

accessory = cv2.imread('assets/mask.png', cv2.IMREAD_UNCHANGED)
if accessory is None or accessory.shape[2] != 4:
    raise ValueError("Default accessory must be a PNG image with an alpha channel.")


cap = cv2.VideoCapture(0)

brightness = 1.0  
screenshot_taken = False  

import threading
ui_thread = threading.Thread(target=create_interface)
ui_thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB as Mediapipe requires RGB input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face landmarks
    face_results = face_mesh.process(rgb_frame)

    # Process hand landmarks
    hand_results = hands.process(rgb_frame)

    # Overlay the accessory on the face
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            left_eye = (int(face_landmarks.landmark[33].x * frame.shape[1]),
                        int(face_landmarks.landmark[33].y * frame.shape[0]))
            right_eye = (int(face_landmarks.landmark[263].x * frame.shape[1]),
                         int(face_landmarks.landmark[263].y * frame.shape[0]))

            eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            width = int(np.linalg.norm(np.array(left_eye) - np.array(right_eye)) * 1.5)
            height = int(width * accessory.shape[0] / accessory.shape[1])

            resized_accessory = cv2.resize(accessory, (width, height))
            x_offset = max(0, eye_center[0] - width // 2)
            y_offset = max(0, eye_center[1] - height // 2)

            x_end = min(x_offset + width, frame.shape[1])
            y_end = min(y_offset + height, frame.shape[0])
            accessory_width = x_end - x_offset
            accessory_height = y_end - y_offset
            resized_accessory = resized_accessory[:accessory_height, :accessory_width]

            for c in range(0, 3):
                frame[y_offset:y_end, x_offset:x_end, c] = (
                    resized_accessory[:, :, c] * (resized_accessory[:, :, 3] / 255.0) +
                    frame[y_offset:y_end, x_offset:x_end, c] * (1.0 - resized_accessory[:, :, 3] / 255.0)
                )

    # Adjust brightness using hand gestures
    if hand_results.multi_hand_landmarks:
        for hand_landmarks, hand_classification in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            hand_label = hand_classification.classification[0].label

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            thumb_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            index_coords = (int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0]))

            cv2.circle(frame, thumb_coords, 10, (0, 255, 0), -1)
            cv2.circle(frame, index_coords, 10, (0, 255, 0), -1)
            cv2.line(frame, thumb_coords, index_coords, (255, 0, 0), 2)

            if hand_label == 'Right':
                distance = np.sqrt((thumb_tip.x - index_finger_tip.x) ** 2 +
                                   (thumb_tip.y - index_finger_tip.y) ** 2)
                if distance < 0.05:
                    brightness = max(0.5, brightness - 0.01)
                elif distance > 0.1:
                    brightness = min(1.5, brightness + 0.01)

            elif hand_label == 'Left':
                thumb_up = (thumb_tip.y < index_finger_tip.y)
                if thumb_up:
                    if not screenshot_taken:
                        screenshot_taken = True
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_filename = f'screenshot_{timestamp}.png'
                        cv2.imwrite(screenshot_filename, frame)
                        print(f"Screenshot saved as {screenshot_filename}")
                else:
                    screenshot_taken = False

    bright_frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)
    cv2.putText(bright_frame, f'Brightness: {brightness:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (122, 155, 255), 2)

    cv2.imshow('Virtual Try-On', bright_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Load the virtual accessory image (glasses) with an alpha channel
accessory = cv2.imread('assets\glasses.png', cv2.IMREAD_UNCHANGED)  # Ensure the image has an alpha channel

# Check if the accessory has an alpha channel; if not, add one
if accessory.shape[2] == 3:
    # Add an alpha channel if missing (fully opaque)
    accessory = cv2.cvtColor(accessory, cv2.COLOR_BGR2BGRA)

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB as Mediapipe requires RGB input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # Draw landmarks if any face is detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get coordinates for the left and right eye corners
            left_eye = (int(face_landmarks.landmark[33].x * frame.shape[1]),
                        int(face_landmarks.landmark[33].y * frame.shape[0]))
            right_eye = (int(face_landmarks.landmark[263].x * frame.shape[1]),
                         int(face_landmarks.landmark[263].y * frame.shape[0]))

            # Calculate accessory size and position
            eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)  # Midpoint between eyes
            width = int(np.linalg.norm(np.array(left_eye) - np.array(right_eye)) * 1.5)  # Scale width slightly larger
            height = int(width * accessory.shape[0] / accessory.shape[1])  # Maintain aspect ratio

            # Resize accessory to fit between the eyes
            resized_accessory = cv2.resize(accessory, (width, height))

            # Position the accessory above the eye center
            x_offset = max(0, eye_center[0] - width // 2)
            y_offset = max(0, eye_center[1] - height // 2)

            # Calculate the region of interest within the frame
            x_end = min(x_offset + width, frame.shape[1])
            y_end = min(y_offset + height, frame.shape[0])

            # Adjust the dimensions of the accessory if it goes out of bounds
            accessory_width = x_end - x_offset
            accessory_height = y_end - y_offset
            resized_accessory = resized_accessory[:accessory_height, :accessory_width]

            # Overlay the accessory with transparency
            for c in range(0, 3):  # Apply to each color channel
                frame[y_offset:y_end, x_offset:x_end, c] = (
                    resized_accessory[:, :, c] * (resized_accessory[:, :, 3] / 255.0) +
                    frame[y_offset:y_end, x_offset:x_end, c] * (1.0 - resized_accessory[:, :, 3] / 255.0)
                )

    # Display the frame with the virtual accessory
    cv2.imshow('Virtual Try-On', frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
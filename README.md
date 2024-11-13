# Virtual Try-On Project

This project uses **OpenCV**, **Mediapipe**, and **NumPy** to create an interactive virtual try-on experience. The application overlays a virtual accessory (e.g., glasses) onto the user's face using webcam input and allows users to control the brightness of the camera feed and capture screenshots using hand gestures.

## Features

- **Virtual Accessory Placement**: Automatically places a virtual accessory (such as glasses) on the user's face, aligned with their eyes.
- **Brightness Control**: Use hand gestures to adjust the brightness of the live video feed.
- **Screenshot Capture**: Capture a screenshot by performing a thumbs-up gesture with the left hand.
- **Real-Time Interaction**: Live webcam feed with real-time adjustments and accessory placement.

## Requirements

- **Python 3.x** (Tested on Python 3.7 and above)
- **OpenCV**: For webcam feed and image processing
- **Mediapipe**: For face and hand landmark detection
- **NumPy**: For numerical operations and image manipulation

To install the required dependencies, you can use the following `pip` command:

```bash
pip install opencv-python mediapipe numpy
```
## Assets Folder
The project relies on an accessory image (e.g., glasses or mask) that should be placed inside the  `assets` folder.
- assets/mask.png: This is the virtual accessory image used for the try-on. You can replace it with any image (e.g., glasses, hats) that has a transparent background (alpha channel).

## Running the Application
To run the application, simply execute the `main.py` file:
```bash
python main.py
```
The webcam will open, and the system will start processing the video feed. The virtual accessory will be placed on the user's face in real-time, and hand gestures will control brightness and allow for screenshot capture.

- Right Hand: Adjusts brightness by controlling the distance between the thumb and index finger.
- Left Hand: Captures a screenshot when a thumbs-up gesture is detected.

Press 'q' to exit the program.

## How It Works
- **Face Landmark Detection:** The program uses Mediapipe’s FaceMesh solution to detect face landmarks, particularly the eyes, to position the virtual accessory.
- **Hand Landmark Detection:** The program uses Mediapipe’s Hands solution to track the user's hand movements. The right hand controls brightness, and the left hand triggers screenshots with a thumbs-up gesture.
- **Accessory Placement:** The accessory image is resized based on the distance between the user's eyes and overlaid on the webcam feed. Transparency is respected using the alpha channel of the accessory image.
- **Brightness Control:** The right hand’s thumb and index finger distance is used to control the brightness of the video feed in real time.
- **Screenshot Functionality:** When the left hand forms a thumbs-up gesture, a screenshot is taken and saved with a timestamped filename.

## Working:
![Screenshot 2024-11-13 145056](https://github.com/user-attachments/assets/a80a3e75-7551-467d-a8a6-7828167a8710)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

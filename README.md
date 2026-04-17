# 🖐️ Hand Gesture Mouse Control

## 📌 About

This project uses computer vision and hand tracking to control the mouse using hand gestures via a webcam. It also supports clicking, scrolling, and a virtual keyboard.

## 🚀 Features

* Cursor movement using index finger
* Left click using thumb + index pinch
* Right click using thumb + middle pinch
* Scroll using two-finger gesture
* Virtual keyboard typing

## 🛠️ Tech Used

* Python
* OpenCV
* MediaPipe
* PyAutoGUI

## ▶️ How to Run

1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   python main.py

## ✋ Controls

* Move Index Finger → Move Cursor
* Thumb + Index → Left Click
* Thumb + Middle → Right Click
* Two Fingers Move → Scroll
* Tap on Virtual Keyboard → Type

## 🧠 How It Works

* Uses MediaPipe Hand Landmarker to detect 21 hand landmarks
* Tracks fingertip positions in real-time
* Maps hand coordinates to screen coordinates
* Detects gestures using distance between fingers
* Executes mouse actions using PyAutoGUI

## 🎯 Applications

* Touchless computer interaction
* Accessibility tools
* Gesture-based UI systems
* Smart interfaces

## ⚠️ Limitations

* Requires good lighting conditions
* Works best with a single hand
* Accuracy depends on camera quality

## 👨‍💻 Author

Shreyash Khot

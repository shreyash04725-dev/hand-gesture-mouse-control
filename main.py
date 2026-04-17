
import cv2
import numpy as np
import pyautogui
import time
import os
import urllib.request
from collections import deque
from virtual_keyboard import VirtualKeyboard

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)


pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


SCREEN_W, SCREEN_H = pyautogui.size()
CAM_W, CAM_H = 1280, 720


SMOOTH_FACTOR = 0.25
prev_mouse_x, prev_mouse_y = 0, 0


CLICK_THRESHOLD  = 35
SCROLL_THRESHOLD = 15
GESTURE_COOLDOWN = 0.35
KEY_COOLDOWN     = 0.4

last_left_click  = 0
last_right_click = 0
last_scroll_time = 0
last_key_time    = 0
finger_y_history = deque(maxlen=6)


latest_landmarks = [None]

vkb = VirtualKeyboard(start_x=80, start_y=400, key_w=70, key_h=60)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm knuckles
]


def distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def smooth_move(raw_x, raw_y):
    global prev_mouse_x, prev_mouse_y
    sx = prev_mouse_x + SMOOTH_FACTOR * (raw_x - prev_mouse_x)
    sy = prev_mouse_y + SMOOTH_FACTOR * (raw_y - prev_mouse_y)
    prev_mouse_x, prev_mouse_y = sx, sy
    return int(sx), int(sy)


def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def draw_hand(frame, landmarks):
    pts = [(int(lm.x * CAM_W), int(lm.y * CAM_H)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 200, 120), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (0, 160, 80),    1,  cv2.LINE_AA)


def get_model():
    path = "hand_landmarker.task"
    if not os.path.exists(path):
        print("Downloading hand landmark model (~9 MB) ...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            path,
        )
        print("Download complete.")
    return path


def on_result(result, output_image, timestamp_ms):
    latest_landmarks[0] = result.hand_landmarks[0] if result.hand_landmarks else None



def main():
    global last_left_click, last_right_click, last_scroll_time, last_key_time

    detector = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=get_model()),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
            result_callback=on_result,
        )
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 60)

    print("Gesture Keyboard active — press Q to quit.")
    ts = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detector.detect_async(mp_img, ts)
        ts += 1

        vkb.draw(frame)
        now = time.time()
        lms = latest_landmarks[0]

        if lms is not None:
            draw_hand(frame, lms)

            index_tip  = lm_px(lms[8],  CAM_W, CAM_H)
            thumb_tip  = lm_px(lms[4],  CAM_W, CAM_H)
            middle_tip = lm_px(lms[12], CAM_W, CAM_H)

            
            raw_mx = int(np.interp(index_tip[0], [0, CAM_W], [0, SCREEN_W]))
            raw_my = int(np.interp(index_tip[1], [0, int(CAM_H * 0.7)], [0, SCREEN_H]))
            pyautogui.moveTo(*smooth_move(raw_mx, raw_my))
            cv2.circle(frame, index_tip, 10, (0, 255, 100), -1)

           
            if distance(thumb_tip, index_tip) < CLICK_THRESHOLD and \
               now - last_left_click > GESTURE_COOLDOWN:
                pyautogui.click()
                last_left_click = now
                cv2.putText(frame, "LEFT CLICK", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            
            if distance(thumb_tip, middle_tip) < CLICK_THRESHOLD and \
               now - last_right_click > GESTURE_COOLDOWN:
                pyautogui.rightClick()
                last_right_click = now
                cv2.putText(frame, "RIGHT CLICK", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 3)

        
            if distance(index_tip, middle_tip) < 60:
                mid_y = (index_tip[1] + middle_tip[1]) // 2
                finger_y_history.append(mid_y)
                if len(finger_y_history) >= 4 and now - last_scroll_time > 0.1:
                    delta = finger_y_history[-1] - finger_y_history[0]
                    if abs(delta) > SCROLL_THRESHOLD:
                        pyautogui.scroll(-int(delta / 15))
                        last_scroll_time = now
                        cv2.putText(frame,
                                    "SCROLL UP" if delta < 0 else "SCROLL DOWN",
                                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (255, 200, 0), 3)
            else:
                finger_y_history.clear()

        
            if now - last_key_time > KEY_COOLDOWN:
                pressed = vkb.check_tap(index_tip, thumb_tip, CLICK_THRESHOLD)
                if pressed:
                    if   pressed == "SPACE": pyautogui.press("space")
                    elif pressed == "BACK":  pyautogui.press("backspace")
                    elif pressed == "ENTER": pyautogui.press("enter")
                    else:                    pyautogui.typewrite(pressed, interval=0.01)
                    last_key_time = now
                    vkb.set_active_key(pressed)

        cv2.putText(frame, "Gesture Keyboard | Q to quit",
                    (20, CAM_H - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 180, 180), 1)
        cv2.imshow("Gesture Keyboard & Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Session ended.")


if __name__ == "__main__":
    main()

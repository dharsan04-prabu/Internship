import cv2
import mediapipe as mp
import pyautogui
import time
import os
import numpy as np
import keyboard  # For better volume control

# Initialize mediapipe and camera
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.75,
                       min_tracking_confidence=0.75)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not detected. Exiting...")
    exit()

# States
spotify_opened = False
last_action_time = 0
cooldown = 1.0  # seconds
prev_hand_x = None
swipe_threshold = 80
prev_time = 0

def count_fingers(landmarks):
    tips = [8, 12, 16, 20]
    dips = [6, 10, 14, 18]
    count = 0
    for tip, dip in zip(tips, dips):
        if landmarks[tip].y < landmarks[dip].y:
            count += 1
    if landmarks[4].x < landmarks[3].x:
        count += 1
    return count

def get_hand_center_x(landmarks, width):
    x_coords = [lm.x for lm in landmarks]
    return int(np.mean(x_coords) * width)

def is_thumb_up(landmarks):
    return (landmarks[4].y < landmarks[3].y and landmarks[4].y < landmarks[2].y)

def is_thumb_down(landmarks):
    return (landmarks[4].y > landmarks[3].y and landmarks[4].y > landmarks[2].y)

def is_peace_sign(landmarks):
    return (landmarks[8].y < landmarks[6].y and
            landmarks[12].y < landmarks[10].y and
            all(landmarks[tip].y > landmarks[base].y for tip, base in zip([16, 20], [14, 18])))

def only_thumb_up(landmarks):
    thumb = is_thumb_up(landmarks)
    others = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    base = [landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
    return thumb and all(f.y > b.y for f, b in zip(others, base))

def only_thumb_down(landmarks):
    thumb = is_thumb_down(landmarks)
    others = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    base = [landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
    return thumb and all(f.y > b.y for f, b in zip(others, base))

while True:
    success, img = cap.read()
    if not success:
        print("⚠️ Failed to grab frame")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    img_height, img_width, _ = img.shape

    current_time = time.time()

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = hand_landmarks.landmark
        fingers = count_fingers(landmarks)
        hand_x = get_hand_center_x(landmarks, img_width)

        if current_time - last_action_time > cooldown:
            action_performed = False

            if fingers == 2 and not spotify_opened:
                print("🟢 Opening Spotify...")
                try:
                    os.system(r'start "" "C:\\Users\\vikra\\AppData\\Roaming\\Spotify\\Spotify.exe"')
                    spotify_opened = True
                    action_performed = True
                except Exception as e:
                    print(f"❌ Failed to open Spotify: {e}")

            elif fingers == 5:
                print("⏯️ Play/Pause toggle")
                pyautogui.press("playpause")
                action_performed = True

            elif prev_hand_x is not None:
                delta_x = hand_x - prev_hand_x
                if abs(delta_x) > swipe_threshold:
                    if delta_x < 0:
                        print("⏮️ Swipe Left: Previous Track")
                        pyautogui.press("prevtrack")
                    else:
                        print("⏭️ Swipe Right: Next Track")
                        pyautogui.press("nexttrack")
                    action_performed = True

            elif only_thumb_up(landmarks):
                print("🔊 Volume Up")
                keyboard.send("volume up")
                action_performed = True

            elif only_thumb_down(landmarks):
                print("🔉 Volume Down")
                keyboard.send("volume down")
                action_performed = True

            elif is_peace_sign(landmarks):
                print("✌️ Peace Sign Detected - Custom Action Triggered")
                action_performed = True

            if action_performed:
                last_action_time = current_time

        prev_hand_x = hand_x

    else:
        print("🙌 No hand detected")

    fps = int(1 / (current_time - prev_time)) if (current_time - prev_time) > 0 else 0
    prev_time = current_time
    cv2.putText(img, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Spotify Hand Controller", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

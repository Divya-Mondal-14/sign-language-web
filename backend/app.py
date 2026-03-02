from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import pyttsx3
import threading
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# Setup MediaPipe with better settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

def speak_text(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    threading.Thread(target=run, daemon=True).start()

def get_fingers(landmarks, handedness):
    fingers = []
    if handedness == "Right":
        fingers.append(1 if landmarks[4][0] < landmarks[3][0] else 0)
    else:
        fingers.append(1 if landmarks[4][0] > landmarks[3][0] else 0)

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(1 if landmarks[tip][1] < landmarks[pip][1] else 0)

    return fingers

def get_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def classify_gesture(fingers, landmarks):
    f = fingers
    thumb, index, middle, ring, pinky = f

    # ✊ Fist = Stop
    if f == [0, 0, 0, 0, 0]:
        return "Stop"

    # ✋ Open Hand = Hello
    if f == [1, 1, 1, 1, 1]:
        return "Hello"

    # ☝️ One finger
    if f == [0, 1, 0, 0, 0]:
        return "One"

    # ✌️ Peace
    if f == [0, 1, 1, 0, 0]:
        return "Peace"

    # 3 fingers
    if f == [0, 1, 1, 1, 0]:
        return "Three"

    # 4 fingers
    if f == [0, 1, 1, 1, 1]:
        return "Four"

    # 👍 Thumbs up = Yes
    if f == [1, 0, 0, 0, 0]:
        return "Yes"

    # 🤙 Call Me
    if f == [1, 0, 0, 0, 1]:
        return "Call Someone"

    # 🤟 I Love You
    if f == [1, 1, 0, 0, 1]:
        return "I Love You"

    # 🤘 Rock On
    if f == [0, 1, 0, 0, 1]:
        return "Rock On"

    # 🖐 Four fingers (no thumb)
    if f == [1, 1, 1, 0, 0]:
        return "Three"

    # 👌 Okay — thumb and index close together, other 3 up
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist_okay = get_distance(thumb_tip, index_tip)
    if dist_okay < 0.05 and middle == 1 and ring == 1 and pinky == 1:
        return "Okay"

    # 🤕 PAIN — index and middle crossed (tips close together), ring and pinky down
    index_tip_pt  = landmarks[8]
    middle_tip_pt = landmarks[12]
    dist_pain = get_distance(index_tip_pt, middle_tip_pt)
    if dist_pain < 0.04 and index == 1 and middle == 1 and ring == 0 and pinky == 0:
        return "Pain"

    # 🚽 TOILET — thumb between index and middle (T-shape)
    # Thumb up, index and middle up, ring and pinky down
    if f == [1, 1, 1, 0, 0]:
        thumb_x  = landmarks[4][0]
        index_x  = landmarks[8][0]
        middle_x = landmarks[12][0]
        if index_x < thumb_x < middle_x or middle_x < thumb_x < index_x:
            return "Toilet"

    # 🍎 FOOD — all four fingers together pointing up, thumb tucked
    if f == [0, 1, 1, 1, 1]:
        # Check all fingertips are close together (bunched)
        tips_pts = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        max_spread = max(
            get_distance(tips_pts[i], tips_pts[j])
            for i in range(len(tips_pts))
            for j in range(i+1, len(tips_pts))
        )
        if max_spread < 0.08:
            return "Food"

    # 💧 WATER — W shape: index, middle, ring up, thumb and pinky down
    if f == [0, 1, 1, 1, 0]:
        return "Water"

    # 🔥 FIRE — all fingers up and spread wide apart
    if f == [1, 1, 1, 1, 1]:
        tips_pts = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        min_spread = min(
            get_distance(tips_pts[i], tips_pts[j])
            for i in range(len(tips_pts))
            for j in range(i+1, len(tips_pts))
        )
        if min_spread > 0.08:
            return "Fire"

    # ⚠️ DANGER — index pointing up, all others curled, hand shaking
    # We detect it as index only pointing up but held at an angle
    if f == [0, 1, 0, 0, 0]:
        index_y_tip  = landmarks[8][1]
        index_y_base = landmarks[5][1]
        if index_y_tip < index_y_base - 0.15:
            return "Danger"

    return "Unknown"

@app.route('/detect', methods=['POST'])
def detect_gesture():
    try:
        data = request.json
        image_data = data['image']

        image_bytes = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_lms = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label

            landmarks = [[lm.x, lm.y] for lm in hand_lms.landmark]
            fingers = get_fingers(landmarks, handedness)
            gesture = classify_gesture(fingers, landmarks)

            return jsonify({
                'success': True,
                'gesture': gesture,
                'fingers': fingers,
                'hand': handedness
            })
        else:
            return jsonify({'success': False, 'gesture': None})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/speak', methods=['POST'])
def speak():
    try:
        text = request.json.get('text', '')
        if text:
            speak_text(text)
            return jsonify({'success': True})
        return jsonify({'success': False})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    print("✅ Backend running on http://localhost:5000")
    app.run(debug=True, port=5000)
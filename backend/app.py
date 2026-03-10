from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import threading
import pyttsx3

app = Flask(__name__)
CORS(app)

# ── TTS engine ──────────────────────────────────────────────────────────────
tts_lock = threading.Lock()

def speak(text):
    def _run():
        with tts_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"TTS error: {e}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()

# ── MediaPipe ────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# ── Semantic Gesture Map ──────────────────────────────────────────────────────
# Judge request #1: each gesture carries category, sub_type, severity, description
# This distinguishes "dog danger" vs "tiger danger", "chest pain" vs "head pain",
# "stop - do not touch" vs "stop - stay back" etc.

GESTURE_SEMANTICS = {
    # ── Basic Communication ──────────────────────────────────────────────
    "Hello": {
        "category": "greeting", "sub_type": "salutation",
        "severity": "none", "description": "General greeting gesture",
        "speech_text": "Hello"
    },
    "Yes": {
        "category": "affirmation", "sub_type": "positive_response",
        "severity": "none", "description": "Affirmative agreement",
        "speech_text": "Yes"
    },
    "Okay": {
        "category": "affirmation", "sub_type": "acknowledgement",
        "severity": "none", "description": "Acknowledgement or approval",
        "speech_text": "Okay"
    },
    "I Love You": {
        "category": "emotion", "sub_type": "affection",
        "severity": "none", "description": "Expression of love",
        "speech_text": "I Love You"
    },
    "Call Someone": {
        "category": "request", "sub_type": "communication_request",
        "severity": "low", "description": "Request to make a phone call",
        "speech_text": "Please call someone"
    },
    "Peace": {
        "category": "gesture", "sub_type": "symbol",
        "severity": "none", "description": "Peace sign",
        "speech_text": "Peace"
    },
    "One": {
        "category": "number", "sub_type": "count",
        "severity": "none", "description": "Number one",
        "speech_text": "One"
    },

    # ── STOP — sub_type distinguishes the REASON to stop ─────────────────
    "Stop": {
        "category": "stop", "sub_type": "general_stop",
        "severity": "medium", "description": "General stop — cease current action",
        "speech_text": "Stop"
    },
    "Stop - Do Not Touch": {
        "category": "stop", "sub_type": "do_not_touch",
        "severity": "high", "description": "Stop — do not touch this object or person",
        "speech_text": "Stop, do not touch"
    },
    "Stop - Stay Back": {
        "category": "stop", "sub_type": "stay_back",
        "severity": "high", "description": "Stop — maintain distance, stay where you are",
        "speech_text": "Stop, stay back"
    },
    "Stop - Wrong Way": {
        "category": "stop", "sub_type": "wrong_direction",
        "severity": "medium", "description": "Stop — you are going the wrong way",
        "speech_text": "Stop, wrong way"
    },

    # ── PAIN — sub_type identifies WHERE or WHAT kind of pain ────────────
    "Pain": {
        "category": "pain", "sub_type": "general_pain",
        "severity": "high", "description": "General pain — location unspecified",
        "speech_text": "I am in pain"
    },
    "Head Pain": {
        "category": "pain", "sub_type": "head_pain",
        "severity": "high", "description": "Pain in the head or headache",
        "speech_text": "I have head pain"
    },
    "Chest Pain": {
        "category": "pain", "sub_type": "chest_pain",
        "severity": "critical", "description": "Chest pain — possible cardiac event",
        "speech_text": "I have chest pain, please help"
    },
    "Stomach Pain": {
        "category": "pain", "sub_type": "stomach_pain",
        "severity": "high", "description": "Abdominal or stomach pain",
        "speech_text": "I have stomach pain"
    },

    # ── DANGER — sub_type distinguishes the SOURCE of danger ─────────────
    "Danger": {
        "category": "danger", "sub_type": "general_danger",
        "severity": "critical", "description": "General danger — unspecified threat",
        "speech_text": "Danger!"
    },
    "Animal Danger - Dog": {
        "category": "danger", "sub_type": "animal_threat_dog",
        "severity": "critical", "description": "Danger from a dog — aggressive animal nearby",
        "speech_text": "Danger! Aggressive dog nearby"
    },
    "Animal Danger - Tiger": {
        "category": "danger", "sub_type": "animal_threat_tiger",
        "severity": "critical", "description": "Danger from a tiger or large wild animal",
        "speech_text": "Danger! Wild animal, tiger nearby"
    },
    "Fire Danger": {
        "category": "danger", "sub_type": "fire_hazard",
        "severity": "critical", "description": "Fire danger — evacuate immediately",
        "speech_text": "Fire danger, evacuate now"
    },
    "Medical Danger": {
        "category": "danger", "sub_type": "medical_emergency",
        "severity": "critical", "description": "Medical emergency — person needs immediate care",
        "speech_text": "Medical emergency, call a doctor"
    },
    "Flood Danger": {
        "category": "danger", "sub_type": "flood_hazard",
        "severity": "critical", "description": "Flood or water danger",
        "speech_text": "Danger! Flooding nearby"
    },

    # ── Basic Needs ──────────────────────────────────────────────────────
    "Water": {
        "category": "basic_need", "sub_type": "hydration",
        "severity": "medium", "description": "Requesting water to drink",
        "speech_text": "I need water"
    },
    "Food": {
        "category": "basic_need", "sub_type": "nourishment",
        "severity": "medium", "description": "Requesting food",
        "speech_text": "I need food"
    },
    "Toilet": {
        "category": "basic_need", "sub_type": "restroom",
        "severity": "medium", "description": "Need to use the toilet",
        "speech_text": "I need the toilet"
    },
    "Fire": {
        "category": "danger", "sub_type": "fire_hazard",
        "severity": "critical", "description": "Fire emergency",
        "speech_text": "Fire! Emergency!"
    },
}

SEVERITY_PREFIX = {
    "critical": "CRITICAL ALERT! ",
    "high":     "Alert! ",
    "medium":   "",
    "low":      "",
    "none":     "",
}

# ── Finger-state helper ───────────────────────────────────────────────────────
def get_finger_states(hand_landmarks):
    lm = hand_landmarks.landmark
    tips  = [4, 8, 12, 16, 20]
    bases = [2, 6, 10, 14, 18]
    fingers = []
    fingers.append(1 if lm[tips[0]].x < lm[bases[0]].x else 0)
    for i in range(1, 5):
        fingers.append(1 if lm[tips[i]].y < lm[bases[i]].y else 0)
    return fingers

def are_fingers_crossed(lm):
    ix, iy = lm[8].x, lm[8].y
    mx, my = lm[12].x, lm[12].y
    return abs(ix - mx) < 0.05 and abs(iy - my) < 0.06

def is_okay_sign(lm):
    dist = ((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2) ** 0.5
    return dist < 0.06

def is_thumb_between_fingers(lm):
    return lm[4].y > lm[8].y and lm[4].y > lm[12].y

def classify_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    f  = get_finger_states(hand_landmarks)
    thumb, index, middle, ring, pinky = f

    if is_okay_sign(lm):
        return "Okay"
    if is_thumb_between_fingers(lm) and not any(f[1:]):
        return "Toilet"
    if are_fingers_crossed(lm):
        return "Pain"

    count = sum(f)

    if count == 0:
        return "Stop"
    if count == 5:
        spread = abs(lm[4].x - lm[20].x) + abs(lm[4].y - lm[20].y)
        return "Fire" if spread > 0.5 else "Hello"
    if thumb==1 and index==0 and middle==0 and ring==0 and pinky==0:
        return "Yes"
    if thumb==0 and index==1 and middle==0 and ring==0 and pinky==0:
        return "Danger"
    if thumb==0 and index==1 and middle==1 and ring==0 and pinky==0:
        return "Peace"
    if thumb==0 and index==1 and middle==1 and ring==1 and pinky==0:
        return "Water"
    if thumb==1 and index==0 and middle==0 and ring==0 and pinky==1:
        return "Call Someone"
    if thumb==1 and index==1 and middle==0 and ring==0 and pinky==1:
        return "I Love You"
    if thumb==0 and index==1 and middle==1 and ring==1 and pinky==1:
        return "Food"
    if count == 1 and index == 1:
        return "One"
    return "Unknown"

# ── Speech-to-Sign keyword map (Judge request #2) ─────────────────────────────
SPEECH_TO_SIGN = {
    "hello": ["Hello"], "hi": ["Hello"], "bye": ["Hello"],
    "yes": ["Yes"], "ok": ["Okay"], "okay": ["Okay"], "fine": ["Okay"],
    "stop": ["Stop"],
    "dont touch": ["Stop - Do Not Touch"], "do not touch": ["Stop - Do Not Touch"],
    "stay back": ["Stop - Stay Back"], "wrong way": ["Stop - Wrong Way"],
    "no entry": ["Stop - Stay Back"],
    "pain": ["Pain"], "hurt": ["Pain"], "ache": ["Pain"],
    "headache": ["Head Pain"], "head pain": ["Head Pain"], "head hurts": ["Head Pain"],
    "chest pain": ["Chest Pain"], "heart pain": ["Chest Pain"],
    "stomach pain": ["Stomach Pain"], "stomach ache": ["Stomach Pain"],
    "tummy ache": ["Stomach Pain"],
    "danger": ["Danger"], "help": ["Danger"],
    "emergency": ["Danger", "Medical Danger"],
    "dog": ["Animal Danger - Dog"], "aggressive dog": ["Animal Danger - Dog"],
    "dog attack": ["Animal Danger - Dog"],
    "tiger": ["Animal Danger - Tiger"], "wild animal": ["Animal Danger - Tiger"],
    "lion": ["Animal Danger - Tiger"], "leopard": ["Animal Danger - Tiger"],
    "fire": ["Fire Danger"], "fire danger": ["Fire Danger"],
    "flood": ["Flood Danger"], "flooding": ["Flood Danger"],
    "medical": ["Medical Danger"], "doctor": ["Medical Danger"],
    "ambulance": ["Medical Danger"],
    "water": ["Water"], "thirsty": ["Water"], "drink": ["Water"],
    "food": ["Food"], "hungry": ["Food"], "eat": ["Food"],
    "toilet": ["Toilet"], "bathroom": ["Toilet"], "restroom": ["Toilet"],
    "love": ["I Love You"], "i love you": ["I Love You"], "love you": ["I Love You"],
    "call": ["Call Someone"], "phone": ["Call Someone"],
    "call someone": ["Call Someone"],
}

def speech_to_sign_lookup(phrase: str):
    phrase_lower = phrase.lower().strip()
    results = []
    matched_gestures = set()
    for key, gestures in SPEECH_TO_SIGN.items():
        if key in phrase_lower:
            for g in gestures:
                if g not in matched_gestures:
                    matched_gestures.add(g)
                    sem = GESTURE_SEMANTICS.get(g, {})
                    results.append({"gesture": g, "matched_keyword": key, **sem})
    return results

# ── API Routes ────────────────────────────────────────────────────────────────

@app.route('/detect', methods=['POST'])
def detect_gesture():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    try:
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr    = np.frombuffer(img_data, np.uint8)
        frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results  = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return jsonify({'gesture': None, 'message': 'No hand detected'})

        gesture  = classify_gesture(results.multi_hand_landmarks[0])
        if gesture == 'Unknown':
            return jsonify({'gesture': 'Unknown', 'message': 'Gesture not recognized'})

        sem      = GESTURE_SEMANTICS.get(gesture, {
            "category": "unknown", "sub_type": "unknown",
            "severity": "none", "description": gesture, "speech_text": gesture
        })
        prefix   = SEVERITY_PREFIX.get(sem.get("severity", "none"), "")
        tts_text = prefix + sem.get("speech_text", gesture)
        speak(tts_text)

        return jsonify({
            'gesture':     gesture,
            'category':    sem['category'],
            'sub_type':    sem['sub_type'],
            'severity':    sem['severity'],
            'description': sem['description'],
            'speech_text': tts_text,
        })
    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/speech-to-sign', methods=['POST'])
def speech_to_sign():
    """NEW — Judge request #2. Text/speech phrase → matching sign gestures + semantics."""
    data = request.json
    if not data or 'phrase' not in data:
        return jsonify({'error': 'No phrase provided'}), 400
    phrase  = data['phrase']
    matches = speech_to_sign_lookup(phrase)
    if not matches:
        return jsonify({'phrase': phrase, 'matches': [], 'message': 'No matching signs found'})
    for m in matches:
        prefix = SEVERITY_PREFIX.get(m.get("severity", "none"), "")
        speak(prefix + m.get("speech_text", m["gesture"]))
    return jsonify({'phrase': phrase, 'matches': matches, 'count': len(matches)})


@app.route('/gesture-library', methods=['GET'])
def gesture_library():
    """NEW — Full semantic gesture catalogue grouped by category."""
    library = [{"gesture": name, **sem} for name, sem in GESTURE_SEMANTICS.items()]
    grouped = {}
    for item in library:
        grouped.setdefault(item["category"], []).append(item)
    return jsonify({"library": library, "grouped": grouped})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("✅ Backend running on http://localhost:5000")
    print("   POST /detect          — gesture from webcam frame")
    print("   POST /speech-to-sign  — speech/text → sign mapping (NEW)")
    print("   GET  /gesture-library — full semantic catalogue (NEW)")
    app.run(debug=False, host='0.0.0.0', port=5000)
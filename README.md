# 🤟 Sign Language to Speech Converter

> A real-time website that converts hand gestures into spoken words — bridging communication between deaf/mute individuals and the general public.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1-black?style=flat-square&logo=flask)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.11-orange?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![HTML](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-yellow?style=flat-square&logo=html5)

---

## 📌 What is this?

Over **70 million deaf people** worldwide use sign language as their primary communication method. Most of the general public doesn't understand sign language — creating a barrier in hospitals, schools, offices, and daily life.

This website solves that by:
- 📷 Detecting hand gestures via webcam in real-time
- 🧠 Using MediaPipe AI to identify hand landmarks
- 🔊 Converting recognized gestures to spoken audio instantly
- 🌐 Running entirely in the browser 

---

## 🎥 Demo
```
Camera Feed → MediaPipe AI → Gesture Classifier → Text-to-Speech → 🔊
```

---

## ✋ Supported Gestures

### Basic Gestures
| Gesture | Sign | Output |
|---------|------|--------|
| ✋ Open Hand | All fingers up | "Hello" |
| ✊ Fist | All fingers closed | "Stop" |
| 👍 Thumbs Up | Only thumb up | "Yes" |
| 👌 Okay Sign | Thumb + index circle | "Okay" |
| ☝️ Index Finger | Only index up | "One" |
| ✌️ Two Fingers | Index + middle up | "Peace" |
| 🤙 Call Someone | Thumb + pinky up | "Call Someone" |
| 🤟 ILY Sign | Thumb + index + pinky | "I Love You" |

### 🚨 Emergency Gestures
| Gesture | Sign | Output |
|---------|------|--------|
| 🤕 Crossed Fingers | Index + middle crossed | "Pain" |
| 🚽 T-Shape | Thumb between fingers | "Toilet" |
| 🍎 Bunched Fingers | All 4 fingers bunched | "Food" |
| 💧 W-Shape | Index + middle + ring up | "Water" |
| ⚠️ Index High | Index pointing straight up | "Danger" |
| 🔥 Wide Open | All fingers spread wide | "Fire" |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, Vanilla JavaScript |
| Backend | Python 3.11, Flask |
| AI / CV | MediaPipe Hands, OpenCV |
| Speech | pyttsx3 (offline Text-to-Speech) |
| Server | Python HTTP Server |

---

## Technical Approach

## 🏗 System Architecture Flow

```
Live Video Input (Webcam Feed)
        ↓
Hand Landmark Detection (MediaPipe – 21 Keypoints)
        ↓
Feature Vector Generation (Normalized X, Y, Z Values)
        ↓
Gesture Classification Model (Supervised ML – SVM / Random Forest)
        ↓
Text Mapping Layer (Gesture → Word Output)
        ↓
Text-to-Speech Engine (pyttsx3 / TTS API)
        ↓
Audio Output (Speaker Output)
```
## ⚙️ How It Works
```
1. Browser captures webcam frame every 800ms
2. Frame sent as base64 image to Flask backend
3. MediaPipe detects hand landmarks (x, y, z)
4. Finger positions calculated (up or down)
5. Rule-based classifier maps positions to gesture
6. Gesture name returned to browser
7. pyttsx3 speaks the gesture aloud
8. History logged with timestamp on screen
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11 (important — MediaPipe doesn't support 3.13)
- Google Chrome browser
- Webcam

---

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/sign-language-app.git
cd sign-language-app
```

**2. Create virtual environment with Python 3.11**
```bash
cd backend
py -3.11 -m venv venv
```

**3. Activate virtual environment**

Windows:
```bash
venv\Scripts\activate
```
Mac/Linux:
```bash
source venv/bin/activate
```

**4. Install dependencies**
```bash
pip install mediapipe==0.10.11 opencv-python flask flask-cors numpy pyttsx3
```

---

### Running the website

**Terminal 1 — Start Backend:**
```bash
cd backend
venv\Scripts\activate
python app.py
```
You should see:
```
✅ Backend running on http://localhost:5000
```

**Terminal 2 — Start Frontend:**
```bash
cd frontend
python -m http.server 8000
```

**Open Chrome and go to:**
```
http://localhost:8000
```

---

## 📁 Project Structure
```
sign-language-app/
│
├── backend/
│   ├── app.py              ← Flask API + MediaPipe + gesture classifier
│   └── venv/               ← Python virtual environment (not pushed)
│
├── frontend/
│   └── index.html          ← Complete website (HTML + CSS + JS)
│
├── .gitignore
└── README.md
```

---

## 🔥 Features

- ✅ Real-time gesture detection (every 800ms)
- ✅ Auto-speaks detected gesture
- ✅ Detection history with timestamps
- ✅ Click any gesture in guide to hear it
- ✅ Emergency signs (Pain, Fire, Danger, Water, Food, Toilet)
- ✅ Finger indicator dots (shows which fingers are up)
- ✅ Handles both left and right hand

---

## 🐛 Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| `No module named mediapipe` | Make sure venv is active and using Python 3.11 |
| Camera not working | Use Chrome browser, allow camera permissions |
| Cannot reach backend | Make sure `python app.py` is running in Terminal 1 |
| `run loop already started` | TTS conflict — restart the backend |
| Gesture not detected | Ensure good lighting and plain background |

---

## 🎯 Real World Use Cases

- 🏥 **Hospitals** — patients communicating needs to staff
- 🏫 **Schools** — inclusive classrooms for hearing impaired
- 🏢 **Customer service** — serving deaf customers
- 🏛️ **Public offices** — accessible government services
- 🏠 **Home** — daily communication aid for families

---

## 📈 Future Scope

- [ ] Full ASL alphabet (A–Z) recognition
- [ ] Mobile app version (React Native)
- [ ] Multi-language speech output
- [ ] Two-way communication (speech → text for the other person)
- [ ] ML model trained on thousands of gestures
- [ ] Cloud deployment for public access
- [ ] Support for multiple sign languages (ISL, BSL, ASL)

---

## 🌍 Social Impact

This project addresses **UN Sustainable Development Goal 10 — Reduced Inequalities** by making everyday communication accessible to the deaf and mute community using nothing more than a standard webcam and a browser.

---



## 👨‍💻 Built By

Made by **Team Byte Brownies** as a hackathon project to demonstrate how AI can solve real human communication barriers.

### 👥 Team Members

- **Divya Mondal (Team Lead)**  
  Backend & Frontend Development  

- **Srineeja Bhowmick**  
  Testing & Debugging  

- **Sharmistha Halder**  
  Ideation & Concept  

---
---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

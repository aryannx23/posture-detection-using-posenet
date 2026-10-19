# Posture Detection System Using PoseNet

An AI-based computer vision application that analyzes human body posture in real time using deep learning. The system uses **PoseNet** (via **TensorFlow.js**) to detect 17 anatomical keypoints from a live webcam feed and evaluates whether the user's posture is correct or incorrect.

---

## 📌 Project Overview

Prolonged improper sitting or standing posture leads to chronic musculoskeletal conditions such as cervical spondylosis, thoracic kyphosis ("hunchback"), forward head posture ("tech-neck"), and lower back strain.

This application provides real-time posture feedback right inside the browser with zero cloud round-trips (100% private, on-device inference).

### Core Features:
- **Real-Time Human Pose Detection**: High-frame-rate tracking at 30+ FPS using lightweight MobileNetV1 PoseNet architecture.
- **Body Keypoint Detection**: Identifies 17 landmark keypoints including nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles.
- **Posture Classification**: Classifies posture into *Optimal*, *Warning*, and *Poor / Slouching* across multiple anatomical dimensions:
  - **Head & Neck Alignment**: Detects lateral tilt and forward head posture.
  - **Shoulder Position**: Calculates acromioclavicular symmetry and shoulder hikes.
  - **Spine Angle**: Analyzes torso tilt and inclination relative to vertical baseline.
  - **Sitting Posture / Slouch**: Detects cervical-thoracic compression when hunching.
- **Live Camera Monitoring**: Seamless webcam integration with mirror view and silhouette guide overlay.
- **Skeleton Visualization**: Real-time color-coded skeleton overlay (emerald for optimal, amber for warning, ruby for deviation) with in-canvas angle readouts.
- **Instant Posture Feedback**: Real-time HUD alerts, soft chime alerts via Web Audio API, and voice alerts using the Web Speech Synthesis API.
- **Posture Health Score (0 - 100%)**: Dynamic gauge, current streak counter, deviation tracker, and live rolling timeline chart.
- **Personalized Calibration**: Calibrate your own unique neutral baseline posture with a single click.
- **Simulated Test Mode**: Built-in cinematic simulation to test the posture evaluation engine without needing a physical camera.

---

## 🔬 Technology Stack

- **PoseNet**: Pre-trained convolutional pose estimation model
- **TensorFlow.js**: WebGL-accelerated client-side deep learning execution
- **Machine Learning & Computer Vision**: Keypoint extraction, coordinate normalization, vector calculations
- **HTML5 & Vanilla CSS3**: Cybernetic dark-mode interface with glassmorphism, responsive grid, and zero external framework lock-in
- **JavaScript (ES6+)**: Modular application architecture, Web Audio API, Web Speech API, Chart.js

---

## 📐 How It Works & Mathematical Formulation

### 1. Camera Input
The webcam captures live video frames via `navigator.mediaDevices.getUserMedia`.

### 2. Pose Detection
Each video frame is passed to PoseNet:
```javascript
const pose = await net.estimateSinglePose(videoElement, {
  flipHorizontal: false,
  decodingMethod: 'single-person'
});
```

### 3. Keypoint Analysis & Angle Calculation
Posture angles are calculated using the 2-argument arctangent function:
$$\text{Angle} = \text{atan2}(y_2 - y_1, x_2 - x_1) \times \left(\frac{180}{\pi}\right)$$

- **Shoulder Tilt Angle:**
  $$\theta_{\text{shoulder}} = \text{atan2}(y_{\text{rightShoulder}} - y_{\text{leftShoulder}}, x_{\text{rightShoulder}} - x_{\text{leftShoulder}})$$
  - Normal range: $\pm 7.0^\circ$ relative to horizontal baseline.

- **Head Alignment Angle:**
  $$\theta_{\text{head}} = \text{atan2}(y_{\text{rightEye}} - y_{\text{leftEye}}, x_{\text{rightEye}} - x_{\text{leftEye}})$$
  - Normal range: $\pm 9.0^\circ$ lateral tilt.

- **Spine & Torso Angle:**
  $$\theta_{\text{spine}} = \text{atan2}(y_{\text{midHip}} - y_{\text{midShoulder}}, x_{\text{midHip}} - x_{\text{midShoulder}})$$
  - Normal upright range: $85^\circ - 98^\circ$.

- **Slouch & Compression Factor:**
  $$\text{Ratio} = \frac{\text{Distance}(\text{HeadCenter}, \text{MidShoulder}) / \text{ShoulderWidth}}{\text{Baseline}}$$
  - Slouch detected if $\text{Ratio} < 0.82$.

### 4. Posture Classification & Display Feedback
Calculated values are evaluated against calibrated ergonomic rules. Feedback is immediately displayed via:
- On-canvas skeleton color shift & warning bubbles
- Real-time posture score meter and stats
- Audio notification sound & spoken advice

---

## 🚀 How to Run the Application

Because browser security policies require a secure origin (`localhost` or `https://`) for camera access, serve the directory using any static web server:

### Option 1: Using Python (Recommended)
Open a terminal in this folder and run:
```bash
python -m http.server 8000
```
Then open your browser and navigate to:
```
http://localhost:8000
```

### Option 2: Using Node / VS Code Live Server
- If using VS Code, right-click `index.html` and choose **"Open with Live Server"**.
- Or run `npx serve .`

---

## 🗂️ Project Structure

```
Posture Posenet/
├── index.html                 # Main ergonomic dashboard & HTML5 structure
├── css/
│   └── style.css              # Cybernetic glassmorphic design system
├── js/
│   ├── audio-alerts.js        # Web Audio chime synthesis & Web Speech voice alerts
│   ├── posture-classifier.js  # Angle calculations (atan2) & classification logic
│   ├── posenet-logic.js       # PoseNet inference, smoothing, and simulation
│   ├── ui.js                  # Canvas skeleton rendering, HUD, and Chart.js integration
│   └── app.js                 # App coordinator, session tracking, and events
└── README.md                  # Comprehensive documentation
```

---

## 🎯 Applications

- **Workplace Ergonomics**: Prevent hunching and chronic back pain during desk work.
- **Healthcare & Physiotherapy**: Track patient posture rehabilitation progress.
- **Fitness Form Correction**: Verify body alignment during exercises.

---

## 🔮 Future Improvements

- [x] Posture scoring system & live rolling timeline
- [x] Voice alerts & synthetic audio chimes
- [x] Session statistics & JSON report export
- [ ] Mobile application support (React Native / PWA)
- [ ] Wearable device integration (Bluetooth IMU sensors)

---

## 👤 Author

**Aryan Raj Pandey**  
*B.Tech CSE (AI & ML)*

## 📄 License

Educational and research purposes.

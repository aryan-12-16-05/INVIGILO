# **INVIGILO Proctoring System - Complete Technical Documentation**

## **📋 OVERVIEW**

The proctoring system uses **AI-powered computer vision** and **behavioral monitoring** to detect suspicious activities during online exams in real-time.

---

## **🏗️ ARCHITECTURE**

### **System Components**

1. **Frontend (React + TypeScript)** - Student exam interface with camera/audio capture
2. **Backend (Python Flask)** - ML processing server with Socket.IO
3. **Database (MongoDB)** - Stores exam data, violations, evidence
4. **ML Models** - Dlib (facial landmarks), OpenCV (computer vision), YOLO (object detection)

---

## **📦 PACKAGES & TECHNOLOGIES**

### **Backend (Python)**

#### **Core Frameworks**
- `Flask` - Web server
- `Flask-Cors` - Cross-origin requests
- `flask-socketio` - Real-time WebSocket communication
- `pymongo` - MongoDB database driver
- `gunicorn` + `eventlet` - Production server

#### **ML & Computer Vision**
- `dlib` - 68-point facial landmark detection
- `opencv-python` (cv2) - Image processing, frame analysis
- `numpy` - Mathematical operations, array processing
- `imutils` - Image transformation utilities
- `face_recognition` - Face detection wrapper
- `scipy` - Scientific computing
- `insightface` + `onnxruntime` - Deep learning face recognition

#### **Audio Processing**
- `pyaudio` - Audio stream capture
- `numpy` - Audio signal analysis (ZCR, amplitude)

#### **Models Used**
- **Dlib Shape Predictor 68**: Detects 68 facial landmarks (eyes, nose, mouth, jawline)
- **Dlib Face Detector**: HOG-based frontal face detector
- **YOLOv3-tiny**: Real-time object detection (phones, books, people)

### **Frontend (JavaScript/TypeScript)**

#### **Core Libraries**
- `React 18` - UI framework
- `TypeScript` - Type-safe JavaScript
- `Vite` - Build tool and dev server
- `socket.io-client` - Real-time communication with backend

#### **UI Components**
- `framer-motion` - Smooth animations
- `lucide-react` - Icon library
- `@radix-ui/*` - Headless accessible components
- `tailwindcss` - Utility-first CSS
- `clsx` + `tailwind-merge` - Conditional styling

#### **Browser APIs**
- `MediaDevices API` - Camera and microphone access
- `MediaRecorder API` - Audio recording
- `Canvas API` - Frame capture and rendering
- `requestAnimationFrame` - 30 FPS smooth rendering
- `Fullscreen API` - Mandatory fullscreen mode
- `Screen Capture API` - Screen sharing (optional)

---

## **🎯 DETECTION ALGORITHMS & THRESHOLDS**

### **1. Face Detection**
**Algorithm**: Dlib HOG-based frontal face detector
**Purpose**: Ensure student is present and verify identity

**Thresholds**:
```python
faceMissingGraceMs: 4000  # 4 seconds before violation
```

**Violations Triggered**:
- `face_missing` - No face detected for 4+ seconds
- `multiple_faces` - More than 1 face detected
- `face_mismatch` - Different person detected (identity verification)

---

### **2. Eye Gaze Detection**
**Algorithm**: Pupil position analysis using 68-point landmarks
**Method**: 
1. Extract eye regions (landmarks 36-41 for left, 42-47 for right)
2. Apply binary threshold to isolate pupil/iris
3. Count white pixels on left vs right side of eye
4. Compare ratio to detect gaze direction

**Thresholds**:
```python
GAZE_RATIO_THRESHOLD = 1.3      # Stricter detection (was 1.2)
GAZE_THRESHOLD_VALUE = 45        # Binary segmentation (was 50)
```

**Detection Logic**:
```python
if rightSideWhite >= 1.3 * leftSideWhite:
    gaze = 'Left'  # Looking left
elif leftSideWhite >= 1.3 * rightSideWhite:
    gaze = 'Right'  # Looking right
else:
    gaze = 'Center'  # Looking at screen
```

**Violations**: 
- `gaze_aversion` - Eyes looking away from screen (Left/Right)

---

### **3. Head Pose Estimation**
**Algorithm**: 3D head pose using PnP (Perspective-n-Point)
**Method**:
1. Map 6 facial landmarks to 3D model points
2. Use `cv2.solvePnP` to get rotation/translation vectors
3. Calculate nose projection angle

**Thresholds**:
```python
HEAD_VERTICAL_ANGLE_THRESHOLD = 25    # Degrees (was 30)
HEAD_HORIZONTAL_OFFSET_THRESHOLD = 8  # Pixels (was 10)
```

**Detection Logic**:
```python
if angle >= 25°: return "Head Down"
if angle <= -25°: return "Head Up"
if nose_x < left_eye_x - 8px: return "Head Left"
if nose_x > right_eye_x + 8px: return "Head Right"
else: return "Forward"
```

**Violations**:
- `head_pose` - Head turned away (Down, Up, Left, Right)

---

### **4. Blink Detection**
**Algorithm**: Eye Aspect Ratio (EAR)
**Method**:
1. Calculate horizontal distance between eye corners
2. Calculate vertical distance (top to bottom eyelid)
3. Ratio = horizontal / vertical

**Thresholds**:
```python
BLINK_THRESHOLD = 3.6  # Higher = eyes more closed
```

**Detection Logic**:
```python
if leftEAR >= 3.6 or rightEAR >= 3.6:
    return "Blink"
else:
    return "No Blink"
```

**Purpose**: Liveness detection (not a photo), natural behavior monitoring

---

### **5. Mouth/Talking Detection**
**Algorithm**: Lip distance measurement
**Method**:
1. Measure pixel distance between outer top lip (landmark 51) and outer bottom lip (landmark 57)
2. Compare against threshold

**Thresholds**:
```python
MOUTH_OPEN_THRESHOLD = 23  # Pixels
```

**Detection Logic**:
```python
if lip_distance > 23px:
    return "Mouth Open"  # Talking detected
else:
    return "Mouth Closed"
```

**Violations**:
- `talking` - Mouth movement detected (potential cheating via communication)

---

### **6. Audio/Voice Detection**
**Algorithm**: Multi-parameter voice classification
**Method**:
1. **Amplitude Analysis**: Check if sound level is in human voice range
2. **Zero-Crossing Rate (ZCR)**: Measure frequency of signal polarity changes
3. **Amplitude Variation**: Distinguish speech from static noise

**Thresholds**:
```python
AUDIO_AMPLITUDE_MIN = 1000           # Minimum for speech
AUDIO_AMPLITUDE_MAX = 20000          # Maximum for speech
AUDIO_ZCR_MIN = 0.05                 # ZCR minimum for voice
AUDIO_ZCR_MAX = 0.35                 # ZCR maximum for voice
AUDIO_AMPLITUDE_VARIATION_MIN = 500  # Variation threshold
```

**Detection Logic**:
```python
is_voice = (
    1000 < amplitude < 20000 AND
    0.05 < zcr < 0.35 AND
    variation > 500
)
```

**Violations**:
- `Voice detected` - Human speech detected (potential communication)

---

### **7. Camera Coverage Detection**
**Algorithm**: Frame brightness analysis
**Method**:
1. Downscale frame to 32x24 pixels for fast processing
2. Calculate average luminance using RGB weights
3. Compare against darkness threshold

**Thresholds**:
```javascript
BRIGHTNESS_THRESHOLD = 35  // Luminance value
darkGraceMs = 5000         // 5 seconds grace period
```

**Detection Logic**:
```javascript
brightness = 0.2126*R + 0.7152*G + 0.0722*B
if brightness < 35 for 5+ seconds:
    trigger "camera_dark_or_covered"
```

**Violations**:
- `camera_dark_or_covered` - Camera blocked/covered

---

### **8. Object Detection (YOLO)**
**Algorithm**: YOLOv3-tiny real-time object detection
**Detectable Objects**:
- Cell phones / Mobile devices
- Books / Study materials
- Additional people
- Prohibited items

**Thresholds**:
```python
CONFIDENCE_THRESHOLD = 0.5  # 50% confidence
NMS_THRESHOLD = 0.4         # Non-max suppression
```

**Violations**:
- `phone_detected` - Mobile device visible
- `prohibited_object` - Books, notes, unauthorized materials

---

### **9. Browser Security (JavaScript)**
**Module**: `browserLock.js`
**Blocks**:
- Tab switching / Alt+Tab
- Fullscreen exit / Escape key
- Developer tools (F12, Ctrl+Shift+I, Ctrl+Shift+C)
- Copy/paste (Ctrl+C, Ctrl+V, Ctrl+X)
- Right-click context menu
- Print screen attempts
- Window blur/focus loss

**Thresholds**:
```javascript
maxWarnings = 5                    // Maximum violations before auto-submit
contextSwitchCooldownMs = 1500     // Debounce tab switches
fullscreenViolationCooldownMs = 2000  // Debounce fullscreen exits
```

**Detection Methods**:
- `fullscreenchange` event listener
- `visibilitychange` event listener
- `window.blur` event listener
- Keyboard event interception
- DevTools dimension detection (width/height discrepancy)

**Violations** (filtered from AI review):
- `browser_lock_violation` - Any browser security breach
- `browser_lock_max_warnings` - Maximum violations reached

---

## **🎬 RENDERING & AI PIPELINE**

### **Architecture: Dual-Loop Decoupled System**

#### **Loop 1: Rendering (30 FPS - UI Smoothness)**
**Technology**: `requestAnimationFrame` (RAF)
**Purpose**: Smooth video preview for student
**Resolution**: 640×360 (downscaled for performance)

```javascript
const render = () => {
    if (video.readyState >= 2) {
        ctx.drawImage(video, 0, 0, 640, 360);
    }
    requestAnimationFrame(render);  // ~30 FPS browser-optimized
};
```

**Optimizations**:
- `useRef` for canvas/context (no React re-renders)
- `alpha: false` - No transparency needed
- `desynchronized: true` - Async rendering
- Fixed container sizes (no layout thrashing)

#### **Loop 2: AI Processing (4 FPS - Detection Accuracy)**
**Technology**: `setInterval` at 250ms
**Purpose**: Send frames to backend for ML analysis
**Resolution**: 240-400px adaptive (accuracy + bandwidth)

```javascript
setInterval(async () => {
    const aiFrame = captureAIFrame();  // Separate capture
    await fetch('/proctor', {
        body: JSON.stringify({ imageDataUrl: aiFrame })
    });
}, 250);  // Exactly 4 FPS
```

**Optimizations**:
- **Backpressure**: Only 1 in-flight request at a time
- **Adaptive quality**: Reduce resolution if network slow
- **Throttling**: Skip frames if processing delayed
- **Separate canvas**: AI capture doesn't block rendering

**Why 4 FPS for AI?**
- Balances accuracy vs bandwidth
- Sufficient for facial landmark detection
- Most violations persist >250ms
- Reduces backend CPU/GPU load
- Scales to multiple students

---

## **📡 REAL-TIME COMMUNICATION**

### **Socket.IO Events**

#### **Student → Backend**
```javascript
socket.emit('student-video-frame', {
    examId, userId, frame, timestamp
});  // Every 500ms for lecturer live feed

socket.emit('join_student', { examId, userId });
socket.emit('join_exam', { examId });
```

#### **Backend → Student**
```javascript
socket.on('student_paused', (data) => {
    // Proctor decision: pause/terminate/allow
    if (data.status === 'terminated') {
        alert('Removed from exam');
        window.location.reload();
    }
});
```

#### **Backend → Lecturer**
```javascript
socket.emit('student-joined', { examId, userId, name });
socket.emit('gaze_aversion', { direction, message });
socket.emit('head_pose', { pose, message });
socket.emit('talking', { status, message });
```

---

## **⚙️ CONFIGURATION & POLICIES**

### **Client-Side Policy**
```javascript
proctorPolicy = {
    faceMissingGraceMs: 4000,      // 4s grace before violation
    darkGraceMs: 5000,              // 5s grace for dark camera
    violationCooldownMs: 7000       // 7s cooldown between same violations
}
```

### **Adaptive Encoding**
```javascript
encodeQuality: 0.4 - 0.8    // JPEG quality (adaptive)
encodeWidth: 220 - 400px     // Frame width (adaptive)

// Rules:
if uploadMs > 1200: decrease quality
if uploadMs < 450: increase quality
```

### **Browser Lock Policy**
```javascript
maxWarnings: 5               // Auto-submit after 5 violations
securityModalOpen: true      // Show warning modal
autoSubmitRequested: true    // Force submission at max warnings
```

---

## **🎯 VIOLATION SEVERITY LEVELS**

```javascript
severity_map = {
    'face_missing': 'critical',
    'multiple_faces': 'critical',
    'phone_detected': 'critical',
    'camera_dark': 'high',
    'gaze_aversion': 'low',
    'head_pose': 'medium',
    'talking': 'medium',
    'tab_switch': 'medium',
    'fullscreen_exit': 'high',
    'dev_tools': 'critical'
}
```

**Status Classification**:
- `critical`: 5+ violations OR risk_score ≥ 80
- `suspicious`: 3+ violations OR risk_score ≥ 60
- `warning`: 1+ violations OR risk_score ≥ 40
- `normal`: 0 violations

---

## **🔄 DATA FLOW**

```
Student Browser
    ↓ getUserMedia()
Video Stream (30 FPS rendering)
    ↓ RAF Loop
Canvas Rendering (smooth UI)
    ↓ Every 250ms
AI Frame Capture
    ↓ toDataURL('image/jpeg', 0.7)
Base64 JPEG
    ↓ fetch('/proctor')
Backend Flask Server
    ↓ base64 decode
NumPy Array
    ↓ OpenCV processing
Dlib Face Detection
    ↓ 68 landmarks
ML Analysis (gaze, pose, mouth, blink)
    ↓ Threshold checks
Violation Detection
    ↓ Socket.IO emit
Lecturer Dashboard (real-time)
    ↓ MongoDB write
Evidence Storage
```

---

## **💾 DATA STORAGE**

### **MongoDB Collections**

**proctor_events**:
```javascript
{
    examId, userId, eventType,
    severity, timestamp,
    details: { snapshot, metrics }
}
```

**proctoring_logs**:
```javascript
{
    exam_id, user_id, violation_type,
    severity, timestamp, details
}
```

**evidence_uploads** (GridFS):
```javascript
{
    examId, userId, evidenceType,
    violationType, violationScore,
    timestamp, fileUrl
}
```

**proctor_decisions**:
```javascript
{
    exam_id, user_id, status,
    reason, timestamp, actor_id
}
```

---

## **🚀 PERFORMANCE METRICS**

- **Rendering**: ~30 FPS (RAF-based)
- **AI Processing**: Exactly 4 FPS (250ms interval)
- **Network**: 2-5 requests/second (adaptive)
- **Frame Size**: 220-400px (adaptive)
- **JPEG Quality**: 40-80% (adaptive)
- **Camera Lag**: <100ms (RAF scheduling)
- **AI Latency**: 200-800ms (network + processing)
- **Memory**: ~50MB per student (stable)

---

## **🔧 KEY CODE LOCATIONS**

### **Backend**
- `server/proctoring_module.py` - All ML algorithms and thresholds
- `server/app.py` - Flask routes, Socket.IO handlers, violation logic
- `server/face_engine.py` - Face recognition and identity verification
- `server/requirements-ml.txt` - ML dependencies

### **Frontend**
- `client/src/App.tsx` - Main exam screen, proctoring loops (lines 2700-3900)
- `client/src/browserLock.js` - Browser security module
- `client/src/components/LiveMonitoringDashboard.tsx` - Lecturer dashboard

### **Configuration Files**
- `server/proctoring_module.py` (lines 1-90) - All detection thresholds
- `client/src/App.tsx` (lines 2740-2755) - Client-side policies

---

## **📊 68-POINT FACIAL LANDMARKS MAP**

```
Landmarks:
0-16:  Jawline
17-21: Left eyebrow
22-26: Right eyebrow
27-35: Nose bridge and tip
36-41: Left eye (outer→inner, top→bottom)
42-47: Right eye (outer→inner, top→bottom)
48-60: Outer lip
61-67: Inner lip
```

**Used for**:
- Eyes (36-47): Gaze detection, blink detection
- Nose (27-35): Head pose estimation
- Mouth (48-67): Talking detection
- All points: 3D pose calculation

---

This is a **production-grade proctoring system** with multi-layered security, real-time ML detection, and smooth UX! 🎯

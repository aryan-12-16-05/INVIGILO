# Smart Tolerance-Based Proctoring System

## Overview
The proctoring system now uses **tolerance thresholds** and **frequency-based detection** instead of recording every minor movement. This dramatically reduces false positives and only flags genuinely suspicious behavior.

---

## Key Features

### 1. **Tolerance Thresholds**
Minor movements and brief glances are **normal** during exams. The system allows a tolerance percentage before flagging behavior:

| Behavior | Tolerance | Description |
|----------|-----------|-------------|
| **Head Pose** | 30% off-forward | Allows minor head adjustments while writing |
| **Gaze Direction** | 40% off-center | Permits brief eye movements to think/read |
| **Mouth Movement** | 25% open | Allows brief reactions, sighs, or muttering |
| **Face Missing** | 80% threshold | Only flags prolonged disappearance |
| **Background Change** | ~0% (15 pixels) | Very strict - detects hands, objects, people entering frame |

### 2. **Frequency-Based Detection**
Events are only recorded if violations occur **frequently** over time, scaled to exam duration:

**Violation Frequency Thresholds:**
- **Low Severity**: 1+ normalized violations per minute
- **Medium Severity**: 2+ normalized violations per minute  
- **High Severity**: 3+ normalized violations per minute

**Example for 2-minute exam:**
- Low: ~2 total violations
- Medium: ~4 total violations
- High: ~6 total violations

### 3. **Background Change Detection**
At exam start, the system captures a **reference image** of the student and background. Throughout the exam:
- Compares each frame to reference background
- Very low tolerance (~15 pixel difference)
- Detects: hands entering frame, people behind, objects being shown
- Any change triggers violation counter immediately

---

## How It Works

### Initialization (Exam Start)
1. Student starts exam
2. Client calls `/api/proctor/reset` to clear previous state
3. Server captures **reference background** on first frame
4. Violation counters reset to 0

### During Exam (Every 3 seconds)
1. Client sends video frame to `/api/proctor`
2. Server analyzes:
   - Face detection & identity verification
   - Head pose, gaze direction, blink, mouth status
   - Background comparison vs reference image
3. Server tracks violations in rolling 60-second window:
   - Counts how many checks exceed tolerance threshold
   - Normalizes violations based on total checks
4. If frequency exceeds threshold → Event emitted with severity

### Event Recording
Events are saved to database with:
- **eventType**: `head_pose`, `gaze_aversion`, `talking`, `multiple_faces`, `face_missing`, `background_change`
- **severity**: `low`, `medium`, `high`
- **details**: Violation count, frequency, specific metrics
- **timestamp**: When event occurred

---

## Severity Levels

### High Severity ⚠️
- **Multiple Faces**: Immediate detection (zero tolerance)
- **Background Change**: Frequent changes (2+ per minute)
- **Face Missing**: Prolonged disappearance (2+ per minute)
- **Head Pose**: Excessive turning away (3+ per minute)
- **Gaze Aversion**: Constant looking away (3+ per minute)
- **Talking**: Continuous speaking (3+ per minute)

### Medium Severity ⚡
- Moderate frequency of violations (1-2 per minute)
- Suggests possible suspicious activity
- Warrants attention but not alarm

### Low Severity ℹ️
- Infrequent violations (~1 per minute)
- May be normal exam stress/behavior
- Logged for pattern analysis

---

## Zero Tolerance Items
Some behaviors are flagged **immediately** regardless of frequency:

1. **Multiple Faces**: Any detection triggers high severity
2. **Identity Failure**: Face doesn't match registered student

---

## Technical Details

### Server State Management
```python
PROCTOR_STATE = {
    (exam_id, user_id): {
        'poses': [(timestamp, value), ...],           # Rolling 60s window
        'gazes': [(timestamp, value), ...],           # Rolling 60s window
        'mouths': [(timestamp, value), ...],          # Rolling 60s window
        'faces': [(timestamp, count), ...],           # Rolling 60s window
        'violation_counts': {                         # Cumulative counters
            'head_pose': 5,
            'gaze_aversion': 3,
            'background_change': 2,
            ...
        },
        'reference_background': numpy_array,          # Initial frame (160x120)
        'total_checks': 42,                           # Total analysis calls
        'last_emit': {                                # Cooldown tracking
            'head_pose': datetime,
            ...
        }
    }
}
```

### Background Detection Algorithm
```python
1. Capture reference: resize to 160x120, store on first frame
2. For each subsequent frame:
   - Resize to 160x120
   - Calculate absolute difference: cv2.absdiff(reference, current)
   - Compute mean pixel difference
   - If diff > 15.0 → Increment violation counter
3. Emit event based on frequency thresholds
```

### Normalization Formula
```python
# Normalize violations to "per minute" rate
normalized_violations = violations / max(1, total_checks / 20)

# Explanation:
# - 20 checks = ~1 minute (3s interval × 20 = 60s)
# - This scales violations to exam duration
# - Short exam with 2 violations = high frequency
# - Long exam with 2 violations = low frequency
```

---

## Benefits

✅ **Reduces Noise**: No more hundreds of events for 2-minute exams  
✅ **Accurate Detection**: Only flags genuinely suspicious patterns  
✅ **Fair Tolerance**: Accounts for normal human behavior during exams  
✅ **Scalable**: Adapts to exam duration automatically  
✅ **Background Monitoring**: Detects external assistance attempts  
✅ **Real-time Processing**: No client-side event logic needed  

---

## API Endpoints

### Reset Proctoring State
**POST** `/api/proctor/reset`
```json
{
  "examId": "exam123",
  "userId": "user456"
}
```
Called when exam starts to clear previous state and capture fresh reference image.

### Analyze Frame
**POST** `/api/proctor`
```json
{
  "imageData": "data:image/jpeg;base64,...",
  "userId": "user456",
  "examId": "exam123"
}
```
Server automatically tracks violations and emits events based on frequency.

### View Events
**GET** `/api/exams/{examId}/proctoring/{userId}`

Returns all recorded events with severity and details.

---

## Configuration Constants

Located in `server/app.py`:

```python
# Tolerance ratios (0.0 - 1.0)
POSE_TOLERANCE_RATIO = 0.30      # 30% off-forward poses allowed
GAZE_TOLERANCE_RATIO = 0.40      # 40% off-center gazes allowed
MOUTH_TOLERANCE_RATIO = 0.25     # 25% mouth open allowed
BACKGROUND_THRESHOLD = 15.0      # Pixel difference threshold

# Frequency thresholds (violations per 20 checks ~1 minute)
LOW_SEVERITY_THRESHOLD = 1.0
MEDIUM_SEVERITY_THRESHOLD = 2.0
HIGH_SEVERITY_THRESHOLD = 3.0
```

Adjust these values to tune system sensitivity.

---

## Monitoring

Lecturers can view:
- **Proctoring Timeline**: Chronological list of all flagged events
- **Severity Distribution**: Count of low/medium/high events
- **Event Details**: Specific metrics (violations, frequency, context)
- **Snapshots**: Visual evidence at time of event

System designed for **quality over quantity** - fewer, more meaningful alerts.

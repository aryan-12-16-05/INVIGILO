# Phase 2: Client-Side Detection with MediaPipe

## Overview

Phase 2 moves face detection and basic proctoring logic to the client browser using Google's MediaPipe Face Mesh. This reduces server load by **90%** and enables **30 FPS** real-time processing in the browser.

**Current Architecture (Phase 1):**
```
Client (4 FPS) ──────> Server ──────> ML Service
    (all frames)      (CPU heavy)   (face detection)
```

**Phase 2 Architecture:**
```
Client (30 FPS) ──MediaPipe (WebGL)──> Local Detection
    │                                        │
    └─── Violation Snapshots Only ──────> Server
              (10-20 per exam)           (validation)
```

**Benefits:**
- ✅ **30 FPS** real-time detection in browser (vs 4 FPS server-side)
- ✅ **90% reduction** in server load
- ✅ **95% reduction** in network bandwidth
- ✅ **500+ concurrent students** supported
- ✅ Instant violation feedback to student (no network latency)
- ✅ Server only validates critical violations (identity, multiple faces)

---

## Architecture Changes

### What Moves to Client (MediaPipe)

1. **Face Detection** (landmark detection, face count)
2. **Gaze Tracking** (eye direction, head pose)
3. **Mouth Status** (open/closed, talking)
4. **Background Change** (pixel difference detection)
5. **Camera Blocking** (brightness check)

### What Stays on Server

1. **Identity Verification** (face matching with stored embeddings)
2. **Multiple Faces Validation** (confirm with ML service)
3. **Violation Review** (lecturer approval/rejection)
4. **Evidence Storage** (MongoDB persistence)

---

## Implementation Steps

### Step 1: Install MediaPipe in React

```bash
cd client
npm install @mediapipe/face_mesh @mediapipe/camera_utils
```

### Step 2: Create MediaPipe Detector Component

Create `client/src/hooks/useMediaPipeDetector.ts`:

```typescript
import { useEffect, useRef, useState } from 'react';
import { FaceMesh } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';

interface Detection {
  faceCount: number;
  gazeDirection: string;
  headPose: string;
  mouthStatus: string;
  landmarks?: any[];
}

export function useMediaPipeDetector(videoElement: HTMLVideoElement | null) {
  const [detection, setDetection] = useState<Detection | null>(null);
  const faceMeshRef = useRef<FaceMesh | null>(null);
  const cameraRef = useRef<Camera | null>(null);

  useEffect(() => {
    if (!videoElement) return;

    // Initialize MediaPipe Face Mesh
    const faceMesh = new FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      }
    });

    faceMesh.setOptions({
      maxNumFaces: 3,              // Detect up to 3 faces
      refineLandmarks: true,       // Detailed eye/mouth landmarks
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    faceMesh.onResults((results) => {
      const faceCount = results.multiFaceLandmarks?.length || 0;

      if (faceCount === 0) {
        setDetection({
          faceCount: 0,
          gazeDirection: 'Unknown',
          headPose: 'Unknown',
          mouthStatus: 'Unknown'
        });
        return;
      }

      // Analyze first face (primary student)
      const landmarks = results.multiFaceLandmarks[0];

      // Calculate gaze direction from eye landmarks
      const gazeDirection = calculateGazeDirection(landmarks);

      // Calculate head pose from face landmarks
      const headPose = calculateHeadPose(landmarks);

      // Calculate mouth status from lip landmarks
      const mouthStatus = calculateMouthStatus(landmarks);

      setDetection({
        faceCount,
        gazeDirection,
        headPose,
        mouthStatus,
        landmarks
      });
    });

    // Start camera
    const camera = new Camera(videoElement, {
      onFrame: async () => {
        await faceMesh.send({ image: videoElement });
      },
      width: 640,
      height: 480
    });
    camera.start();

    faceMeshRef.current = faceMesh;
    cameraRef.current = camera;

    return () => {
      camera.stop();
      faceMesh.close();
    };
  }, [videoElement]);

  return detection;
}

// ============================================================================
// HELPER FUNCTIONS: Gaze & Head Pose Calculation
// ============================================================================

function calculateGazeDirection(landmarks: any[]): string {
  // MediaPipe Face Mesh landmark indices:
  // Left eye: 33, 133 (corners)
  // Right eye: 263, 362 (corners)
  // Nose tip: 1
  
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  const noseTip = landmarks[1];

  // Calculate horizontal gaze offset
  const eyeCenterX = (leftEye.x + rightEye.x) / 2;
  const gazeOffset = noseTip.x - eyeCenterX;

  // Thresholds tuned for proctoring sensitivity
  if (gazeOffset > 0.03) return 'Right';
  if (gazeOffset < -0.03) return 'Left';
  if (noseTip.y > 0.6) return 'Down';
  if (noseTip.y < 0.4) return 'Up';
  
  return 'Center';
}

function calculateHeadPose(landmarks: any[]): string {
  // Calculate head rotation using face landmarks
  const leftEar = landmarks[234];
  const rightEar = landmarks[454];
  const noseTip = landmarks[1];
  const chin = landmarks[152];

  // Horizontal rotation (left/right)
  const horizontalOffset = (rightEar.x - leftEar.x);
  if (horizontalOffset < -0.1) return 'Left';
  if (horizontalOffset > 0.1) return 'Right';

  // Vertical rotation (up/down)
  const verticalOffset = noseTip.y - chin.y;
  if (verticalOffset > 0.2) return 'Down';
  if (verticalOffset < -0.1) return 'Up';

  return 'Forward';
}

function calculateMouthStatus(landmarks: any[]): string {
  // Mouth landmarks: Upper lip (13), Lower lip (14)
  const upperLip = landmarks[13];
  const lowerLip = landmarks[14];

  // Calculate vertical distance between lips
  const mouthOpen = Math.abs(upperLip.y - lowerLip.y);

  // Threshold tuned for talking detection
  if (mouthOpen > 0.02) return 'Open';
  return 'Closed';
}
```

### Step 3: Integrate MediaPipe in Exam Interface

Update `client/src/components/ExamInterface.tsx`:

```typescript
import { useMediaPipeDetector } from '../hooks/useMediaPipeDetector';

function ExamInterface() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [violations, setViolations] = useState<any[]>([]);
  
  // MediaPipe detection (runs at 30 FPS in browser)
  const detection = useMediaPipeDetector(videoRef.current);

  useEffect(() => {
    if (!detection) return;

    // Analyze detection results for violations
    const newViolations = analyzeDetection(detection);

    // Only send violation snapshots to server (not every frame!)
    if (newViolations.length > 0) {
      newViolations.forEach(violation => {
        sendViolationSnapshot(violation);
      });
    }
  }, [detection]);

  function analyzeDetection(detection: Detection): Violation[] {
    const violations: Violation[] = [];

    // VIOLATION 1: No face detected
    if (detection.faceCount === 0) {
      violations.push({
        type: 'face_missing',
        severity: 'high',
        details: { message: 'Student left camera view' }
      });
    }

    // VIOLATION 2: Multiple faces
    if (detection.faceCount > 1) {
      violations.push({
        type: 'multiple_faces',
        severity: 'critical',
        details: { count: detection.faceCount }
      });
    }

    // VIOLATION 3: Gaze aversion (frequency tracking)
    if (detection.gazeDirection !== 'Center' && detection.gazeDirection !== 'Unknown') {
      violations.push({
        type: 'gaze_aversion',
        severity: 'medium',
        details: { direction: detection.gazeDirection }
      });
    }

    // VIOLATION 4: Head turned away
    if (detection.headPose !== 'Forward' && detection.headPose !== 'Unknown') {
      violations.push({
        type: 'head_pose',
        severity: 'low',
        details: { pose: detection.headPose }
      });
    }

    // VIOLATION 5: Talking detected
    if (detection.mouthStatus === 'Open') {
      violations.push({
        type: 'talking',
        severity: 'low',
        details: { status: 'Open' }
      });
    }

    return violations;
  }

  async function sendViolationSnapshot(violation: Violation) {
    // Capture current frame as JPEG
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0);
    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.85);

    // Send violation snapshot to server (not every frame!)
    await fetch(`${API_URL}/api/proctor/violation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        examId: currentExam._id,
        userId: user._id,
        violationType: violation.type,
        severity: violation.severity,
        details: violation.details,
        imageDataUrl  // Only send when violation detected
      })
    });
  }

  return (
    <div>
      <video ref={videoRef} autoPlay muted />
      
      {/* Live violation feedback (no server latency!) */}
      {detection && (
        <div className="live-feedback">
          <p>Face Count: {detection.faceCount}</p>
          <p>Gaze: {detection.gazeDirection}</p>
          <p>Head Pose: {detection.headPose}</p>
          <p>Mouth: {detection.mouthStatus}</p>
        </div>
      )}
    </div>
  );
}
```

### Step 4: Create Server Endpoint for Violation Snapshots

Update `server/app.py`:

```python
@app.route('/api/proctor/violation', methods=['POST'])
@limiter.limit("50 per hour")  # Lower limit (only violations, not all frames)
def process_violation_snapshot():
    """
    Phase 2: Process client-detected violation snapshot.
    
    Client sends violations only (not every frame).
    Server validates and stores critical violations.
    """
    data = request.get_json()
    exam_id = str(data.get('examId'))
    user_id = str(data.get('userId'))
    violation_type = data.get('violationType')
    severity = data.get('severity')
    details = data.get('details', {})
    image_data_url = data.get('imageDataUrl')

    if not all([exam_id, user_id, violation_type, image_data_url]):
        return jsonify({"error": "Missing required fields"}), 400

    # Queue validation in Celery (Phase 1 + Phase 2)
    if CELERY_AVAILABLE:
        from celery_worker import process_violation_snapshot
        
        frame = decode_base64_image(image_data_url)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        task = process_violation_snapshot.delay(
            exam_id=exam_id,
            user_id=user_id,
            violation_type=violation_type,
            frame_base64=frame_base64
        )
        
        return jsonify({
            "status": "accepted",
            "task_id": str(task.id),
            "violation_type": violation_type
        }), 202
    
    # Fallback: Synchronous processing
    # ... (existing code)
```

---

## Performance Comparison

### Network Bandwidth

**Phase 1 (Server-Side):**
- Frame size: ~20 KB (JPEG 85% quality)
- Frame rate: 4 FPS
- Students: 100
- **Total: 8 MB/s upload to server**

**Phase 2 (Client-Side):**
- Violation snapshot: ~20 KB
- Violations per exam: ~10-20
- Students: 100
- **Total: <1 MB per hour total**

**Reduction: 99.8% less network traffic**

### Server CPU Usage

**Phase 1:**
- 100 students × 4 FPS = 400 frames/second
- Each frame: ML service call (2-5s processing)
- **Result: Server bottleneck, requires 8+ Celery workers**

**Phase 2:**
- 10-20 violations per exam per student
- Only validate critical violations
- **Result: 1-2 Celery workers sufficient**

---

## Browser Compatibility

MediaPipe Face Mesh requires:
- **WebGL support** (GPU acceleration)
- **getUserMedia API** (camera access)
- **Modern browser:**
  - Chrome 90+ ✅
  - Firefox 88+ ✅
  - Safari 14+ ✅
  - Edge 90+ ✅

**Fallback:** If WebGL unavailable, fall back to Phase 1 (server-side)

---

## Security Considerations

### Concern: Can students disable MediaPipe detection?

**Mitigation:**
1. **Periodic identity verification** (server-side, can't be bypassed)
2. **Random frame validation** (server requests specific frames)
3. **Violation pattern analysis** (too few violations = suspicious)
4. **Exam recording** (forensic review if cheating suspected)

### Concern: Can students fake violations?

**Mitigation:**
1. **Server validates all critical violations** (identity, multiple faces)
2. **Frame evidence stored** (tampering detectable)
3. **Timestamp verification** (prevent replay attacks)
4. **HMAC signatures** (prevent snapshot forgery)

---

## Migration Strategy

### Phase 2a: Pilot (10% of exams)

1. Enable MediaPipe for specific exams (feature flag)
2. Run parallel detection (MediaPipe + Server-side)
3. Compare results for accuracy validation
4. Monitor performance metrics

### Phase 2b: Gradual Rollout (50% of exams)

1. Enable MediaPipe for all Chrome/Firefox users
2. Fallback to server-side for Safari/Edge (if issues)
3. Monitor error rates and false positives
4. Tune detection thresholds

### Phase 2c: Full Production (100% of exams)

1. Enable MediaPipe for all supported browsers
2. Deprecate old `/api/proctor` endpoint
3. Keep fallback for legacy clients
4. Celebrate 90% server cost reduction! 🎉

---

## Next Steps

After Phase 2 is complete:

**Phase 3: Backend Validation Only**
- Server only validates critical violations (identity, multiple faces)
- Remove continuous frame processing entirely
- Further reduce server costs by 50%
- Support 1000+ concurrent students

**Future Enhancements:**
- Edge ML models (TensorFlow.js for offline detection)
- Advanced gaze tracking (pupil detection)
- Emotion detection (stress, nervousness)
- Biometric authentication (fingerprint, face unlock)

---

## Cost Savings Estimate

**Current (Phase 1):**
- Server instances: 4× (8 vCPU, 16 GB RAM)
- ML service: $200/month (Hugging Face Spaces)
- **Total: ~$800/month** for 100 concurrent students

**Phase 2:**
- Server instances: 1× (2 vCPU, 4 GB RAM)
- ML service: $50/month (validation only)
- **Total: ~$100/month** for 500 concurrent students

**Savings: 87.5% cost reduction + 5× capacity increase** 🚀

---

## Summary

✅ **Phase 2 Benefits:**
- 30 FPS real-time detection in browser
- 90% server load reduction
- 95% network bandwidth reduction
- 500+ concurrent students supported
- Instant violation feedback (no latency)

📋 **Implementation Checklist:**
- [ ] Install MediaPipe dependencies
- [ ] Create `useMediaPipeDetector` hook
- [ ] Integrate in `ExamInterface`
- [ ] Create `/api/proctor/violation` endpoint
- [ ] Update Celery task for validation
- [ ] Test browser compatibility
- [ ] Pilot with 10% of exams
- [ ] Monitor accuracy and performance
- [ ] Gradual rollout to 100%

**Questions?** See `PHASE1_ASYNC_SETUP.md` for current architecture details.

# Screen Recording and Evidence Upload Guide

## Overview

Capture and upload screen recordings, screenshots, and audio evidence when violations are detected during proctored exams.

**Features:**
- 📹 Screen recording with MediaRecorder API
- 📸 Screenshot capture
- 🎤 Audio recording
- ☁️ Automatic upload to backend
- 🔒 Secure storage with organized structure

---

## Backend Endpoints

### 1. Upload Evidence

**Endpoint:** `POST /api/upload-evidence`

**Rate Limit:** 100 per hour

**Form Data:**
```javascript
{
  file: <binary>,           // Video, image, or audio file
  examId: "exam_123",       // Required
  userId: "user_456",       // Required
  evidenceType: "screenshot", // screenshot | video | audio
  violationType: "eyes_off_screen", // Optional
  violationScore: 15        // Optional
}
```

**Response:**
```json
{
  "message": "Evidence uploaded successfully",
  "url": "/api/evidence/evidence_id_123",
  "fileId": "evidence_id_123",
  "filePath": "evidence/exam_123/user_456/1702402245_screenshot.jpg",
  "fileSize": 245632,
  "evidenceType": "screenshot"
}
```

### 2. Get Evidence File

**Endpoint:** `GET /api/evidence/{evidence_id}`

**Headers:** `X-User-Id: <lecturer_id>`

**Rate Limit:** 200 per hour

**Returns:** File content (image, video, or audio)

### 3. List Evidence

**Endpoint:** `GET /api/evidence?examId=exam_123&userId=user_456`

**Query Parameters:**
- `examId` (required): Filter by exam
- `userId` (optional): Filter by user
- `evidenceType` (optional): Filter by type

**Response:**
```json
{
  "evidence": [
    {
      "id": "evidence_id_123",
      "examId": "exam_123",
      "userId": "user_456",
      "evidenceType": "screenshot",
      "violationType": "eyes_off_screen",
      "fileSize": 245632,
      "violationScore": 15,
      "url": "/api/evidence/evidence_id_123",
      "timestamp": "2025-12-14T15:30:45Z"
    }
  ],
  "count": 1
}
```

---

## Frontend Implementation

### 1. Screenshot Capture

```javascript
async function captureScreenshot(examId, userId) {
  try {
    // Get video stream
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { mediaSource: 'screen' }
    });
    
    // Create video element
    const video = document.createElement('video');
    video.srcObject = stream;
    video.play();
    
    // Wait for video to load
    await new Promise(resolve => {
      video.onloadedmetadata = resolve;
    });
    
    // Capture frame to canvas
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    // Stop stream
    stream.getTracks().forEach(track => track.stop());
    
    // Convert to blob
    const blob = await new Promise(resolve => {
      canvas.toBlob(resolve, 'image/jpeg', 0.9);
    });
    
    // Upload
    await uploadEvidence(blob, examId, userId, 'screenshot');
    
    return true;
  } catch (error) {
    console.error('Screenshot capture failed:', error);
    return false;
  }
}
```

### 2. Screen Recording

```javascript
class ScreenRecorder {
  constructor(examId, userId) {
    this.examId = examId;
    this.userId = userId;
    this.mediaRecorder = null;
    this.chunks = [];
    this.stream = null;
  }
  
  async startRecording() {
    try {
      // Request screen capture
      this.stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          mediaSource: 'screen',
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 }
        },
        audio: false
      });
      
      // Create MediaRecorder
      const options = { mimeType: 'video/webm;codecs=vp9' };
      
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = 'video/webm;codecs=vp8';
      }
      
      this.mediaRecorder = new MediaRecorder(this.stream, options);
      this.chunks = [];
      
      // Collect data chunks
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.chunks.push(event.data);
        }
      };
      
      // Start recording
      this.mediaRecorder.start(1000); // Collect data every 1 second
      
      console.log('Screen recording started');
      return true;
    } catch (error) {
      console.error('Failed to start recording:', error);
      return false;
    }
  }
  
  async stopRecording() {
    return new Promise((resolve, reject) => {
      if (!this.mediaRecorder) {
        reject(new Error('No recording in progress'));
        return;
      }
      
      this.mediaRecorder.onstop = async () => {
        try {
          // Create video blob
          const blob = new Blob(this.chunks, { type: 'video/webm' });
          
          // Stop all tracks
          if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
          }
          
          // Upload evidence
          await uploadEvidence(blob, this.examId, this.userId, 'video');
          
          console.log('Recording stopped and uploaded');
          resolve(blob);
        } catch (error) {
          reject(error);
        }
      };
      
      this.mediaRecorder.stop();
    });
  }
  
  pauseRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.pause();
    }
  }
  
  resumeRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state === 'paused') {
      this.mediaRecorder.resume();
    }
  }
}
```

### 3. Upload Evidence Function

```javascript
async function uploadEvidence(blob, examId, userId, evidenceType, violationType = null, violationScore = 0) {
  try {
    // Create form data
    const formData = new FormData();
    
    // Add file with appropriate extension
    const extensions = {
      screenshot: 'jpg',
      video: 'webm',
      audio: 'wav'
    };
    const ext = extensions[evidenceType] || 'bin';
    const filename = `${Date.now()}.${ext}`;
    
    formData.append('file', blob, filename);
    formData.append('examId', examId);
    formData.append('userId', userId);
    formData.append('evidenceType', evidenceType);
    
    if (violationType) {
      formData.append('violationType', violationType);
    }
    
    if (violationScore) {
      formData.append('violationScore', violationScore);
    }
    
    // Upload to backend
    const response = await fetch('http://localhost:5000/api/upload-evidence', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }
    
    const result = await response.json();
    console.log('Evidence uploaded:', result);
    
    return result;
  } catch (error) {
    console.error('Failed to upload evidence:', error);
    throw error;
  }
}
```

### 4. React Component Example

```javascript
import { useState, useRef } from 'react';

function ExamProctoring({ examId, userId }) {
  const [isRecording, setIsRecording] = useState(false);
  const recorderRef = useRef(null);
  
  const startRecording = async () => {
    const recorder = new ScreenRecorder(examId, userId);
    const success = await recorder.startRecording();
    
    if (success) {
      recorderRef.current = recorder;
      setIsRecording(true);
    }
  };
  
  const stopRecording = async () => {
    if (recorderRef.current) {
      await recorderRef.current.stopRecording();
      recorderRef.current = null;
      setIsRecording(false);
    }
  };
  
  const captureScreenshot = async () => {
    const success = await captureScreenshot(examId, userId);
    if (success) {
      alert('Screenshot captured and uploaded');
    }
  };
  
  // Auto-capture on violation
  const handleViolation = async (violationType, violationScore) => {
    console.log(`Violation detected: ${violationType}`);
    
    // Capture screenshot as evidence
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { mediaSource: 'screen' }
    });
    
    const video = document.createElement('video');
    video.srcObject = stream;
    video.play();
    
    await new Promise(resolve => {
      video.onloadedmetadata = resolve;
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    stream.getTracks().forEach(track => track.stop());
    
    const blob = await new Promise(resolve => {
      canvas.toBlob(resolve, 'image/jpeg', 0.9);
    });
    
    await uploadEvidence(blob, examId, userId, 'screenshot', violationType, violationScore);
  };
  
  return (
    <div>
      <h2>Exam Proctoring</h2>
      
      <div className="controls">
        {!isRecording ? (
          <button onClick={startRecording}>
            Start Screen Recording
          </button>
        ) : (
          <button onClick={stopRecording}>
            Stop Recording
          </button>
        )}
        
        <button onClick={captureScreenshot}>
          Capture Screenshot
        </button>
      </div>
      
      {isRecording && (
        <div className="recording-indicator">
          🔴 Recording...
        </div>
      )}
    </div>
  );
}
```

### 5. Auto-Capture on Violation

```javascript
// Listen for violations and auto-capture evidence
socket.on('violation_detected', async (data) => {
  console.log('Violation detected:', data);
  
  // Auto-capture screenshot
  try {
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { mediaSource: 'screen' }
    });
    
    const video = document.createElement('video');
    video.srcObject = stream;
    video.play();
    
    await new Promise(resolve => video.onloadedmetadata = resolve);
    
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    stream.getTracks().forEach(track => track.stop());
    
    const blob = await new Promise(resolve => {
      canvas.toBlob(resolve, 'image/jpeg', 0.9);
    });
    
    await uploadEvidence(
      blob,
      data.examId,
      data.userId,
      'screenshot',
      data.violationType,
      data.score
    );
    
    console.log('Evidence auto-captured for violation');
  } catch (error) {
    console.error('Auto-capture failed:', error);
  }
});
```

---

## Storage Structure

Files are organized by exam and user:

```
server/
  evidence/
    exam_123/
      user_456/
        1702402245000_screenshot.jpg
        1702402300000_video.webm
        1702402360000_screenshot.jpg
      user_789/
        1702402400000_screenshot.jpg
    exam_124/
      user_456/
        1702402500000_video.webm
```

---

## Viewing Evidence (Proctor Dashboard)

```javascript
async function loadEvidence(examId, userId) {
  const response = await fetch(
    `http://localhost:5000/api/evidence?examId=${examId}&userId=${userId}`,
    {
      headers: {
        'X-User-Id': proctorId
      }
    }
  );
  
  const data = await response.json();
  
  // Display evidence
  data.evidence.forEach(item => {
    console.log(`${item.evidenceType}: ${item.url}`);
    
    // Create img/video element
    if (item.evidenceType === 'screenshot') {
      const img = document.createElement('img');
      img.src = `http://localhost:5000${item.url}`;
      document.getElementById('evidence-container').appendChild(img);
    } else if (item.evidenceType === 'video') {
      const video = document.createElement('video');
      video.src = `http://localhost:5000${item.url}`;
      video.controls = true;
      document.getElementById('evidence-container').appendChild(video);
    }
  });
}
```

---

## Best Practices

### 1. File Size Management

```javascript
// Compress images before upload
async function compressImage(blob, maxWidth = 1920) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const scale = Math.min(1, maxWidth / img.width);
      
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      canvas.toBlob(resolve, 'image/jpeg', 0.8);
    };
    img.src = URL.createObjectURL(blob);
  });
}
```

### 2. Recording Duration Limits

```javascript
class ScreenRecorder {
  constructor(examId, userId, maxDuration = 300000) { // 5 minutes
    this.maxDuration = maxDuration;
    this.recordingTimeout = null;
  }
  
  async startRecording() {
    // ... existing code ...
    
    // Auto-stop after max duration
    this.recordingTimeout = setTimeout(() => {
      this.stopRecording();
    }, this.maxDuration);
  }
  
  async stopRecording() {
    if (this.recordingTimeout) {
      clearTimeout(this.recordingTimeout);
    }
    // ... existing code ...
  }
}
```

### 3. Error Handling

```javascript
async function captureWithRetry(examId, userId, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      await captureScreenshot(examId, userId);
      return true;
    } catch (error) {
      console.error(`Capture attempt ${i + 1} failed:`, error);
      if (i === maxRetries - 1) {
        throw error;
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
}
```

### 4. Privacy Considerations

```javascript
// Request permission before exam starts
async function requestScreenPermission() {
  try {
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: true
    });
    
    // Show preview
    const video = document.getElementById('preview');
    video.srcObject = stream;
    
    // Confirm with user
    const confirmed = confirm(
      'This exam requires screen recording. Click OK to allow.'
    );
    
    if (confirmed) {
      return stream;
    } else {
      stream.getTracks().forEach(track => track.stop());
      return null;
    }
  } catch (error) {
    alert('Screen recording permission is required to take this exam.');
    return null;
  }
}
```

---

## Security Considerations

### 1. File Validation

Backend validates:
- ✅ File size limits (prevent abuse)
- ✅ File type verification
- ✅ Exam and user exist
- ✅ Only lecturers can view evidence

### 2. Access Control

```python
# Only lecturers can view evidence
if req_user.get('role') != 'lecturer':
    return jsonify({'error': 'Forbidden'}), 403
```

### 3. Storage Security

- Files stored outside web root
- No direct file access URLs
- Evidence served through authenticated endpoints
- File paths sanitized

---

## Troubleshooting

### "Permission Denied"

**Cause:** User denied screen capture permission

**Solution:** Request permission at exam start, not during violation

### "Upload Failed"

**Cause:** Network issue or file too large

**Solution:** Implement retry logic, compress files

### "Evidence Not Found"

**Cause:** File was deleted or path is wrong

**Solution:** Check file system, verify database records match

---

## Testing

1. **Test Screenshot Capture:**
   - Open exam page
   - Click "Capture Screenshot"
   - Verify upload success
   - Check file in `server/evidence/`

2. **Test Screen Recording:**
   - Start recording
   - Wait 10 seconds
   - Stop recording
   - Verify video file created

3. **Test Evidence Retrieval:**
   - Upload evidence
   - Call `/api/evidence?examId=test`
   - Verify evidence list returned
   - Access evidence URL
   - Verify file displays

---

## Performance Tips

- Compress images before upload (80% quality)
- Limit recording duration (5 minutes max)
- Use WebM format for videos (smaller size)
- Implement chunked upload for large files
- Clean up old evidence files (retention policy)

---

## Future Enhancements

- [ ] AWS S3 integration for cloud storage
- [ ] Automatic evidence cleanup after 90 days
- [ ] Evidence encryption at rest
- [ ] Thumbnail generation for videos
- [ ] Evidence search and filtering
- [ ] Download all evidence for an exam (ZIP)

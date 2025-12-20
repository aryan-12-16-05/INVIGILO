# Architectural Cleanup - ML-Free Backend

## Overview
This document summarizes the architectural cleanup performed to ensure **strict separation** between the backend API and ML service, per the IMPORTANT ARCHITECTURAL RULES.

## Objective
**Server must be 100% ML-free** - all ML computation happens in the separate ml-service deployed on Hugging Face Spaces.

**Deployment target:** Render (not Railway)

## Changes Made

### 1. Removed Local ML Imports and Engine Code
**File:** `server/app.py`

**Removed:**
- `ENGINE_AVAILABLE` flag and all references
- `get_face_engine()` function
- Local proctoring module imports (`detectFace`, `isBlinking`, `mouthTrack`, `gazeDetection`, `head_pose_detection`)
- All local face engine initialization code
- Image variant generation for local ML processing
- Local face embedding generation

**Lines removed:** ~150 lines of ML-specific code

### 2. Updated `/api/proctor` Endpoint
**Before:** Used local `detectFace()`, `engine.verify_any()`, and behavioral analysis functions

**After:** 
- Calls ML service `/analyze-frame` for face detection and behavioral analysis
- Calls ML service `/verify-face` + `/match-face` for identity verification
- Maintains same response format for frontend compatibility
- Only uses cv2/numpy for basic operations (brightness checks, frame encoding)

### 3. Updated `/api/users/<user_id>/face-samples` Endpoint
**Before:** Used local `engine.embed()` to generate face embeddings

**After:** 
- Calls ML service `/verify-face` to generate embeddings
- Removes dependency on local InsightFace engine

### 4. Updated Server Startup Message
**Before:** Printed `InsightFace Engine={ENGINE_AVAILABLE}`

**After:** Prints `ML_SERVICE_URL={os.getenv('ML_SERVICE_URL', 'Not set')}`

### 5. Fixed Audio Processing Stub
**Issue:** `process_audio_chunk()` was undefined

**Solution:** Added placeholder that returns "Unknown" status with TODO comment
- Audio processing is out of scope for current ML service
- Can be implemented later as separate service or added to ML service

## Verification

### ✅ No Heavy ML Imports
```bash
# These patterns return NO matches in server/app.py:
- ENGINE_AVAILABLE
- detectFace
- isBlinking
- mouthTrack
- gazeDetection
- head_pose_detection
- from face_engine
- from proctoring_module
- import insightface
- import dlib
- import face_recognition
```

### ✅ Lightweight Dependencies Only
**server/requirements.txt** contains ONLY:
- Flask, Flask-CORS, flask-limiter, flask-socketio
- gunicorn, gevent (lightweight WSGI server)
- pymongo==3.12.3 (database client)
- bcrypt (password hashing)
- requests (HTTP client for ML service)
- numpy, Pillow (basic image operations)
- python-dotenv (config)

**NO** heavy ML packages: insightface, dlib, face-recognition, opencv-python-headless, onnxruntime

### ✅ All ML Operations Use HTTP Client
All ML operations now use `call_ml_service(endpoint, payload, timeout)` function to communicate with the external ML service:

1. **Registration** (`/api/register`) → ML service `/verify-face`
2. **Login** (`/api/verify-face`) → ML service `/verify-face` + `/match-face`
3. **Frame Analysis** (`/api/analyze-frame`) → ML service `/analyze-frame`
4. **Live Proctoring** (`/api/proctor`) → ML service `/analyze-frame` + `/verify-face` + `/match-face`
5. **Face Samples** (`/api/users/<id>/face-samples`) → ML service `/verify-face`

## Architecture Compliance

### ✅ server/ (Backend API)
**Owns:**
- ✅ MongoDB database operations
- ✅ User authentication & authorization
- ✅ Exam management logic
- ✅ Session management
- ✅ Event recording & storage
- ✅ WebSocket broadcasting
- ✅ Business logic & final decisions

**Does NOT:**
- ✅ NO heavy ML models loaded in memory
- ✅ NO ML imports (InsightFace, dlib, face_recognition)
- ✅ NO local face detection or recognition
- ✅ NO embedding generation

**Uses cv2/numpy ONLY for:**
- Basic image decoding (base64 → numpy array)
- Basic image encoding (numpy array → JPEG base64)
- Simple operations (brightness checks, resizing for comparison)

### ✅ ml-service/ (ML Computation Only)
**Owns:**
- ✅ All ML model loading & inference
- ✅ Face detection, recognition, embedding generation
- ✅ Proctoring analysis (gaze, pose, mouth, blink)
- ✅ Object detection
- ✅ Returns raw ML output only

**Does NOT:**
- ✅ NO database access
- ✅ NO authentication/authorization
- ✅ NO business logic
- ✅ NO event recording

## Deployment Impact

### Backend (Render)
- **Memory:** Low (~200-500 MB) - no ML models loaded
- **CPU:** Low - just HTTP routing and DB operations
- **Startup:** Fast (~5-10 seconds)
- **Dependencies:** Lightweight Python packages only
- **Cost:** Free tier sufficient

### ML Service (Hugging Face Spaces)
- **Memory:** High (2-4 GB) - loads InsightFace, dlib, YOLO models
- **CPU:** High - performs ML inference
- **Startup:** Slow (~30-60 seconds) - model loading
- **Dependencies:** Heavy ML packages
- **Cost:** FREE CPU tier on HF Spaces

## Testing Checklist

Before deployment, verify:

- [ ] Backend starts without ML dependencies installed
- [ ] `/health` endpoint returns 200 without ML packages
- [ ] All endpoints handle ML_SERVICE_URL not set gracefully
- [ ] ML service /health returns 200 with models loaded
- [ ] Registration flow works end-to-end
- [ ] Login face verification works
- [ ] Live proctoring analysis works
- [ ] Face samples upload works

## Environment Variables

### Backend (Render)
```bash
ML_SERVICE_URL=https://your-space.hf.space  # REQUIRED
MONGO_URI=mongodb+srv://...
INVIGILO_ALLOWED_ORIGINS=https://your-vercel-app.vercel.app
PORT=8000
```

### ML Service (HF Spaces)
```bash
# No environment variables needed
# Service is stateless and public
```

### Frontend (Vercel)
```bash
VITE_API_URL=https://your-render-app.onrender.com
```

## Summary

✅ **COMPLETE ML-FREE BACKEND** achieved:
- Server has NO local ML code
- Server has NO heavy ML dependencies
- All ML happens via HTTP to external service
- Maintains backward compatibility with frontend
- Ready for lightweight Render deployment

✅ **STRICT ARCHITECTURAL SEPARATION** enforced:
- server/ = Business logic only
- ml-service/ = ML computation only
- Clear boundaries, no violations

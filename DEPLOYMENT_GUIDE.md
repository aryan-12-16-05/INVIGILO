# Invigilo Proctoring System - Complete Deployment Guide

## Architecture Overview

This system uses a **distributed microservices architecture** to work around free-tier limitations:

```
┌─────────────┐         ┌─────────────────┐         ┌──────────────────┐
│   Vercel    │────────>│  Railway/Render │────────>│ Hugging Face     │
│  (Frontend) │  HTTPS  │  (Backend API)  │  HTTPS  │ Spaces (ML)      │
│  React App  │         │  Flask + MongoDB│         │ InsightFace/dlib │
└─────────────┘         └─────────────────┘         └──────────────────┘
```

- **Frontend (Vercel)**: Serves React UI, handles user interactions
- **Backend API (Railway/Render)**: Lightweight Flask server for auth, exams, database operations
- **ML Service (Hugging Face Spaces)**: Heavy ML processing (face recognition, proctoring analysis)

---

## Prerequisites

Before you begin, make sure you have:

1. **Git** installed on your machine
2. **Node.js** (v16 or higher) and **npm** installed
3. **Python 3.11** installed
4. Accounts on:
   - [Vercel](https://vercel.com) (for frontend)
   - [Railway](https://railway.app) or [Render](https://render.com) (for backend)
   - [Hugging Face](https://huggingface.co) (for ML service)
   - [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) (for database)

---

## Part 1: Local Testing

### Step 1.1: Test ML Service Locally

```powershell
# Navigate to ml-service directory
cd ml-service

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run ML service (default port 5001)
python app.py
```

The ML service should start on `http://localhost:5001`. You should see:
```
[FACE_ENGINE] Attempting to import InsightFace...
[ML-SERVICE] Starting ML service...
* Running on http://127.0.0.1:5001
```

**Test ML endpoints:**
```powershell
# Health check
curl http://localhost:5001/health

# Expected response: {"status":"healthy"}
```

### Step 1.2: Test Backend API Locally

Open a **new terminal** (keep ML service running):

```powershell
# Navigate to server directory
cd server

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Create .env file with configuration
@"
MONGO_URI=your_mongodb_atlas_connection_string
ML_SERVICE_URL=http://localhost:5001
INVIGILO_ALLOWED_ORIGINS=http://localhost:5173
"@ | Out-File -FilePath .env -Encoding UTF8

# Run backend server (default port 5000)
python app.py
```

The backend should start on `http://localhost:5000`. You should see:
```
[FACE_ENGINE] InsightFace not available (expected - running on Railway)
* Running on http://127.0.0.1:5000
```

### Step 1.3: Test Frontend Locally

Open a **third terminal**:

```powershell
# Navigate to client directory
cd client

# Install dependencies
npm install

# Create .env.local file
@"
VITE_API_URL=http://localhost:5000/api
"@ | Out-File -FilePath .env.local -Encoding UTF8

# Run development server
npm run dev
```

Frontend should start on `http://localhost:5173`. Open your browser and test:
1. Registration with face verification
2. Login with face verification
3. Start an exam and verify proctoring works

---

## Part 2: Deploy ML Service to Hugging Face Spaces

### Step 2.1: Create Hugging Face Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Configure:
   - **Name**: `invigilo-ml-service` (or your choice)
   - **License**: Apache 2.0
   - **SDK**: **Gradio** (we'll replace with Flask)
   - **Hardware**: **CPU Basic** (Free tier)
   - **Visibility**: Public or Private (your choice)
4. Click **"Create Space"**

### Step 2.2: Push Code to Hugging Face

```powershell
# Navigate to ml-service directory
cd ml-service

# Initialize git (if not already initialized)
git init

# Add Hugging Face as remote (replace USERNAME and SPACE_NAME)
git remote add hf https://huggingface.co/spaces/USERNAME/invigilo-ml-service

# Create .gitignore
@"
venv/
__pycache__/
*.pyc
.env
"@ | Out-File -FilePath .gitignore -Encoding UTF8

# Stage all files
git add .

# Commit
git commit -m "Initial ML service deployment"

# Push to Hugging Face
git push hf main
```

### Step 2.3: Configure Hugging Face Space

1. Go to your Space's **Settings** tab
2. Under **"Hardware"**, confirm you're using **CPU Basic** (free)
3. Under **"Files and versions"**, verify these files were uploaded:
   - `app.py`
   - `requirements.txt`
   - `face_engine.py`
   - `proctoring_module.py`
   - `face_models/` directory
   - `object_detection_model/` directory

### Step 2.4: Get ML Service URL

1. Wait for the Space to build (5-10 minutes for first build)
2. Once deployed, your ML service URL will be:
   ```
   https://USERNAME-invigilo-ml-service.hf.space
   ```
3. Test it: `curl https://USERNAME-invigilo-ml-service.hf.space/health`
   - Expected: `{"status":"healthy"}`

**⚠️ Important**: Hugging Face Spaces can "sleep" after inactivity. First request may take 30-60 seconds to wake up.

---

## Part 3: Deploy Backend to Railway

### Step 3.1: Create Railway Project

1. Go to [Railway](https://railway.app)
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Connect your GitHub account and select your `invigilo-proctoring` repository
4. Railway will detect the project

### Step 3.2: Configure Railway Service

1. In Railway dashboard, click **"Settings"**
2. Under **"Root Directory"**, set to: `server`
3. Under **"Build Command"**, leave empty (Railway auto-detects)
4. Under **"Start Command"**, set to:
   ```
   gunicorn --worker-class gevent --workers 1 --bind 0.0.0.0:$PORT app:app --timeout 300
   ```

### Step 3.3: Set Environment Variables

In Railway dashboard, go to **"Variables"** tab and add:

```
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/invigilo?retryWrites=true&w=majority
ML_SERVICE_URL=https://USERNAME-invigilo-ml-service.hf.space
INVIGILO_ALLOWED_ORIGINS=*
PORT=5000
```

**Replace**:
- `username:password@cluster.mongodb.net` with your MongoDB Atlas credentials
- `USERNAME-invigilo-ml-service.hf.space` with your actual HF Space URL

### Step 3.4: Deploy Backend

1. Railway will automatically deploy after you set environment variables
2. Wait for deployment (2-3 minutes)
3. Railway will provide a public URL like: `https://invigilo-production.up.railway.app`
4. Test it:
   ```powershell
   curl https://your-railway-app.up.railway.app/
   # Expected: "Invigilo Backend Running"
   ```

---

## Part 4: Deploy Frontend to Vercel

### Step 4.1: Create Vercel Project

1. Go to [Vercel](https://vercel.com)
2. Click **"Add New"** → **"Project"**
3. Import your GitHub repository
4. Configure:
   - **Framework Preset**: Vite
   - **Root Directory**: `client`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

### Step 4.2: Set Environment Variables

In Vercel dashboard, go to **Settings** → **Environment Variables** and add:

```
VITE_API_URL=https://your-railway-app.up.railway.app/api
```

**⚠️ Important**: Replace with your actual Railway URL from Part 3.

### Step 4.3: Deploy Frontend

1. Click **"Deploy"**
2. Wait for deployment (1-2 minutes)
3. Vercel will provide a URL like: `https://invigilo-proctoring.vercel.app`

---

## Part 5: Final Configuration & Testing

### Step 5.1: Update CORS Settings

Go back to **Railway** environment variables and update:

```
INVIGILO_ALLOWED_ORIGINS=https://invigilo-proctoring.vercel.app
```

Replace with your actual Vercel URL. Railway will automatically redeploy.

### Step 5.2: End-to-End Testing

1. Open your Vercel URL in a browser
2. **Test Registration**:
   - Register as a student with face images
   - Verify registration succeeds (check Railway logs for ML service calls)
3. **Test Login**:
   - Login with face verification
   - Verify face matching works correctly
4. **Test Proctoring**:
   - Start an exam
   - Allow camera access
   - Verify real-time proctoring detects:
     - Face presence
     - Multiple faces
     - Gaze direction
     - Mouth movement
     - Head pose

### Step 5.3: Monitor Logs

**Railway Logs** (Backend):
```
[REGISTER] Received registration request
[REGISTER] Calling ML service for face verification
[REGISTER] ML service returned embeddings
[REGISTER] User registered successfully
```

**Hugging Face Logs** (ML Service):
```
[ML-SERVICE] POST /verify-face
[ML-SERVICE] Generating face embedding
[ML-SERVICE] Embedding generated successfully
```

---

## Troubleshooting

### Issue: ML Service Returns 500 Error

**Symptoms**: Registration/login fails with "ML service error"

**Solutions**:
1. Check HF Space is running: `curl https://your-space.hf.space/health`
2. Verify `ML_SERVICE_URL` in Railway has correct URL (no trailing slash)
3. Check HF Space logs for memory/timeout issues
4. ML service may be "sleeping" - first request takes longer

### Issue: CORS Errors in Browser

**Symptoms**: Frontend shows "CORS policy blocked" errors

**Solutions**:
1. Verify `INVIGILO_ALLOWED_ORIGINS` in Railway matches your Vercel URL exactly
2. Include `https://` in the origin URL
3. During development, can use `*` to allow all origins

### Issue: Face Verification Always Fails

**Symptoms**: Login shows "Face verification failed" with low similarity

**Solutions**:
1. Check lighting during registration and login (good lighting required)
2. Verify `FACE_SIMILARITY_THRESHOLD` setting (default 0.56)
3. Check ML service logs for face detection errors
4. Try re-registering with clearer face images

### Issue: MongoDB Connection Fails

**Symptoms**: Railway logs show "MongoClient connection error"

**Solutions**:
1. Verify `MONGO_URI` is correct in Railway
2. Whitelist Railway's IP in MongoDB Atlas (or use 0.0.0.0/0 for testing)
3. Check MongoDB Atlas cluster is running
4. Verify database user has read/write permissions

### Issue: Hugging Face Space Build Fails

**Symptoms**: HF Space shows "Build failed" status

**Solutions**:
1. Check `requirements.txt` has all dependencies
2. Verify model files (`face_models/`, `object_detection_model/`) were uploaded
3. Check HF Space build logs for specific errors
4. Ensure Python version compatibility (Python 3.11 recommended)

---

## Cost Breakdown (Free Tier Limits)

| Service | Free Tier | Limits |
|---------|-----------|--------|
| **Vercel** | Free | Unlimited bandwidth, 100 GB/month |
| **Railway** | $5 free credit/month | ~500 hours of usage |
| **Hugging Face** | Free CPU Space | Sleeps after 48h inactivity |
| **MongoDB Atlas** | Free M0 cluster | 512 MB storage |

**Estimated Monthly Cost**: $0-5 (Railway credit)

---

## Architecture Benefits

✅ **Scalability**: Each service can scale independently
✅ **Reliability**: If ML service sleeps, backend remains operational
✅ **Free-tier friendly**: Distributed load across platforms
✅ **Easy debugging**: Isolated services with separate logs
✅ **Security**: ML processing isolated from user data

---

## Production Recommendations

For production deployment, consider:

1. **Upgrade Hardware**: 
   - Hugging Face: GPU Space ($0.60/hour) for faster ML processing
   - Railway: Pro plan ($20/month) for more resources

2. **Add Monitoring**:
   - Railway: Enable log drains to external service
   - Hugging Face: Set up health check alerts
   - Frontend: Add error tracking (Sentry, LogRocket)

3. **Optimize Performance**:
   - Add Redis cache for frequent embeddings
   - Use CDN for frontend assets
   - Implement request queuing for ML service

4. **Security Enhancements**:
   - Use API keys for backend ↔ ML service communication
   - Implement rate limiting per user
   - Add request signing/verification
   - Use environment-specific CORS origins

---

## Quick Reference

### Environment Variables Summary

**Frontend (Vercel)**:
```
VITE_API_URL=https://your-railway-app.up.railway.app/api
```

**Backend (Railway)**:
```
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/invigilo
ML_SERVICE_URL=https://your-space.hf.space
INVIGILO_ALLOWED_ORIGINS=https://your-vercel-app.vercel.app
PORT=5000
```

**ML Service (Hugging Face)**: No environment variables needed

---

## Support & Resources

- **Documentation**: See `PROCTORING_SYSTEM.md` for system architecture
- **GitHub Issues**: Report bugs on repository
- **Railway Docs**: https://docs.railway.app
- **Hugging Face Docs**: https://huggingface.co/docs/hub/spaces
- **Vercel Docs**: https://vercel.com/docs

---

**Deployment Complete! 🎉**

Your Invigilo Proctoring System is now live with:
- Secure face verification ✓
- Real-time proctoring ✓
- Distributed architecture ✓
- Free-tier optimized ✓

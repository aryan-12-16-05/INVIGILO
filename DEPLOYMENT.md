# Invigilo Proctoring System - Complete Deployment Guide

## 🏗️ Architecture Overview

```
┌─────────────┐         ┌─────────────┐         ┌──────────────────┐
│   Vercel    │────────>│   Render    │────────>│ Hugging Face     │
│  (Frontend) │  HTTPS  │  (Backend)  │  HTTPS  │ Spaces (ML)      │
│  React+Vite │         │ Flask+Mongo │         │ InsightFace/dlib │
└─────────────┘         └─────────────┘         └──────────────────┘
                              │
                              ↓
                        ┌─────────────┐
                        │ MongoDB     │
                        │ Atlas       │
                        │ (Database)  │
                        └─────────────┘
```

**Why This Architecture?**
- **Render FREE tier** has limited memory (~512MB) - can't run heavy ML models
- **Hugging Face Spaces FREE** provides sufficient resources for ML inference
- **MongoDB Atlas FREE** tier (M0) provides 512MB storage - sufficient for the app
- **Vercel FREE** tier is perfect for static React apps

---

## 📋 Prerequisites

Before starting, create accounts on:
1. ✅ [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) - Database
2. ✅ [Hugging Face](https://huggingface.co/join) - ML Service
3. ✅ [Render](https://render.com/register) - Backend API
4. ✅ [Vercel](https://vercel.com/signup) - Frontend
5. ✅ [GitHub](https://github.com/signup) - Code repository

**Local Requirements:**
- Git installed
- Node.js v18+ and npm
- Python 3.11+

---

## 🗄️ Step 1: MongoDB Atlas Setup

### 1.1 Create a Free Cluster

1. Go to [MongoDB Atlas](https://cloud.mongodb.com)
2. Click **"Build a Database"**
3. Choose deployment option:
   - Select **"M0 FREE"** (Shared, 512MB storage)
   - Click **"Create"**

4. Choose cloud provider & region:
   - **Provider**: AWS (recommended)
   - **Region**: Choose closest to your users (e.g., `us-east-1`, `eu-west-1`, `ap-south-1`)
   - **Name**: `invigilo-cluster` (or any name you prefer)
   - Click **"Create Cluster"**

### 1.2 Create Database User

1. In the **"Security Quickstart"** popup (or Security > Database Access):
   - Click **"Add New Database User"**
   - **Authentication Method**: Password
   - **Username**: `invigilo_admin`
   - **Password**: Generate a secure password (click "Autogenerate Secure Password" and **SAVE IT**)
   - **Database User Privileges**: Select **"Read and write to any database"**
   - Click **"Add User"**

### 1.3 Configure Network Access

1. Go to **"Security" > "Network Access"**
2. Click **"Add IP Address"**
3. Choose:
   - **"Allow Access from Anywhere"** (0.0.0.0/0)
   - Description: `All IPs for Render/Vercel`
   - Click **"Confirm"**

> ⚠️ **Security Note**: For production, restrict to specific IPs. For now, this allows Render/Vercel/HF Spaces to connect.

### 1.4 Get Connection String

1. Go to **"Database" > "Connect"**
2. Click **"Connect your application"**
3. **Driver**: Python
4. **Version**: 3.6 or later
5. Copy the connection string:
   ```
   mongodb+srv://invigilo_admin:<password>@invigilo-cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
6. Replace `<password>` with your actual password
7. **Save this connection string** - you'll need it for Render

---

## 🤖 Step 2: Deploy ML Service to Hugging Face Spaces

### 2.1 Create a New Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Configure:
   - **Owner**: Your username
   - **Space name**: `invigilo-ml-service`
   - **License**: MIT (or your choice)
   - **Select the Space SDK**: **Gradio** (or Docker)
   - **Space hardware**: **CPU basic** (FREE)
   - **Visibility**: Public (required for free tier)
   - Click **"Create Space"**

### 2.2 Upload ML Service Code

#### Option A: Via Git (Recommended)

```powershell
# Clone the Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/invigilo-ml-service
cd invigilo-ml-service

# Copy ML service files from your project
# Copy from your invigilo-proctoring/ml-service/ directory:
cp -r ../invigilo-proctoring/ml-service/* .

# Create .gitignore
echo "__pycache__/
*.pyc
*.pyo
venv/
.env" > .gitignore

# Commit and push
git add .
git commit -m "Initial ML service deployment"
git push
```

#### Option B: Via Web Interface

1. In your Space, click **"Files" > "Add file" > "Upload files"**
2. Upload these files from `ml-service/` directory:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `face_engine.py`
   - `proctoring_module.py`
   - `face_models/` (entire folder)
   - `object_detection_model/` (entire folder)

### 2.3 Configure Space

1. Go to **"Settings"** in your Space
2. Verify settings:
   - **Space SDK**: Gradio or Docker
   - **Python version**: 3.11
   - **Hardware**: CPU basic (free)
   - **Persistent storage**: OFF (not needed)

### 2.4 Wait for Build

1. Go to **"Logs"** tab
2. Wait for the build to complete (~5-10 minutes first time)
3. Look for: `Running on public URL: https://YOUR_USERNAME-invigilo-ml-service.hf.space`
4. **Save this URL** - this is your `ML_SERVICE_URL`

### 2.5 Test ML Service

1. Visit your Space URL: `https://YOUR_USERNAME-invigilo-ml-service.hf.space`
2. You should see a Gradio interface (if using Gradio) or API endpoints
3. Test the `/health` endpoint:
   ```
   https://YOUR_USERNAME-invigilo-ml-service.hf.space/health
   ```
4. Should return: `{"status": "ok", "models_loaded": true}`

---

## 🚀 Step 3: Deploy Backend to Render

### 3.1 Prepare Repository

1. Ensure your code is pushed to GitHub:
   ```powershell
   cd invigilo-proctoring
   git add -A
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

### 3.2 Create Web Service on Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** > **"Web Service"**
3. Connect your GitHub repository:
   - Click **"Connect account"** (if first time)
   - Authorize Render to access your repos
   - Find `invigilo-proctoring` repository
   - Click **"Connect"**

### 3.3 Configure Web Service

Fill in the following settings:

#### Basic Settings:
- **Name**: `invigilo-backend` (or any unique name)
- **Region**: Choose closest to your users (e.g., `Oregon (US West)`, `Frankfurt (Europe)`, `Singapore (Southeast Asia)`)
- **Branch**: `main`
- **Root Directory**: `server` ⚠️ **IMPORTANT**
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn -k gevent -w 1 --bind 0.0.0.0:$PORT app:app`

#### Instance Type:
- **Plan**: **Free** ($0/month)
- **Auto-Deploy**: ON (deploys automatically on git push)

### 3.4 Add Environment Variables

Click **"Environment"** section and add these variables:

| Key | Value | Notes |
|-----|-------|-------|
| `MONGO_URI` | `mongodb+srv://invigilo_admin:<password>@...` | Your MongoDB connection string from Step 1.4 |
| `ML_SERVICE_URL` | `https://YOUR_USERNAME-invigilo-ml-service.hf.space` | Your HF Space URL from Step 2.4 |
| `INVIGILO_ALLOWED_ORIGINS` | `https://invigilo-proctoring.vercel.app` | Will update after Vercel deployment |
| `FLASK_SECRET_KEY` | Generate: `python -c "import secrets; print(secrets.token_hex(32))"` | Run locally to generate |
| `PORT` | `8000` | Render provides this, but explicit is good |
| `GEMINI_API_KEY` | (optional) | For AI question generation feature |
| `FACE_SIMILARITY_THRESHOLD` | `0.56` | Face recognition threshold (optional) |

**Important Notes:**
- Replace `<password>` in `MONGO_URI` with your actual MongoDB password
- Replace `YOUR_USERNAME` in `ML_SERVICE_URL` with your actual HF username
- We'll update `INVIGILO_ALLOWED_ORIGINS` after deploying frontend

### 3.5 Deploy

1. Click **"Create Web Service"**
2. Render will start building your app
3. Watch the **"Logs"** tab for progress
4. Build takes ~3-5 minutes
5. Look for: `Your service is live 🎉`
6. **Save your backend URL**: `https://invigilo-backend.onrender.com` (or your custom name)

### 3.6 Verify Backend

1. Visit: `https://invigilo-backend.onrender.com/api/health`
2. Should return:
   ```json
   {
     "status": "ok",
     "service": "invigilo-server",
     "startedAt": "2025-12-20T10:00:00Z",
     "time": "2025-12-20T10:05:00Z"
   }
   ```

---

## 🎨 Step 4: Deploy Frontend to Vercel

### 4.1 Prepare Frontend Configuration

1. Update environment variable template in `client/.env.example`:
   ```env
   VITE_API_URL=https://invigilo-backend.onrender.com
   ```

### 4.2 Deploy to Vercel

#### Option A: Via Vercel CLI (Recommended)

```powershell
# Install Vercel CLI globally (if not installed)
npm install -g vercel

# Navigate to client directory
cd client

# Login to Vercel
vercel login

# Deploy
vercel

# Follow prompts:
# ? Set up and deploy "~/invigilo-proctoring/client"? Yes
# ? Which scope? Your username
# ? Link to existing project? No
# ? What's your project's name? invigilo-proctoring
# ? In which directory is your code located? ./
# ? Want to modify these settings? No
```

#### Option B: Via Vercel Dashboard

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click **"Add New..." > "Project"**
3. Import Git Repository:
   - Click **"Import"** on your `invigilo-proctoring` repo
   - If not listed, click **"Import Git Repository"** and enter GitHub URL

4. Configure Project:
   - **Framework Preset**: Vite
   - **Root Directory**: `client` ⚠️ **IMPORTANT - Click "Edit" and set to `client`**
   - **Build Command**: `npm run build` (auto-detected)
   - **Output Directory**: `dist` (auto-detected)
   - **Install Command**: `npm install` (auto-detected)

5. Add Environment Variables:
   - Click **"Environment Variables"**
   - Add variable:
     - **Name**: `VITE_API_URL`
     - **Value**: `https://invigilo-backend.onrender.com` (your Render URL from Step 3.5)
     - **Environment**: Production, Preview, Development (check all)
   - Click **"Add"**

6. Click **"Deploy"**

### 4.3 Wait for Deployment

1. Vercel will build your app (~2-3 minutes)
2. Watch the build logs
3. Look for: **"✓ Ready"** with a green checkmark
4. Vercel provides URLs:
   - **Production**: `https://invigilo-proctoring.vercel.app`
   - **Preview**: `https://invigilo-proctoring-git-main-username.vercel.app`

5. **Save your production URL**

### 4.4 Update CORS on Backend

Now that you have the frontend URL, update Render:

1. Go back to [Render Dashboard](https://dashboard.render.com)
2. Select your `invigilo-backend` service
3. Go to **"Environment"**
4. Find `INVIGILO_ALLOWED_ORIGINS`
5. Update value to: `https://invigilo-proctoring.vercel.app,https://invigilo-proctoring-git-main-username.vercel.app`
   - Include both production and preview URLs
   - Separate multiple URLs with commas (no spaces)
6. Click **"Save Changes"**
7. Render will automatically redeploy (~1 minute)

---

## ✅ Step 5: Verification & Testing

### 5.1 Test ML Service

```powershell
# Test health endpoint
curl https://YOUR_USERNAME-invigilo-ml-service.hf.space/health

# Expected response:
# {"status": "ok", "models_loaded": true}
```

### 5.2 Test Backend API

```powershell
# Test health endpoint
curl https://invigilo-backend.onrender.com/api/health

# Expected response:
# {"status": "ok", "service": "invigilo-server", ...}
```

### 5.3 Test Frontend

1. Visit: `https://invigilo-proctoring.vercel.app`
2. You should see the login page
3. Check browser console (F12) for errors

### 5.4 Test Full Registration Flow

1. Click **"Register"**
2. Fill in student/lecturer details
3. Enable camera for face capture
4. Complete registration
5. Verify you receive success message
6. Check MongoDB Atlas:
   - Go to **"Database" > "Browse Collections"**
   - You should see `invigilo_db.users` with your new user

### 5.5 Test Login with Face Verification

1. Go to login page
2. Enter credentials
3. Enable camera for face verification
4. Complete login
5. You should be redirected to dashboard

---

## 🔧 Step 6: Custom Domain (Optional)

### 6.1 Add Custom Domain to Vercel

1. Go to Vercel project settings
2. Click **"Domains"**
3. Add your domain (e.g., `proctoring.yourdomain.com`)
4. Follow DNS configuration instructions
5. Update `INVIGILO_ALLOWED_ORIGINS` on Render to include your custom domain

### 6.2 Add Custom Domain to Render

1. Go to Render service settings
2. Click **"Custom Domain"**
3. Follow instructions to add your domain
4. Update `VITE_API_URL` on Vercel to use your custom backend domain

---

## 🐛 Troubleshooting

### Issue: Render build fails with "No module named 'insightface'"

**Solution**: This is intentional! Backend should NOT have heavy ML dependencies.
- Verify `server/requirements.txt` does NOT include `insightface`, `dlib`, `face_recognition`
- ML processing happens on Hugging Face Spaces

### Issue: "ML service unavailable" error

**Causes:**
1. HF Space is sleeping (cold start ~30 seconds on free tier)
2. `ML_SERVICE_URL` environment variable incorrect on Render
3. HF Space build failed

**Solutions:**
1. Visit HF Space URL directly to wake it up
2. Check Render environment variables, ensure no trailing slash
3. Check HF Space logs for build errors

### Issue: CORS errors in browser console

**Causes:**
1. `INVIGILO_ALLOWED_ORIGINS` not set correctly on Render
2. Frontend URL not included in allowed origins

**Solutions:**
1. Go to Render > Environment
2. Verify `INVIGILO_ALLOWED_ORIGINS` includes your Vercel URL
3. No spaces in the comma-separated list
4. Include both production and preview URLs

### Issue: MongoDB connection fails

**Causes:**
1. IP not whitelisted in MongoDB Atlas
2. Wrong connection string
3. User credentials incorrect

**Solutions:**
1. MongoDB Atlas > Network Access > Add IP Address > Allow from Anywhere
2. Verify `MONGO_URI` format: `mongodb+srv://username:password@cluster.xxxxx.mongodb.net/`
3. Check username and password are correct (no special characters issues)

### Issue: Face verification fails during registration

**Causes:**
1. ML service not responding
2. Poor lighting conditions
3. Camera not working

**Solutions:**
1. Check ML service health endpoint
2. Ensure good lighting on face
3. Grant browser camera permissions
4. Check browser console for specific errors

### Issue: Render service keeps sleeping

**Note**: Free tier services sleep after 15 minutes of inactivity. First request after sleep takes ~30 seconds.

**Solutions** (optional):
1. Upgrade to paid plan ($7/month) for always-on
2. Use a cron job to ping your service every 10 minutes:
   - Use [cron-job.org](https://cron-job.org) (free)
   - Ping: `https://invigilo-backend.onrender.com/api/health`
   - Schedule: Every 10 minutes

---

## 📊 Monitoring & Maintenance

### Monitor Deployments

1. **Render**: Dashboard shows logs, metrics, deployment history
2. **Vercel**: Deployments tab shows build logs, analytics
3. **HF Spaces**: Logs tab shows ML service activity
4. **MongoDB**: Metrics shows database usage, operations

### Check Logs

**Render Backend Logs:**
```
Render Dashboard > invigilo-backend > Logs
```

**Vercel Frontend Logs:**
```
Vercel Dashboard > invigilo-proctoring > Deployments > [Click deployment] > Logs
```

**HF Spaces ML Logs:**
```
HF Space > Logs tab
```

### Database Backups (Important!)

MongoDB Atlas M0 (free) does NOT include automated backups.

**Manual Backup:**
1. MongoDB Atlas > Cluster > ... > Command Line Tools
2. Install `mongodump`
3. Run backup:
   ```powershell
   mongodump --uri="mongodb+srv://invigilo_admin:password@cluster.mongodb.net/invigilo_db"
   ```

**Recommendation**: Schedule manual backups weekly

---

## 🎯 Performance Tips

1. **Reduce ML Service Cold Starts**:
   - Keep HF Space "warm" by pinging every 10-15 minutes
   - Consider using Hugging Face's inference API for production

2. **Optimize Render Free Tier**:
   - Service sleeps after 15 minutes inactivity
   - Consider cron job to keep alive during exam hours
   - Or upgrade to paid plan for better reliability

3. **MongoDB Query Optimization**:
   - Indexes are already created in code
   - Monitor slow queries in Atlas Performance Advisor
   - Keep proctor events TTL enabled (30 days default)

4. **Frontend Performance**:
   - Vercel automatically caches static assets
   - Use browser DevTools to check load times
   - Images are optimized during build

---

## 💰 Cost Summary

| Service | Tier | Cost | Limits |
|---------|------|------|--------|
| **Vercel** | Free | $0 | 100GB bandwidth, 100 deployments/month |
| **Render** | Free | $0 | 750 hours/month, sleeps after 15min inactive |
| **HF Spaces** | Free CPU | $0 | Public spaces only, limited compute |
| **MongoDB Atlas** | M0 Free | $0 | 512MB storage, shared CPU |
| **TOTAL** | - | **$0/month** | Sufficient for testing/small scale |

**Production Recommendations** (optional upgrades):
- Render: $7/month for always-on, better performance
- HF Spaces: $9/month for GPU, persistent storage
- MongoDB: $9/month for M2 (2GB, automated backups)
- Total production: ~$25/month

---

## 🎓 Next Steps

1. **Customize Branding**: Update app name, logo, colors in frontend
2. **Configure Settings**: Adjust face recognition thresholds in admin panel
3. **Add Users**: Invite lecturers and students via registration
4. **Create Exams**: Use AI question generation or manual entry
5. **Monitor**: Keep eye on logs during first real exam session
6. **Backup**: Set up regular database backups
7. **Scale**: Upgrade services as usage grows

---

## 📞 Support & Resources

- **GitHub Issues**: Report bugs on your repository
- **Render Docs**: https://render.com/docs
- **Vercel Docs**: https://vercel.com/docs
- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **MongoDB Docs**: https://docs.atlas.mongodb.com

---

## ✅ Deployment Checklist

- [ ] MongoDB Atlas cluster created with database user
- [ ] MongoDB connection string saved and working
- [ ] HF Space created with ML service deployed
- [ ] ML service `/health` endpoint responding
- [ ] Render backend deployed successfully
- [ ] Backend `/api/health` endpoint responding
- [ ] Vercel frontend deployed successfully
- [ ] CORS configured correctly on Render
- [ ] Registration flow tested successfully
- [ ] Login with face verification tested
- [ ] Exam creation tested
- [ ] Proctoring features tested
- [ ] Database shows correct data in MongoDB Atlas
- [ ] All environment variables configured correctly
- [ ] Custom domains configured (if applicable)
- [ ] Database backup strategy established

---

**🎉 Congratulations! Your Invigilo Proctoring System is now live!**

Access your app at: `https://invigilo-proctoring.vercel.app`

# Invigilo Development Quick Start

## Start All Services (Local Development)

### Terminal 1: Redis Server
```bash
redis-server
```

### Terminal 2: Flask API
```bash
cd server
python app.py
```

### Terminal 3: Celery Worker (Phase 1 - Optional)
```bash
cd server
celery -A celery_worker.celery_app worker --loglevel=info --pool=solo
```

### Terminal 4: React Client
```bash
cd client
npm run dev
```

---

## Environment Variables Required

```env
# MongoDB
MONGO_URI=mongodb+srv://...

# ML Service (Hugging Face)
ML_SERVICE_URL=https://your-ml-service.hf.space
ML_SHARED_SECRET=your-secret-key

# Redis (Phase 1 - Async Queue)
REDIS_URL=redis://localhost:6379/0

# Optional
FACE_SIMILARITY_THRESHOLD=0.58
```

---

## Verify Services

```bash
# Check Redis
redis-cli ping  # Should output: PONG

# Check Flask API
curl http://localhost:5000/api/health

# Check Celery Worker
celery -A celery_worker.celery_app inspect active

# Check React Client
# Open http://localhost:5173 in browser
```

---

## Architecture Modes

### Synchronous Mode (Default - Backward Compatible)
```json
// Client request
{
  "imageDataUrl": "data:image/jpeg;base64,...",
  "userId": "123",
  "examId": "456",
  "async": false  // or omit
}
```
- Response: 200 OK with results immediately
- Use for: Testing, small exams (<20 students)

### Async Mode (Phase 1 - Production Scale)
```json
// Client request
{
  "imageDataUrl": "data:image/jpeg;base64,...",
  "userId": "123",
  "examId": "456",
  "async": true  // 👈 Enable background processing
}
```
- Response: 202 Accepted with task_id
- Violations arrive via Socket.IO
- Use for: Large exams (50+ students)

---

## Monitoring Commands

```bash
# Celery worker status
celery -A celery_worker.celery_app inspect stats

# Active tasks
celery -A celery_worker.celery_app inspect active

# Task history
celery -A celery_worker.celery_app events

# Redis monitoring
redis-cli monitor

# Flower dashboard (optional)
pip install flower
celery -A celery_worker.celery_app flower --port=5555
# Open http://localhost:5555
```

---

## Common Issues

### "CELERY WARNING: Celery not available"
- **Solution:** Install dependencies: `pip install celery redis`
- **Impact:** System falls back to synchronous mode (slower)

### "Connection refused to Redis"
- **Solution:** Start Redis server: `redis-server`
- **Impact:** Async mode unavailable

### "ML service unavailable"
- **Solution:** Check `ML_SERVICE_URL` is correct and service is running
- **Impact:** Frame analysis returns "Unknown" for all fields

---

## Deployment Checklist

### Render Production

1. **Web Service (Flask API):**
   - Build: `pip install -r server/requirements.txt`
   - Start: `cd server && gunicorn -k eventlet -w 1 -b 0.0.0.0:$PORT app:app`
   - Env vars: `MONGO_URI`, `ML_SERVICE_URL`, `ML_SHARED_SECRET`, `REDIS_URL`

2. **Background Worker (Celery):**
   - Build: `pip install -r server/requirements.txt`
   - Start: `cd server && celery -A celery_worker.celery_app worker --loglevel=info --concurrency=4`
   - Env vars: Same as Web Service

3. **Redis Add-on:**
   - Add Redis to service
   - Copy `REDIS_URL` to both Web Service and Background Worker

---

## Performance Targets

| Metric | Synchronous | Async (Phase 1) | MediaPipe (Phase 2) |
|--------|-------------|-----------------|---------------------|
| Request Latency | 2-5s | <50ms | <50ms |
| Max Students | 20 | 100+ | 500+ |
| Timeout Rate | 15% | <1% | <0.1% |
| Server CPU | 80-100% | 40-60% | 10-20% |
| Network Usage | High | High | Low |

---

## Next Steps

1. ✅ **Phase 1 (Current):** Async queue with Celery + Redis
2. 📋 **Phase 2 (Next):** MediaPipe client-side detection
3. 📋 **Phase 3 (Future):** Backend validation only

**See:** `PHASE1_ASYNC_SETUP.md` for detailed setup instructions.

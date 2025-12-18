# Deployment (Free Tier) — Vercel (client) + Render (server) + MongoDB Atlas

This repo is a **Vite/React (client)** + **Flask/Socket.IO (server)** app.

## 0) Prereqs

- MongoDB Atlas cluster created (free tier is fine)
- A GitHub repo with this project pushed

You’ll deploy:
- **Client** → Vercel
- **Server API + WebSocket** → Render

---

## 1) Backend on Render

### Create a new Render Web Service


### Environment variables (Render)

Add these in Render → Service → *Environment*:

  - Example:
    - `https://invigilo.vercel.app,https://invigilo-aryan.vercel.app`

Optional recommended:

### Free-tier memory note (important)

This repo has **optional heavy ML dependencies** (InsightFace / dlib / OpenCV / audio) that can cause Render free tier builds to fail (OOM / native wheels).

For a reliable free-tier deploy:

- Keep Render using `server/requirements.txt` (core only)
- Set `INVIGILO_ENABLE_HEAVY_ML=0` (default)

If you want full ML features locally or on a larger host, install the extra deps too:

- `server/requirements-ml.txt`
- Set `INVIGILO_ENABLE_HEAVY_ML=1`

### Start command

Render start command (recommended):

- `gunicorn -k eventlet -w 1 app:app`

Notes:
- `-w 1` is common for Socket.IO on free tier (multiple workers require sticky sessions).
- Render will expose a public URL like: `https://<service>.onrender.com`.

### Health check

A health endpoint exists:
- `GET /api/health`

---

## 2) Frontend on Vercel

### Create a new Vercel project

- **Framework:** Vite
- **Root directory:** `client`

### Environment variables (Vercel)

Add:
- `VITE_API_URL` = `https://<your-render-service>.onrender.com/api`

Deploy.

---

## 3) Quick sanity checks after deploy

- Open the Vercel URL (site loads)
- Verify backend is reachable:
  - `https://<your-render-service>.onrender.com/api/health`
- Try login/signup
- Start an exam and confirm proctor events appear

---

## 4) Common issues

### CORS errors

If requests fail with CORS:
- Ensure Render has `INVIGILO_ALLOWED_ORIGINS` set to your Vercel domain(s)
- Redeploy Render service

### Socket.IO not receiving events

- Ensure your frontend connects to the correct backend base URL
- Ensure `INVIGILO_ALLOWED_ORIGINS` includes the Vercel domain
- Use a single Gunicorn worker (`-w 1`) on free tier

### File evidence persistence

Render free tier disks are limited. For real scale, store evidence in S3/R2 and keep only URLs in Mongo.

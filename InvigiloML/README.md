---
title: InvigiloML
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Invigilo ML Service

Heavy ML processing service for face recognition and proctoring analysis.

## Deployment: Hugging Face Spaces

This service runs on Hugging Face Spaces (FREE tier with GPU).

### Setup:

1. Create new Space on Hugging Face
2. Choose: **Gradio** SDK (or Docker if you prefer)
3. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `face_engine.py` (copy from `../server/`)
   - `proctoring_module.py` (copy from `../server/`)
   - Download models (see below)

### Models Required:

Place in `face_models/` directory:
- `shape_predictor_68_face_landmarks.dat`

Place in `object_detection_model/config/`:
- `yolov3-tiny.cfg`
- `coco.names`

### Environment Variables:

None required - runs standalone.

### Testing Locally:

```bash
python app.py
```

Service runs on http://localhost:7860

### Endpoints:

- `GET /health` - Health check
- `POST /verify-face` - Generate face embedding
- `POST /match-face` - Compare embeddings  
- `POST /analyze-frame` - Proctoring analysis

### Connect from Backend:

In main backend (`server/`), set:
```
ML_SERVICE_URL=https://your-space.hf.space
```

Backend will call ML service via HTTP.

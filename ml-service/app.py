"""
Invigilo ML Service - Hugging Face Spaces Deployment
====================================================
Heavy ML processing endpoint for face recognition and proctoring analysis.

This service runs independently on Hugging Face Spaces (FREE tier).
The main backend (Render) calls these endpoints via HTTP.

Endpoints:
- POST /verify-face: Generate face embedding from image
- POST /match-face: Compare two face embeddings
- POST /analyze-frame: Full proctoring analysis (face, gaze, pose, etc.)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import base64
import numpy as np
import hmac
import hashlib
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Allow calls from Render backend

ML_SHARED_SECRET = os.getenv('ML_SHARED_SECRET', '').strip()


def _verify_request_signature(payload: dict) -> bool:
    """Verify HMAC signature sent by the backend.

    If ML_SHARED_SECRET is not set, signature verification is disabled.
    """
    if not ML_SHARED_SECRET:
        return True
    try:
        supplied = (request.headers.get('X-Signature') or '').strip()
        if not supplied:
            return False
        payload_str = json.dumps(payload, sort_keys=True)
        expected = hmac.new(
            ML_SHARED_SECRET.encode('utf-8'),
            payload_str.encode('utf-8'),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(supplied, expected)
    except Exception:
        return False

# Import heavy ML modules (only runs on HF Spaces)
try:
    from face_engine import get_engine
    engine = get_engine()
    print('[ML-SERVICE] Face engine loaded successfully')
except Exception as e:
    print(f'[ML-SERVICE] ERROR: Could not load face engine: {e}')
    engine = None

try:
    from proctoring_module import (
        detectFace, isBlinking, mouthTrack,
        headPoseEstimation, gazeEstimation
    )
    print('[ML-SERVICE] Proctoring modules loaded successfully')
    PROCTORING_AVAILABLE = True
except Exception as e:
    print(f'[ML-SERVICE] WARNING: Proctoring modules unavailable: {e}')
    PROCTORING_AVAILABLE = False


def _bool_env(name: str, default: bool) -> bool:
    try:
        raw = os.getenv(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    except Exception:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        if raw is None:
            return float(default)
        return float(str(raw).strip())
    except Exception:
        return float(default)


def decode_base64_image(data_url):
    """Decode base64-encoded image"""
    try:
        if ',' in data_url:
            header, encoded = data_url.split(',', 1)
        else:
            encoded = data_url
        img_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(img_bytes))
        rgb = np.array(image.convert('RGB'))
        # Return as BGR for OpenCV compatibility
        return rgb[:, :, ::-1]
    except Exception as e:
        print(f"[ML-SERVICE] Error decoding image: {e}")
        return None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "invigilo-ml",
        "engine_available": engine is not None,
        "proctoring_available": PROCTORING_AVAILABLE
    }), 200


@app.route('/verify-face', methods=['POST'])
def verify_face():
    """
    Generate face embedding from image.
    
    Request:
        {
            "imageDataUrl": "data:image/jpeg;base64,..."
        }
    
    Response:
        {
            "embedding": [0.1, 0.2, ...],  # 512-dim vector
            "face_detected": true
        }
    """
    if engine is None:
        return jsonify({"error": "Face engine not available"}), 500
    
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    if not _verify_request_signature(data):
        return jsonify({"error": "Forbidden"}), 403

    image_data = data.get('imageDataUrl')
    
    if not image_data:
        return jsonify({"error": "imageDataUrl required"}), 400
    
    img = decode_base64_image(image_data)
    if img is None:
        return jsonify({"error": "Invalid image data"}), 400
    
    try:
        embedding = engine.embed(img)
        if embedding is None:
            return jsonify({
                "face_detected": False,
                "error": "No face detected in image"
            }), 400
        
        return jsonify({
            "embedding": embedding.tolist(),
            "face_detected": True
        }), 200
    
    except Exception as e:
        print(f'[ML-SERVICE] Error in verify-face: {e}')
        return jsonify({"error": str(e)}), 500


@app.route('/match-face', methods=['POST'])
def match_face():
    """
    Compare two face embeddings.
    
    Request:
        {
            "embedding1": [0.1, 0.2, ...],
            "embedding2": [0.1, 0.2, ...]
        }
    
    Response:
        {
            "similarity": 0.85,
            "match": true,  # if similarity > threshold
            "threshold": 0.6
        }
    """
    if engine is None:
        return jsonify({"error": "Face engine not available"}), 500
    
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    if not _verify_request_signature(data):
        return jsonify({"error": "Forbidden"}), 403

    emb1 = np.array(data.get('embedding1'))
    emb2 = np.array(data.get('embedding2'))
    
    if emb1 is None or emb2 is None:
        return jsonify({"error": "embedding1 and embedding2 required"}), 400
    
    try:
        similarity = engine.match(emb1, emb2)
        # Keep threshold consistent with the backend defaults (can be overridden on HF Spaces)
        threshold = _float_env('FACE_SIMILARITY_THRESHOLD', 0.58)
        
        return jsonify({
            "similarity": float(similarity),
            "match": bool(similarity >= threshold),
            "threshold": threshold
        }), 200
    
    except Exception as e:
        print(f'[ML-SERVICE] Error in match-face: {e}')
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    """
    Full proctoring analysis on a single frame.
    
    Request:
        {
            "imageDataUrl": "data:image/jpeg;base64,..."
        }
    
    Response:
        {
            "faceCount": 1,
            "blinkStatus": "No Blink",
            "mouthStatus": "Closed",
            "headPose": "Forward",
            "gazeDirection": "Center",
            "violations": []
        }
    """
    # HF Spaces free-tier can fail to build native deps (e.g., dlib). We must not hard-fail
    # the whole pipeline; instead return a safe, explicit degraded response.
    allow_degraded = _bool_env('ALLOW_DEGRADED_PROCTORING', True)
    
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    if not _verify_request_signature(data):
        return jsonify({"error": "Forbidden"}), 403

    image_data = data.get('imageDataUrl')
    
    if not image_data:
        return jsonify({"error": "imageDataUrl required"}), 400
    
    img = decode_base64_image(image_data)
    if img is None:
        return jsonify({"error": "Invalid image data"}), 400

    if not PROCTORING_AVAILABLE:
        if not allow_degraded:
            return jsonify({"error": "Proctoring modules not available"}), 500

        # Degraded response: keep schema stable so the backend can skip eventing.
        face_count = None
        blink_status = "Unknown"
        mouth_status = "Unknown"
        head_pose = "Unknown"
        gaze_dir = "Unknown"
        violations: list = []
        return jsonify({
            "proctoringAvailable": False,
            "proctoring_available": False,
            "faceCount": face_count,
            "face_count": face_count,
            "blinkStatus": blink_status,
            "blink_status": blink_status,
            "mouthStatus": mouth_status,
            "mouth_status": mouth_status,
            "headPose": head_pose,
            "head_pose": head_pose,
            "gazeDirection": gaze_dir,
            "gaze_direction": gaze_dir,
            "violations": violations,
        }), 200
    
    try:
        # Run all proctoring checks
        face_count, faces = detectFace(img)
        
        violations = []
        
        # Face count check
        if face_count == 0:
            violations.append("No face detected")
        elif face_count > 1:
            violations.append(f"Multiple faces detected ({face_count})")
        
        # Blink, mouth, gaze, pose (only if 1 face)
        blink_status = "Unknown"
        mouth_status = "Unknown"
        head_pose = "Unknown"
        gaze_dir = "Unknown"
        
        if face_count == 1:
            blink_status = isBlinking(faces, img)
            mouth_status = mouthTrack(faces, img)
            head_pose = headPoseEstimation(faces, img)
            gaze_dir = gazeEstimation(faces, img)
            
            # Check for violations
            if mouth_status == "Mouth Open":
                violations.append("Mouth open (possible talking)")
            
            if head_pose not in ["Forward", "Unknown"]:
                violations.append(f"Head turned {head_pose}")
            
            if gaze_dir not in ["Center", "Unknown"]:
                violations.append(f"Looking {gaze_dir}")
        
        return jsonify({
            "proctoringAvailable": True,
            "proctoring_available": True,
            # Provide both formats for backward/forward compatibility
            "faceCount": face_count,
            "face_count": face_count,
            "blinkStatus": blink_status,
            "blink_status": blink_status,
            "mouthStatus": mouth_status,
            "mouth_status": mouth_status,
            "headPose": head_pose,
            "head_pose": head_pose,
            "gazeDirection": gaze_dir,
            "gaze_direction": gaze_dir,
            "violations": violations
        }), 200
    
    except Exception as e:
        print(f'[ML-SERVICE] Error in analyze-frame: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))  # HF Spaces default port
    print(f'[ML-SERVICE] Starting on port {port}')
    app.run(host='0.0.0.0', port=port, debug=False)

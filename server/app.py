import os
import re
from flask import Flask, jsonify, request, redirect, url_for
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from pymongo import MongoClient
import bcrypt
from bson import ObjectId
from dotenv import load_dotenv
import requests
import json
import datetime
import numpy as np
import base64
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except Exception as e:
    print(f"[CV2] OpenCV unavailable: {e}")
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore
import time
from PIL import Image
import io
import hmac
import hashlib
import threading
from werkzeug.exceptions import HTTPException
from collections import Counter, deque
import math

def _bool_env(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


# ============================================================================
# ML SERVICE CLIENT (Hugging Face Spaces)
# ============================================================================
# Backend delegates heavy ML to separate service on Hugging Face Spaces
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "")
ML_SERVICE_TIMEOUT = 30  # seconds
ML_SHARED_SECRET = os.getenv("ML_SHARED_SECRET", "")  # HMAC secret for ML service auth

# Reuse HTTP connections for lower latency (especially when ML service is remote).
_ML_HTTP_SESSION = requests.Session()
try:
    _adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
    _ML_HTTP_SESSION.mount('http://', _adapter)
    _ML_HTTP_SESSION.mount('https://', _adapter)
except Exception:
    # If mounting fails for any reason, Session() still works.
    pass

# Simple in-process circuit breaker to avoid hammering HF Spaces on failures.
_ML_CB_LOCK = threading.Lock()
_ML_CB_STATE = {
    'fail_window_start': 0.0,
    'fail_count': 0,
    'open_until': 0.0,
}


def _ml_cb_settings():
    try:
        fail_threshold = int(os.getenv('ML_CB_FAIL_THRESHOLD', '3'))
    except Exception:
        fail_threshold = 3
    try:
        window_s = float(os.getenv('ML_CB_WINDOW_S', '30'))
    except Exception:
        window_s = 30.0
    try:
        cooldown_s = float(os.getenv('ML_CB_COOLDOWN_S', '45'))
    except Exception:
        cooldown_s = 45.0
    return fail_threshold, window_s, cooldown_s


def _ml_cb_is_open(now_ts: float):
    with _ML_CB_LOCK:
        open_until = float(_ML_CB_STATE.get('open_until') or 0.0)
    if now_ts < open_until:
        return True, max(0, int(open_until - now_ts))
    return False, 0


def _ml_cb_record_success():
    with _ML_CB_LOCK:
        _ML_CB_STATE['fail_window_start'] = 0.0
        _ML_CB_STATE['fail_count'] = 0
        _ML_CB_STATE['open_until'] = 0.0


def _ml_cb_record_failure(now_ts: float):
    fail_threshold, window_s, cooldown_s = _ml_cb_settings()
    with _ML_CB_LOCK:
        window_start = float(_ML_CB_STATE.get('fail_window_start') or 0.0)
        if (window_start == 0.0) or ((now_ts - window_start) > window_s):
            _ML_CB_STATE['fail_window_start'] = now_ts
            _ML_CB_STATE['fail_count'] = 1
        else:
            _ML_CB_STATE['fail_count'] = int(_ML_CB_STATE.get('fail_count') or 0) + 1

        if int(_ML_CB_STATE.get('fail_count') or 0) >= int(fail_threshold):
            _ML_CB_STATE['open_until'] = now_ts + float(cooldown_s)

def call_ml_service(endpoint: str, payload: dict, timeout: int = ML_SERVICE_TIMEOUT):
    """
    Call ML service endpoint via HTTP with HMAC authentication.
    
    Args:
        endpoint: e.g., "/verify-face", "/analyze-frame"
        payload: JSON request body
        timeout: request timeout in seconds
    
    Returns:
        (success: bool, data: dict)
    """
    if not ML_SERVICE_URL:
        print(f"[ML-CLIENT] ERROR: ML_SERVICE_URL not configured")
        return False, {"error": "ML service not configured"}

    now_ts = time.time()
    is_open, retry_after_s = _ml_cb_is_open(now_ts)
    if is_open:
        return False, {
            "error": "ML service temporarily unavailable",
            "circuitOpen": True,
            "retryAfterSeconds": retry_after_s,
        }
    
    url = ML_SERVICE_URL.rstrip('/') + endpoint
    
    # Generate HMAC signature for authentication
    headers = {"Content-Type": "application/json"}
    if ML_SHARED_SECRET:
        try:
            # Create deterministic payload string (sorted keys for consistency)
            payload_str = json.dumps(payload, sort_keys=True)
            
            # Generate HMAC-SHA256 signature
            signature = hmac.new(
                ML_SHARED_SECRET.encode('utf-8'),
                payload_str.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            headers["X-Signature"] = signature
            if _bool_env('ML_CLIENT_DEBUG', '0'):
                print(f"[ML-CLIENT] Added HMAC signature to request")
        except Exception as e:
            print(f"[ML-CLIENT] WARNING: Failed to generate signature: {e}")
    else:
        print(f"[ML-CLIENT] WARNING: ML_SHARED_SECRET not set - requests are not authenticated!")
    
    try:
        if _bool_env('ML_CLIENT_DEBUG', '0'):
            print(f"[ML-CLIENT] Calling {url}")
        response = _ML_HTTP_SESSION.post(url, json=payload, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            _ml_cb_record_success()
            return True, response.json()
        else:
            print(f"[ML-CLIENT] ERROR: {response.status_code} - {response.text}")
            _ml_cb_record_failure(time.time())
            return False, {"error": f"ML service error: {response.status_code}"}
    
    except requests.Timeout:
        print(f"[ML-CLIENT] ERROR: Request timeout after {timeout}s")
        _ml_cb_record_failure(time.time())
        return False, {"error": "ML service timeout"}
    except Exception as e:
        print(f"[ML-CLIENT] ERROR: {e}")
        _ml_cb_record_failure(time.time())
        return False, {"error": str(e)}


# ============================================================================
# NO LOCAL ML - All ML processing delegated to ml-service
# ============================================================================
# This backend is ML-free for lightweight deployment on Render.
# All face recognition, proctoring analysis happens via HTTP calls to ML service.

# --- Setup ---
load_dotenv()
app = Flask(__name__)

# Build/version identifier (useful to confirm Render is running the latest code)
APP_BUILD_ID = os.getenv("APP_BUILD_ID", "2025-12-21-render-debug")

app.secret_key = os.getenv('FLASK_SECRET_KEY', os.getenv('SECRET_KEY', 'dev-secret-change-me'))


def _internal_allowed() -> bool:
    """Restrict internal tooling endpoints.

    - If INTERNAL_DASHBOARD_TOKEN is set, require it via query param `token` or header `X-Internal-Token`.
    - If not set, allow only localhost access.
    """
    token = os.getenv('INTERNAL_DASHBOARD_TOKEN', '').strip()
    if token:
        supplied = (request.args.get('token') or request.headers.get('X-Internal-Token') or '').strip()
        return supplied == token
    # Fallback: localhost only
    ra = (request.remote_addr or '').strip()
    return ra in {'127.0.0.1', '::1'}

def _sign_token(user_id: str, ttl_seconds: int = 3600):
    """Create a simple HMAC-signed token (no external deps).

    Format: v1.<user_id>.<exp_ts>.<sig>
    """
    exp = int(time.time()) + int(ttl_seconds)
    msg = f"v1.{user_id}.{exp}".encode('utf-8')
    key = app.secret_key.encode('utf-8')
    sig = hmac.new(key, msg, hashlib.sha256).hexdigest()
    return f"v1.{user_id}.{exp}.{sig}"


def _verify_token(token: str):
    """Verify token created by _sign_token.

    Returns (ok: bool, user_id: str|None)
    """
    try:
        raw = (token or '').strip()
        if raw.lower().startswith('bearer '):
            raw = raw.split(' ', 1)[1].strip()
        parts = raw.split('.')
        if len(parts) != 4:
            return False, None
        v, user_id, exp_s, sig = parts
        if v != 'v1':
            return False, None
        exp = int(exp_s)
        if int(time.time()) > exp:
            return False, None
        msg = f"v1.{user_id}.{exp}".encode('utf-8')
        key = app.secret_key.encode('utf-8')
        expected = hmac.new(key, msg, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            return False, None
        return True, str(user_id)
    except Exception:
        return False, None


def _get_auth_header_token() -> str:
    try:
        return (request.headers.get('Authorization') or '').strip()
    except Exception:
        return ''


def _get_authenticated_user_id():
    """Best-effort auth: prefer Authorization Bearer token, fall back to X-User-Id."""
    auth = _get_auth_header_token()
    if auth:
        ok, uid = _verify_token(auth)
        if ok and uid:
            return uid
    # Back-compat fallback
    return (request.headers.get('X-User-Id') or '').strip() or None


def _get_authenticated_user_doc():
    uid = _get_authenticated_user_id()
    if not uid:
        return None
    try:
        return users_collection.find_one({'_id': ObjectId(uid)}) if ObjectId.is_valid(uid) else None
    except Exception:
        return None


# Socket.IO per-connection auth cache (best-effort)
SOCKET_AUTH = {}  # sid -> {userId, role}

# Initialize rate limiter to prevent abuse and brute force attacks
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per day", "200 per hour"],  # Global fallback limits
    storage_uri="memory://",  # In-memory storage (upgrade to Redis for production)
)

# Enable CORS with comprehensive configuration for all API endpoints
# Production: set INVIGILO_ALLOWED_ORIGINS to a comma-separated list of frontend origins
# e.g. https://invigilo.vercel.app,https://invigilo-aryan.vercel.app
_allowed_origins_env = os.getenv("INVIGILO_ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = "*" if _allowed_origins_env.strip() == "*" else [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]

# Allow CORS for API routes and a few legacy non-/api routes used by older clients.
# Note: Prefer /api/* in new code.
CORS(app, resources={
    r"/api/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-User-Id"],
        "max_age": 3600
    },
    # Legacy endpoints (kept for backwards-compat)
    r"/register": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-User-Id"],
        "max_age": 3600
    },
    r"/login": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-User-Id"],
        "max_age": 3600
    }
})

# ✅ Initialize WebSocket for Real-Time Proctor Updates
# IMPORTANT (Render): gevent/eventlet monkey-patching has proven unstable in this
# deployment. Default to 'threading' for reliability, and allow override via env.
_socketio_async_mode = os.getenv('SOCKETIO_ASYNC_MODE', '').strip() or 'threading'
socketio = SocketIO(
    app,
    cors_allowed_origins=ALLOWED_ORIGINS,
    async_mode=_socketio_async_mode,
    logger=True,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25
)

APP_START = datetime.datetime.utcnow()


@app.route('/', methods=['GET', 'HEAD'])
def root_ok():
    """Root endpoint for platform checks.

    Render (and some upstream proxies) probe `/` with HEAD/GET. Our API lives under
    `/api/*`, but returning 200 here avoids noisy 404 logs.
    """
    return "OK", 200


@app.route('/api/health', methods=['GET'])
def health():
    """Lightweight health endpoint for platform checks (Render)"""
    return jsonify({
        "status": "ok",
        "service": "invigilo-server",
        "startedAt": APP_START.isoformat() + "Z",
        "time": datetime.datetime.utcnow().isoformat() + "Z",
    }), 200


@app.route('/api/version', methods=['GET'])
def version():
    """Expose a small version payload for debugging deployments."""
    return jsonify({
        "service": "invigilo-server",
        "buildId": APP_BUILD_ID,
        "time": datetime.datetime.utcnow().isoformat() + "Z",
        "mlServiceUrlConfigured": bool(os.getenv("ML_SERVICE_URL", "").strip()),
        "mongoConfigured": bool(os.getenv("MONGO_URI", "").strip())
    }), 200


@app.errorhandler(Exception)
def handle_unhandled_exception(e):
    """Return JSON for API routes and log the full traceback.

    This prevents generic HTML 500 pages from hiding the real error.
    """
    # If it's an HTTPException (e.g. 404), keep its status code.
    status_code = 500
    detail = str(e)
    if isinstance(e, HTTPException):
        status_code = e.code or 500
        detail = e.description

    app.logger.exception("Unhandled exception on %s %s", request.method, request.path)

    if request.path.startswith('/api/') or request.path in {'/register', '/login'}:
        return jsonify({
            "error": "internal_server_error" if status_code >= 500 else "http_error",
            "detail": detail,
            "status": status_code,
            "path": request.path
        }), status_code

    # Non-API routes: return a simple text response.
    return "Internal Server Error", status_code

# Rate limit error handler
@app.errorhandler(429)
def ratelimit_handler(e):
    """
    Custom error handler for rate limit exceeded (429 Too Many Requests).
    Logs the violation and returns a friendly error message with retry information.
    """
    # Log rate limit violation for monitoring
    app.logger.warning(f"Rate limit exceeded from {get_remote_address()}: {e.description}")
    
    # Extract retry-after time from the exception
    retry_after = getattr(e, 'retry_after', 3600)  # Default to 1 hour if not available
    
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later.",
        "retry_after": retry_after
    }), 429

# ================================================================================
# PASSWORD VALIDATION
# ================================================================================
def validate_password(password):
    """
    Validates password strength according to security requirements.
    
    Requirements:
    - Minimum 8 characters
    - At least 1 uppercase letter (A-Z)
    - At least 1 lowercase letter (a-z)
    - At least 1 digit (0-9)
    - At least 1 special character (!@#$%^&*-_=+)
    
    Args:
        password (str): The password to validate
    
    Returns:
        tuple: (is_valid: bool, error_message: str)
            - is_valid: True if password meets all requirements, False otherwise
            - error_message: Specific feedback if invalid, empty string if valid
    
    Examples:
        >>> validate_password("abc123")
        (False, "Password must contain at least one uppercase letter")
        
        >>> validate_password("Abc123!")
        (True, "")
        
        >>> validate_password("Pass@123")
        (True, "")
    """
    if not password:
        return False, "Password is required"
    
    # Check minimum length
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    # Check for at least one uppercase letter
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    # Check for at least one lowercase letter
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    # Check for at least one digit
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one digit"
    
    # Check for at least one special character
    if not re.search(r'[!@#$%^&*\-_=+]', password):
        return False, "Password must contain at least one special character (!@#$%^&*-_=+)"
    
    # All checks passed
    return True, ""

# ================================================================================
# END PASSWORD VALIDATION
# ================================================================================

# ================================================================================
# ================================================================================
# USER DATA SANITIZATION
# ================================================================================
def sanitize_user_response(user):
    """
    Remove sensitive fields from user object before sending to client.
    
    Sensitive fields include:
    - password: Hashed password
    - faceEmbedding: Legacy single face embedding (128-dim biometric vector)
    - faceEmbeddings: Array of face embeddings (biometric data)
    - salt: Password salt (if stored separately)
    
    These fields are PRIVATE and should NEVER be sent to the frontend.
    Face verification happens on the backend only - clients receive
    only the verification result (verified: true/false, similarity score).
    
    Args:
        user (dict): User document from database
    
    Returns:
        dict: Sanitized user object safe for client consumption
    
    Example:
        user = users_collection.find_one({'email': email})
        user = sanitize_user_response(user)
        return jsonify({'user': user}), 200
    """
    if user is None:
        return None
    
    if not isinstance(user, dict):
        return user
    
    # Create a copy to avoid modifying the original
    sanitized = user.copy()
    
    # Remove sensitive biometric and authentication data
    sanitized.pop('password', None)
    sanitized.pop('faceEmbedding', None)  # Legacy single embedding
    sanitized.pop('faceEmbeddings', None)  # Array of embeddings
    sanitized.pop('salt', None)  # Password salt (if exists)
    
    return sanitized

# ================================================================================
# END USER DATA SANITIZATION
# ================================================================================

PROCTOR_STATE = {}
"""
In-memory rolling state for proctoring signals per (examId,userId):
PROCTOR_STATE[(examId,userId)] = {
    'poses': [(ts, headPose)],
    'gazes': [(ts, gazeDirection)],
    'mouths': [(ts, mouthStatus)],
    'faces': [(ts, faceCount)],
    'brightness': [(ts, avg_brightness)],
    'last_emit': { 'event_key': ts }
}
Note: This is a best-effort cache for live sessions; it resets on server restart.
"""

# --- Database ---
MONGO_URI = os.getenv("MONGO_URI")
client = None
db = None
users_collection = None
exams_collection = None
proctor_events_collection = None
exam_attempts_collection = None
proctoring_logs_collection = None

if not MONGO_URI:
    # In production (Render) MONGO_URI must be set. For local dev, allow the app
    # to boot so non-DB endpoints (health, etc.) can still respond.
    print('[DB] WARNING: MONGO_URI is not set. DB-backed endpoints will fail until it is provided.')
else:
    client = MongoClient(MONGO_URI)
    db = client['invigilo_db']
    users_collection = db['users']
    exams_collection = db['exams']
    proctor_events_collection = db['proctor_events']
    exam_attempts_collection = db['exam_attempts']
    proctoring_logs_collection = db['proctoring_logs']
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
settings_collection = db['settings'] if db is not None else None

# Helper to enforce DB availability inside endpoints.
def require_db():
    if db is None:
        return False, (jsonify({
            'error': 'db_not_configured',
            'message': 'Database not configured. Set MONGO_URI.'
        }), 500)
    return True, None

# Ensure important DB indexes for performance
try:
    if proctor_events_collection is None or exams_collection is None or users_collection is None:
        raise RuntimeError('db_not_configured')
    proctor_events_collection.create_index([('examId', 1), ('timestamp', -1)])
    proctor_events_collection.create_index([('examId', 1), ('userId', 1), ('timestamp', -1)])
    exams_collection.create_index('completedBy')
    users_collection.create_index('role')
    users_collection.create_index('email', unique=False)
    users_collection.create_index('phoneNumber', unique=False)
    users_collection.create_index('studentId', sparse=True)
    users_collection.create_index('lecturerId', sparse=True)
    # Privacy/retention: expire proctoring events after a configurable window (default 30 days)
    try:
        ttl_days = int(os.getenv('PROCTOR_EVENT_TTL_DAYS', '30'))
        if ttl_days > 0:
            proctor_events_collection.create_index('timestamp', expireAfterSeconds=ttl_days * 24 * 3600)
    except Exception as _e:
        print(f"TTL index setup warning: {_e}")
except Exception as e:
    print(f"Index creation warning: {e}")

# --- Helpers ---
def serialize_doc(doc):
    if doc and '_id' in doc:
        doc['_id'] = str(doc['_id'])
    if doc and 'questions' in doc:
        for q in doc.get('questions', []):
            if q and '_id' in q:
                q['_id'] = str(q['_id'])
    return doc

def decode_base64_image(data_url):
    """Decode base64-encoded image from frontend"""
    try:
        header, encoded = data_url.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(img_bytes))
        # Convert to a BGR numpy array.
        # Prefer OpenCV when available; otherwise use numpy channel flip.
        rgb = np.array(image.convert('RGB'))
        if CV2_AVAILABLE and cv2 is not None:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return rgb[:, :, ::-1]
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None

# --- ROUTES ---

# ✅ REGISTER USER + FACE ENROLLMENT
@app.route('/register', methods=['POST', 'OPTIONS'])
@app.route('/api/register', methods=['POST'])
#@limiter.limit("5 per hour")
def register_user():
    def _impl():
        print('[REGISTER] Received registration request')
        db_ok, db_err = require_db()
        if not db_ok:
            print('[REGISTER] ERROR: DB not configured (MONGO_URI missing)')
            return db_err

        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict) or not data:
            return jsonify({
                "error": "invalid_json",
                "message": "Request body must be JSON (Content-Type: application/json)."
            }), 400
        print(f'[REGISTER] User: {data.get("fullName")}, Email: {data.get("email")}, Role: {data.get("role")}')

        # imageDataUrl (single) OR imageDataUrls (list) is required for face enrollment
        required = ['fullName', 'email', 'phoneNumber', 'roleId', 'password', 'role', 'institution', 'department']
        if not all(field in data for field in required):
            missing = [f for f in required if f not in data]
            print(f'[REGISTER] ERROR: Missing required fields: {missing}')
            return jsonify({"error": "Missing required fields"}), 400

        # Validate password strength
        is_valid, error_message = validate_password(data.get('password', ''))
        if not is_valid:
            print(f'[REGISTER] ERROR: Password validation failed: {error_message}')
            return jsonify({
                "error": "Invalid password",
                "message": error_message,
                "requirements": [
                    "At least 8 characters long",
                    "At least one uppercase letter (A-Z)",
                    "At least one lowercase letter (a-z)",
                    "At least one digit (0-9)",
                    "At least one special character (!@#$%^&*-_=+)"
                ]
            }), 400

        existing = {"$or": [{"email": data['email']}, {"phoneNumber": data['phoneNumber']}]}
        if data['role'] == 'student':
            existing["$or"].append({"studentId": data['roleId']})
        else:
            existing["$or"].append({"lecturerId": data['roleId']})
        if users_collection.find_one(existing):
            print(f'[REGISTER] ERROR: User already exists: {data.get("email")}')
            return jsonify({"error": "User already exists"}), 409

        # --- Decode and process face image (REQUIRED for registration) ---
        images: list = []
        if data.get('imageDataUrls') and isinstance(data.get('imageDataUrls'), list):
            print(f'[REGISTER] Processing {len(data["imageDataUrls"])} face images')
            for d in data['imageDataUrls']:
                img = decode_base64_image(d)
                if img is not None:
                    images.append(img)
        elif data.get('imageDataUrl'):
            print(f'[REGISTER] Processing 1 face image')
            img = decode_base64_image(data['imageDataUrl'])
            if img is not None:
                images.append(img)

        if not images:
            print('[REGISTER] ERROR: No valid face images provided')
            return jsonify({"error": "Face image required for registration"}), 400

        print(f'[REGISTER] Successfully decoded {len(images)} face image(s)')

        try:
            # Generate face embeddings using ML Service (Hugging Face Spaces)
            face_vectors = []

            print('[REGISTER] Calling ML service for face verification')
            for i, img in enumerate(images):
                print(f'[REGISTER] Processing image {i+1}/{len(images)}')

                # Convert image to base64 for transmission
                _, buffer = cv2.imencode('.jpg', img) if CV2_AVAILABLE else (None, None)
                if buffer is None:
                    # Fallback if CV2 not available
                    pil_img = Image.fromarray(img[:, :, ::-1])  # BGR to RGB
                    img_buffer = io.BytesIO()
                    pil_img.save(img_buffer, format='JPEG')
                    img_bytes = img_buffer.getvalue()
                else:
                    img_bytes = buffer.tobytes()

                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                image_data_url = f"data:image/jpeg;base64,{img_b64}"

                # Call ML service
                success, result = call_ml_service('/verify-face', {
                    'imageDataUrl': image_data_url
                })

                if success and result.get('face_detected'):
                    embedding = result.get('embedding')
                    if embedding:
                        face_vectors.append(embedding)
                        print(f'[REGISTER] Successfully generated embedding {i+1}, dimension: {len(embedding)}')
                    else:
                        print(f'[REGISTER] No embedding returned for image {i+1}')
                else:
                    error_msg = result.get('error', 'Unknown error')
                    print(f'[REGISTER] ML service error for image {i+1}: {error_msg}')

            if not face_vectors:
                print('[REGISTER] ERROR: No face embeddings could be generated')
                return jsonify({
                    "error": "No face detected in uploaded images",
                    "message": "Please ensure your face is clearly visible and try again"
                }), 400

            print(f'[REGISTER] Generated {len(face_vectors)} embeddings from {len(images)} images')

            # Compute average embedding for improved robustness
            face_embedding_avg = None
            if face_vectors:
                try:
                    # Convert to numpy arrays and compute mean
                    embeddings_array = np.array(face_vectors)
                    face_embedding_avg = np.mean(embeddings_array, axis=0).tolist()
                    print(f'[REGISTER] Computed average embedding from {len(face_vectors)} samples')
                except Exception as e:
                    print(f'[REGISTER] WARNING: Failed to compute average embedding: {e}')
                    face_embedding_avg = face_vectors[0]  # Fallback to first embedding

            hashed_pw = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())

            new_user = {
                "name": data['fullName'],
                "email": data['email'],
                "phoneNumber": data['phoneNumber'],
                "role": data['role'],
                "password": hashed_pw,
                "institution": data['institution'],
                "department": data['department'],
                # Backcompat: keep the first embedding in faceEmbedding
                "faceEmbedding": (face_vectors[0] if face_vectors else None),
                # Store all individual embeddings
                "faceEmbeddings": face_vectors,
                # Store averaged embedding for improved robustness
                "faceEmbeddingAvg": face_embedding_avg,
                "faceVerified": bool(face_vectors),
                "isActive": True,
                "createdAt": datetime.datetime.utcnow()
            }
            if data['role'] == 'student':
                new_user['studentId'] = data['roleId']
                new_user['year'] = data.get('year')
            else:
                new_user['lecturerId'] = data['roleId']

            result = users_collection.insert_one(new_user)
            new_id = str(result.inserted_id)
            new_user['_id'] = new_id

            print(f'[REGISTER] User registered successfully: {new_id}, faceVerified: True')

            return jsonify({
                "message": "User registered successfully with face verification!",
                "userId": new_id
            }), 201

        except ValueError as e:
            print(f'[REGISTER] ValueError: {e}')
            return jsonify({"error": "No or multiple faces detected"}), 400

    try:
        return _impl()
    except Exception as e:
        print(f"[REGISTER] Unhandled registration error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "internal_server_error",
            "detail": str(e)
        }), 500

# ✅ LOGIN
@app.route('/login', methods=['POST', 'OPTIONS'])
@app.route('/api/login', methods=['POST'])
#@limiter.limit("10 per hour")
def login_user():
    try:
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict) or not data:
            return jsonify({
                "error": "invalid_json",
                "message": "Request body must be JSON (Content-Type: application/json)."
            }), 400
        identifier, password, role = data.get('identifier'), data.get('password'), data.get('role')
        if not all([identifier, password, role]):
            return jsonify({"error": "Missing fields"}), 400

        user = users_collection.find_one({
            "role": role,
            "$or": [{"email": identifier}, {"phoneNumber": identifier},
                    {"studentId": identifier}, {"lecturerId": identifier}]
        })

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            raw_user_id = str(user.get('_id'))
            user = serialize_doc(user)
            user = sanitize_user_response(user)  # Remove all sensitive fields

            try:
                ttl = int(os.getenv('AUTH_TOKEN_TTL_S', '43200'))  # 12h default
            except Exception:
                ttl = 43200
            token = _sign_token(raw_user_id, ttl_seconds=ttl)
            return jsonify({"message": "Login successful", "user": user, "token": token}), 200
        return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        print(f"[LOGIN] Unhandled login error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "internal_server_error",
            "detail": str(e)
        }), 500

# ✅ ANALYZE FRAME - Server-Side Violation Detection (CRITICAL)
@app.route('/api/analyze-frame', methods=['POST'])
@limiter.limit("100 per hour")
def analyze_frame():
    """
    CRITICAL: Server-side frame analysis for proctoring violation detection.
    
    Frontend validation can be bypassed - this endpoint provides authoritative violation detection.
    Analyzes a single frame for multiple violations: face count, identity verification,
    gaze direction, mouth movement, and head pose.
    
    Request Body:
        examId (str): Exam identifier
        userId (str): User identifier taking the exam
        imageDataUrl (str): Base64-encoded image data (data:image/jpeg;base64,...)
    
    Returns:
        200 OK: {violations: [...], score: int, status: "ok"}
        400 Bad Request: Missing parameters or invalid image
        404 Not Found: User or exam not found
        500 Internal Error: Processing failed
    """
    try:
        # 1. Validate input parameters
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        exam_id = data.get('examId')
        user_id = data.get('userId')
        image_data_url = data.get('imageDataUrl')
        
        if not all([exam_id, user_id, image_data_url]):
            return jsonify({"error": "examId, userId, and imageDataUrl are required"}), 400
        
        # 2. Validate exam exists
        try:
            exam = exams_collection.find_one({'_id': ObjectId(exam_id)}) if ObjectId.is_valid(exam_id) else None
            if not exam:
                return jsonify({"error": "Exam not found"}), 404
        except Exception as e:
            return jsonify({"error": "Invalid exam ID format"}), 400
        
        # 3. Validate user exists and is taking this exam
        try:
            user = users_collection.find_one({'_id': ObjectId(user_id)}) if ObjectId.is_valid(user_id) else None
            if not user:
                return jsonify({"error": "User not found"}), 404
        except Exception as e:
            return jsonify({"error": "Invalid user ID format"}), 400
        
        # 4. Get user's stored embeddings for face verification
        stored_embeddings = user.get('faceEmbeddings', []) or []
        if not stored_embeddings and user.get('faceEmbedding'):
            stored_embeddings = [user.get('faceEmbedding')]
        
        # 5. Convert image to base64 for ML service
        try:
            # Remove data URL prefix if present
            if ',' in image_data_url:
                image_base64 = image_data_url.split(',', 1)[1]
            else:
                image_base64 = image_data_url
        except Exception as e:
            app.logger.error(f"Image processing error: {e}")
            return jsonify({"error": "Invalid image data format"}), 400
        
        # 6. Call ML service for comprehensive frame analysis (HF expects imageDataUrl)
        ml_payload = {
            'imageDataUrl': image_data_url
        }

        ok_ml, ml_result = call_ml_service('/analyze-frame', ml_payload, timeout=30)

        if not ok_ml or not isinstance(ml_result, dict) or 'violations' not in ml_result:
            app.logger.error(f"ML service returned invalid response: {ml_result}")
            return jsonify({"error": "ML service failed to analyze frame", "detail": (ml_result or {}).get('error') if isinstance(ml_result, dict) else str(ml_result)}), 502
        
        violations = ml_result['violations']
        face_count = ml_result.get('faceCount', ml_result.get('face_count', 0))
        total_score = ml_result.get('score', 0)
        
        # 7. Store violations in database
        timestamp = datetime.datetime.utcnow()
        for violation in violations:
            try:
                proctor_events_collection.insert_one({
                    'examId': exam_id,
                    'userId': user_id,
                    'type': 'violation',
                    'violationType': violation['type'],
                    'severity': violation.get('severity', 'unknown'),
                    'score': violation.get('score', 0),
                    'message': violation.get('message', ''),
                    'timestamp': timestamp
                })
                
                # ✅ Broadcast violation to proctors via WebSocket in real-time
                broadcast_violation(exam_id, user_id, violation)
                
            except Exception as e:
                app.logger.error(f"Failed to store violation: {e}")
        
        # 8. Return analysis results
        return jsonify({
            "violations": violations,
            "score": total_score,
            "status": "ok",
            "faceCount": face_count
        }), 200
        
    except Exception as e:
        app.logger.exception("Error in analyze_frame")
        return jsonify({"error": "Internal processing error", "detail": str(e)}), 500

# Helper: face threshold from settings or env
def get_face_threshold(
    default_val=0.58,
    *,
    key: str = 'FACE_SIMILARITY_THRESHOLD',
    env_var: str = 'FACE_SIMILARITY_THRESHOLD'
):
    try:
        cfg = settings_collection.find_one({'key': key})
        if cfg and isinstance(cfg.get('value'), (int, float)):
            return float(cfg['value'])
    except Exception:
        pass
    try:
        return float(os.getenv(env_var, str(default_val)))
    except Exception:
        return float(default_val)


def _cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two embedding vectors.

    Returns None if vectors are invalid.
    """
    try:
        a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
        b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
        if a.size == 0 or b.size == 0 or a.shape != b.shape:
            return None
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return None
        return float(np.dot(a, b) / denom)
    except Exception:
        return None


def _norm_head_pose(v):
    s = str(v or '').strip().lower().replace(' ', '-').replace('_', '-')
    if not s or s in ('unknown', 'n/a', 'na', 'none'):
        return 'unknown'
    if s.startswith('head-'):
        s = s[len('head-'):]
    if s in ('forward', 'center', 'centred', 'centered', 'normal', 'straight'):
        return 'forward'
    has_down = 'down' in s
    has_up = 'up' in s
    has_left = 'left' in s
    has_right = 'right' in s
    if has_down and has_left:
        return 'down_left'
    if has_down and has_right:
        return 'down_right'
    if has_up and has_left:
        return 'up_left'
    if has_up and has_right:
        return 'up_right'
    if has_down:
        return 'down'
    if has_up:
        return 'up'
    if has_left:
        return 'left'
    if has_right:
        return 'right'
    return 'unknown'


def _norm_gaze(v):
    s = str(v or '').strip().lower().replace(' ', '-').replace('_', '-')
    if not s or s in ('unknown', 'n/a', 'na', 'none'):
        return 'unknown'
    if s in ('center', 'centre', 'forward', 'straight', 'normal'):
        return 'center'
    if 'left' in s:
        return 'left'
    if 'right' in s:
        return 'right'
    if 'up' in s:
        return 'up'
    if 'down' in s:
        return 'down'
    return 'unknown'


def _decay_score(score: float, dt_seconds: float, half_life_seconds: float = 300.0) -> float:
    """Exponential decay with a configurable half-life (default 5 minutes)."""
    try:
        if score <= 0:
            return 0.0
        if dt_seconds <= 0:
            return float(score)
        # score * 0.5^(dt/half_life)
        return float(score) * (0.5 ** (float(dt_seconds) / float(half_life_seconds)))
    except Exception:
        return float(score)


def _sum_flag_duration(samples, now_ts: float, window_seconds: float, flag_index: int = 1) -> float:
    """Approximate total time a boolean flag was true in a rolling window.

    samples: iterable of tuples (ts, flag, ...)
    flag_index: tuple index of boolean flag
    """
    try:
        if not samples:
            return 0.0
        start_ts = now_ts - float(window_seconds)
        # Ensure time-ordered
        items = list(samples)
        items.sort(key=lambda x: x[0])

        # Trim to window (keep one sample before start for continuity if present)
        trimmed = []
        prev = None
        for it in items:
            if it[0] < start_ts:
                prev = it
                continue
            if prev is not None:
                trimmed.append(prev)
                prev = None
            trimmed.append(it)
        if not trimmed:
            return 0.0

        total = 0.0
        for i in range(len(trimmed)):
            t0 = max(float(trimmed[i][0]), start_ts)
            t1 = float(trimmed[i + 1][0]) if i + 1 < len(trimmed) else float(now_ts)
            if t1 <= start_ts:
                continue
            if bool(trimmed[i][flag_index]):
                total += max(0.0, t1 - t0)
        return float(total)
    except Exception:
        return 0.0

# ✅ FACE VERIFICATION
@app.route('/api/verify-face', methods=['POST'])
#@limiter.limit("20 per hour")
def verify_face():
    print('[FACE-VERIFY] Received face verification request')
    data = request.get_json(silent=True) or {}
    identifier = data.get('identifier')
    role = data.get('role')
    image_data_url = data.get('imageDataUrl')
    
    print(f'[FACE-VERIFY] Identifier: {identifier}, Role: {role}')
    print(f'[FACE-VERIFY] Image data URL length: {len(image_data_url) if image_data_url else 0}')

    if not identifier or not image_data_url or not role:
        print('[FACE-VERIFY] ERROR: Missing parameters')
        return jsonify({"error": "Missing parameters"}), 400

    user = users_collection.find_one({
        "role": role,
        "$or": [{"email": identifier}, {"phoneNumber": identifier},
                {"studentId": identifier}, {"lecturerId": identifier}]
    })
    if not user:
        print(f'[FACE-VERIFY] ERROR: User not found for identifier {identifier}')
        return jsonify({"error": "User not found or no face data"}), 404
        
    if 'faceEmbedding' not in user and 'faceEmbeddings' not in user:
        print(f'[FACE-VERIFY] ERROR: No face embedding stored for user {identifier}')
        return jsonify({"error": "User not found or no face data"}), 404
    
    print(f'[FACE-VERIFY] User found: {user.get("name")}, has embeddings: {bool(user.get("faceEmbedding") or user.get("faceEmbeddings"))}')

    try:
        # Normalize to data URL (HF ML service expects imageDataUrl)
        normalized_image_data_url = image_data_url
        if isinstance(normalized_image_data_url, str) and not normalized_image_data_url.startswith('data:'):
            normalized_image_data_url = f"data:image/jpeg;base64,{normalized_image_data_url}"

        # Optional: shrink the image before sending to ML service to reduce payload and latency.
        # Defaults are chosen to keep accuracy while improving speed.
        try:
            max_dim = int(os.getenv('FACE_VERIFY_MAX_DIM', '640'))
        except Exception:
            max_dim = 640
        try:
            jpeg_quality = int(os.getenv('FACE_VERIFY_JPEG_QUALITY', '85'))
        except Exception:
            jpeg_quality = 85

        if max_dim and max_dim > 0:
            try:
                import base64
                import io
                from PIL import Image

                img_data = normalized_image_data_url.split(',')[1] if ',' in normalized_image_data_url else normalized_image_data_url
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                w, h = img.size
                if max(w, h) > max_dim:
                    scale = float(max_dim) / float(max(w, h))
                    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                    img = img.resize(new_size, Image.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=max(30, min(95, jpeg_quality)))
                resized_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                normalized_image_data_url = f"data:image/jpeg;base64,{resized_b64}"
            except Exception as e:
                print(f'[FACE-VERIFY] Resize/encode skipped: {e}')
        
        # Get stored embeddings - prioritize averaged embedding if available
        stored_embeddings = []
        if user.get('faceEmbeddingAvg'):
            # Use averaged embedding first for better robustness
            stored_embeddings.append(user.get('faceEmbeddingAvg'))
            print('[FACE-VERIFY] Using averaged embedding for verification')
        if user.get('faceEmbeddings') and isinstance(user.get('faceEmbeddings'), list):
            stored_embeddings.extend([e for e in user['faceEmbeddings'] if isinstance(e, (list, tuple))])
        if not stored_embeddings and user.get('faceEmbedding'):
            stored_embeddings = [user.get('faceEmbedding')]
        
        if not stored_embeddings:
            print('[FACE-VERIFY] ERROR: No stored embeddings found')
            return jsonify({"error": "No face data stored for user"}), 404
        
        # Use a login-specific threshold (lower than proctoring by default).
        # This reduces false rejections at login without affecting in-exam proctor identity checks.
        THRESHOLD = get_face_threshold(
            default_val=0.55,
            key='FACE_SIMILARITY_THRESHOLD_LOGIN',
            env_var='FACE_SIMILARITY_THRESHOLD_LOGIN'
        )

        def _best_similarities_for_embedding(new_emb):
            sims_local = []
            for stored in stored_embeddings:
                sim = _cosine_similarity(new_emb, stored)
                if sim is not None:
                    sims_local.append(sim)
            return sims_local

        # 1) Generate embedding from current image.
        ok_verify, verify_result = call_ml_service('/verify-face', {
            'imageDataUrl': normalized_image_data_url
        }, timeout=8)

        embeddings_to_try = []
        if ok_verify and isinstance(verify_result, dict) and 'embedding' in verify_result:
            embeddings_to_try.append(verify_result['embedding'])
            print('[FACE-VERIFY] Generated embedding from original image')

        # 2) If first attempt fails threshold, optionally retry with a simple brightness variant.
        # This addresses intermittent failures due to under/overexposure and camera auto-gain.
        enable_retry = _bool_env('FACE_VERIFY_ENABLE_RETRY', '1')
        sims = []
        if embeddings_to_try:
            sims = _best_similarities_for_embedding(embeddings_to_try[0])
            if sims and max(sims) >= THRESHOLD:
                max_sim = max(sims)
                print(f'[FACE-VERIFY] Face verify for {identifier}: 1 variant, {len(stored_embeddings)} stored, max={max_sim:.3f} thr={THRESHOLD}')
                return jsonify({
                    "message": "Face verified successfully",
                    "verified": True,
                    "similarity": float(max_sim),
                    "similarities": sims,
                    "threshold": float(THRESHOLD)
                }), 200

            if not enable_retry:
                max_sim = max(sims) if sims else None
                return jsonify({
                    "message": "Face verification failed",
                    "verified": False,
                    "similarity": float(max_sim) if max_sim is not None else None,
                    "similarities": sims,
                    "threshold": float(THRESHOLD)
                }), 401

        # Prepare brightness-enhanced variant (even if original embed exists but was low).
        try:
            import base64
            import io
            from PIL import Image, ImageEnhance

            img_data = normalized_image_data_url.split(',')[1] if ',' in normalized_image_data_url else normalized_image_data_url
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))

            enhancer = ImageEnhance.Brightness(img)
            bright_img = enhancer.enhance(1.25)

            buffer = io.BytesIO()
            bright_img.save(buffer, format='JPEG', quality=95)
            bright_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            bright_url = f"data:image/jpeg;base64,{bright_b64}"

            ok_bright, bright_result = call_ml_service('/verify-face', {
                'imageDataUrl': bright_url
            }, timeout=8)

            if ok_bright and isinstance(bright_result, dict) and 'embedding' in bright_result:
                embeddings_to_try.append(bright_result['embedding'])
                print('[FACE-VERIFY] Generated embedding from brightness-enhanced image')
        except Exception as e:
            print(f'[FACE-VERIFY] Preprocessing failed: {e}')

        if not embeddings_to_try:
            detail = (verify_result or {}).get('error') if isinstance(verify_result, dict) else str(verify_result)
            print(f'[FACE-VERIFY] ERROR: Failed to generate any embeddings: {detail}')
            return jsonify({
                "error": "Face verification failed",
                "detail": "Could not detect face in image",
                "verified": False,
                "threshold": float(THRESHOLD)
            }), 401

        # 3) Match against all stored embeddings using all generated embeddings
        all_similarities = []
        for new_emb in embeddings_to_try:
            all_similarities.extend(_best_similarities_for_embedding(new_emb))

        if not all_similarities:
            print('[FACE-VERIFY] ERROR: No similarities computed (invalid embeddings)')
            return jsonify({
                "error": "Failed to verify face",
                "detail": "No valid stored embeddings to compare",
                "threshold": float(THRESHOLD)
            }), 400

        max_sim = max(all_similarities)
        print(f'[FACE-VERIFY] Face verify for {identifier}: {len(embeddings_to_try)} variants, {len(stored_embeddings)} stored, similarities={all_similarities[:5]}..., max={max_sim:.3f} thr={THRESHOLD}')

        if max_sim >= THRESHOLD:
            return jsonify({
                "message": "Face verified successfully",
                "verified": True,
                "similarity": float(max_sim),
                "similarities": all_similarities,
                "threshold": float(THRESHOLD)
            }), 200

        return jsonify({
            "message": "Face verification failed",
            "verified": False,
            "similarity": float(max_sim),
            "similarities": all_similarities,
            "threshold": float(THRESHOLD)
        }), 401

    except ValueError:
        return jsonify({"error": "No face detected"}), 400
    except Exception as e:
        app.logger.exception('Verification error')
        return jsonify({"error": "Internal verification error", "detail": str(e)}), 500

# --- (Keep all your other routes exactly as before) ---
# exams, proctoring, ai_generate_questions, etc. remain unchanged.



@app.route('/api/proctor', methods=['POST'])
# Proctoring runs at ~4 FPS; 100/hour would break normal operation.
@limiter.limit("20000 per hour")
def proctor_activity():
    """
    Proctoring frame analysis endpoint (synchronous).
    Processes the frame immediately and returns results.
    """
    data = request.get_json()
    image_data_url = data.get('imageDataUrl')
    user_id = data.get('userId')
    exam_id = str(data.get('examId') or '')
    exam_active = data.get('examActive', True)  # True by default for backward compatibility

    if not image_data_url or not user_id:
        return jsonify({"error": "Image data and User ID are required"}), 400

    frame = decode_base64_image(image_data_url)
    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400
    
    # Convert frame to base64 for ML service
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"[PROCTOR] Error encoding frame: {e}")
        return jsonify({"error": "Failed to encode frame"}), 500
    
    # Async/Celery mode removed: processing is always synchronous
    
    # ============================================================================
    # SYNCHRONOUS MODE (ORIGINAL FLOW - BACKWARD COMPATIBLE)
    # ============================================================================
    
    # NOTE: Proctoring is sampling-based; avoid single-frame punishments.
    # We'll track sustained darkness/absence over a rolling time window.
    try:
        mean_brightness = float(np.mean(frame))
    except Exception:
        mean_brightness = None

    # If the exam is inactive (submission/exit), do not treat blank frames as suspicious.
    if mean_brightness is not None and mean_brightness < 10 and not exam_active:
        print(f"[PROCTOR] Skipping blank frame - exam inactive (brightness: {mean_brightness:.2f})")
        return jsonify({
            "faceCount": 0,
            "identityVerified": False,
            "similarity": None,
            "blinkStatus": "Unknown",
            "gazeDirection": "Unknown",
            "mouthStatus": "Unknown",
            "headPose": "Unknown",
            "message": "Blank frame - exam inactive"
        }), 200

    # Call ML service for full proctoring analysis.
    # If the frame is nearly black, skip ML call (cheap + avoids noise); treat as 0-face sample.
    blank_like = bool(mean_brightness is not None and mean_brightness < 10)
    try:
        # Standardized payload format: imageDataUrl with data URL prefix
        ml_payload = {
            'imageDataUrl': f"data:image/jpeg;base64,{frame_base64}"
        }
        if blank_like:
            ok_ml, ml_result = True, {}
            face_count = 0
            blink_status = 'Unknown'
            gaze_direction = 'Unknown'
            mouth_status = 'Unknown'
            head_pose = 'Unknown'
        else:
            ok_ml, ml_result = call_ml_service('/analyze-frame', ml_payload, timeout=30)

            if not ok_ml or not isinstance(ml_result, dict):
                print("[PROCTOR] ML service unavailable or returned invalid response")
                return jsonify({
                    "faceCount": 0,
                    "identityVerified": False,
                    "similarity": None,
                    "blinkStatus": "Unknown",
                    "gazeDirection": "Unknown",
                    "mouthStatus": "Unknown",
                    "headPose": "Unknown",
                    "message": "ML service unavailable"
                }), 200

            # If the ML service is running in degraded mode (e.g., missing native deps on HF),
            # do not treat this as suspicious behavior; just return Unknowns and skip eventing.
            try:
                degraded = (ml_result.get('proctoring_available') is False) or (ml_result.get('proctoringAvailable') is False)
            except Exception:
                degraded = False
            if degraded:
                return jsonify({
                    "faceCount": None,
                    "identityVerified": False,
                    "similarity": None,
                    "blinkStatus": "Unknown",
                    "gazeDirection": "Unknown",
                    "mouthStatus": "Unknown",
                    "headPose": "Unknown",
                    "proctoringAvailable": False,
                    "message": "ML proctoring unavailable (degraded mode)"
                }), 200

            # Extract ML results
            face_count = ml_result.get('face_count', ml_result.get('faceCount', 0))
            blink_status = ml_result.get('blink_status', ml_result.get('blinkStatus', 'Unknown'))
            gaze_direction = ml_result.get('gaze_direction', ml_result.get('gazeDirection', 'Unknown'))
            mouth_status = ml_result.get('mouth_status', ml_result.get('mouthStatus', 'Unknown'))
            head_pose = ml_result.get('head_pose', ml_result.get('headPose', 'Unknown'))

    except Exception as e:
        print(f"[PROCTOR] Error calling ML service: {e}")
        return jsonify({"error": "ML service error"}), 500

    # Identity verification is expensive (extra ML call). Do NOT run it every frame.
    # We perform it on an interval inside the temporal detection block and keep the
    # latest known value in per-session state.
    identity_verified = False
    similarity_score = None

    # Build response
    results = {
        "faceCount": face_count,
        "identityVerified": identity_verified,
        "similarity": similarity_score,
        "blinkStatus": blink_status,
        "gazeDirection": gaze_direction,
        "mouthStatus": mouth_status,
        "headPose": head_pose
    }
    # Timestamp-based temporal consensus + conservative eventing.
    try:
        if exam_id and user_id:
            now_dt = datetime.datetime.utcnow()
            now_ts = now_dt.timestamp()
            key = (str(exam_id), str(user_id))

            st = PROCTOR_STATE.get(key) or {
                'last_emit': {},
                'reference_background': None,
                'samples': deque(),
                'risk_score': 0.0,
                'risk_last_ts': None,
                'face_absent_since': None,
                'gaze_away_since': None,
                'head_turned_since': None,
                'mouth_open_since': None,
                'dark_since': None,
                'background_change_since': None,
                'suppress_camera_until': 0.0,
                'recent_activity': {},  # activityType -> ts (from /api/log-activity)
                'last_identity': {
                    'similarity': None,
                    'verified': False,
                    'threshold': None,
                    'checked_at': None,
                },
                'identity': {
                    'last_check_ts': 0.0,
                    'fail_count': 0,
                    'pass_count': 0,
                    'recent_sims': deque(),  # (ts, similarity)
                    'cached_embeddings': None,
                    'cached_embeddings_at': None,
                },
            }

            # Detect large gaps (slow network / paused capture) and reset volatile counters.
            try:
                last_sample_ts = st.get('last_sample_ts')
                # Client should sample ~every 3–5 seconds; treat only larger gaps as discontinuities.
                gap_reset_s = float(os.getenv('PROCTOR_DISCONTINUITY_RESET_S', '8'))
                if last_sample_ts and (now_ts - float(last_sample_ts)) > gap_reset_s:
                    # Treat as a discontinuity: don't punish camera issues immediately after resuming.
                    st['face_absent_since'] = None
                    st['gaze_away_since'] = None
                    st['head_turned_since'] = None
                    st['mouth_open_since'] = None
                    st['dark_since'] = None
                    st['background_change_since'] = None
                    st['suppress_camera_until'] = now_ts + 2.0
                    try:
                        st['samples'] = deque()
                    except Exception:
                        pass
            except Exception:
                pass
            st['last_sample_ts'] = now_ts

            # Store reference background on first good frame
            if st.get('reference_background') is None and frame is not None and (mean_brightness is None or mean_brightness >= 30):
                try:
                    ref_small = cv2.resize(frame, (160, 120))
                    st['reference_background'] = ref_small.copy()
                except Exception as e:
                    print(f"[PROCTOR] Error storing reference background: {e}")

            gaze_norm = _norm_gaze(gaze_direction)
            head_norm = _norm_head_pose(head_pose)
            mouth_open = bool(mouth_status and str(mouth_status).lower() in ('open', 'talking', 'speaking'))
            face_present = bool(face_count and int(face_count) > 0)

            # Brightness: prefer mean brightness (works even if HSV conversion fails)
            brightness_val = float(mean_brightness) if mean_brightness is not None else None

            # Rolling samples over 15 seconds
            WINDOW_S = 15.0
            try:
                st['samples'].append((now_ts, bool(face_present), bool(mouth_open), bool(head_norm not in ('forward', 'unknown')), bool(gaze_norm not in ('center', 'unknown')), brightness_val, gaze_norm, head_norm))
                cutoff = now_ts - WINDOW_S
                while st['samples'] and float(st['samples'][0][0]) < cutoff:
                    st['samples'].popleft()
            except Exception:
                # If deque fails for any reason, keep system functional
                st['samples'] = deque([(now_ts, bool(face_present), bool(mouth_open), bool(head_norm not in ('forward', 'unknown')), bool(gaze_norm not in ('center', 'unknown')), brightness_val, gaze_norm, head_norm)])

            # Update episode start times (timestamp-based)
            if not face_present:
                if st.get('face_absent_since') is None:
                    st['face_absent_since'] = now_ts
            else:
                st['face_absent_since'] = None

            gaze_away = bool(gaze_norm not in ('center', 'unknown'))
            if gaze_away:
                if st.get('gaze_away_since') is None:
                    st['gaze_away_since'] = now_ts
            else:
                st['gaze_away_since'] = None

            head_turned = bool(head_norm not in ('forward', 'unknown'))
            if head_turned:
                if st.get('head_turned_since') is None:
                    st['head_turned_since'] = now_ts
            else:
                st['head_turned_since'] = None

            if mouth_open:
                if st.get('mouth_open_since') is None:
                    st['mouth_open_since'] = now_ts
            else:
                st['mouth_open_since'] = None

            # Sustained darkness: require consensus of darkness + face absence.
            DARK_THR = 12.0
            is_dark = bool(brightness_val is not None and brightness_val < DARK_THR)
            if is_dark and (not face_present):
                if st.get('dark_since') is None:
                    st['dark_since'] = now_ts
            else:
                st['dark_since'] = None

            # Risk score decay update
            prev_ts = st.get('risk_last_ts')
            if prev_ts is not None:
                st['risk_score'] = _decay_score(float(st.get('risk_score') or 0.0), now_ts - float(prev_ts), half_life_seconds=300.0)
            st['risk_last_ts'] = now_ts

            # Attach standard metrics for debugging/tuning ("why" fields)
            def _build_metrics(extra=None):
                m = {
                    'faceCount': int(face_count) if isinstance(face_count, (int, float)) else face_count,
                    'similarity': float(similarity_score) if isinstance(similarity_score, (int, float)) else similarity_score,
                    'identityVerified': bool(identity_verified),
                    'blinkStatus': blink_status,
                    'gazeDirection': gaze_direction,
                    'mouthStatus': mouth_status,
                    'headPose': head_pose,
                    'gaze_norm': gaze_norm,
                    'head_norm': head_norm,
                    'brightness': brightness_val,
                }
                if isinstance(extra, dict):
                    for k, v in extra.items():
                        m[k] = v
                return m

            COOLDOWN_S = {
                'camera_blocked': 10.0,
                'multiple_faces': 10.0,
                'identity_mismatch': 15.0,
                'face_missing': 10.0,
                'background_change': 30.0,
                'gaze_sustained': 15.0,
                'gaze_extreme': 20.0,
                'gaze_frequency': 20.0,
                'head_pose': 15.0,
                'talking': 15.0,
                'default': 8.0,
            }
            RISK_DB_THRESHOLD = 140.0

            def emit_event(ev_type: str, details: dict, severity: str, risk_score: float = 0.0):
                # Event cooldowns (prevent DB spam and alert fatigue)
                last_time = st['last_emit'].get(ev_type)
                cd = float(COOLDOWN_S.get(ev_type, COOLDOWN_S['default']))
                if last_time and (now_dt - last_time).total_seconds() < cd:
                    return

                # Update rolling risk score (used for score-based decisioning)
                try:
                    st['risk_score'] = float(st.get('risk_score') or 0.0) + float(risk_score or 0.0)
                except Exception:
                    pass

                # Store only if severity >= medium OR aggregated score exceeded
                should_store = severity in ('medium', 'high', 'critical') or (float(st.get('risk_score') or 0.0) >= RISK_DB_THRESHOLD)
                # Emit only HIGH / CRITICAL events
                should_emit_socket = severity in ('high', 'critical')

                # Evidence frames are expensive + privacy-sensitive: keep only for high/critical.
                frame_evidence = None
                if severity in ('high', 'critical') and frame is not None:
                    try:
                        mb = float(np.mean(frame))
                        if mb >= 30:
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            fb64 = base64.b64encode(buffer).decode('utf-8')
                            frame_evidence = f"data:image/jpeg;base64,{fb64}"
                    except Exception:
                        frame_evidence = None

                metrics = _build_metrics(extra={'risk_total': float(st.get('risk_score') or 0.0)})
                if isinstance(details, dict):
                    for k in ('gaze_norm', 'head_norm', 'pose_norm'):
                        if k in details:
                            metrics[k] = details.get(k)

                if should_store and proctor_events_collection is not None:
                    try:
                        proctor_events_collection.insert_one({
                            'examId': str(exam_id),
                            'userId': str(user_id),
                            'eventType': ev_type,
                            'details': details,
                            'metrics': metrics,
                            'severity': severity,
                            'timestamp': now_dt,
                            'frameEvidence': frame_evidence
                        })
                    except Exception as e:
                        print(f"[PROCTOR] DB insert failed for {ev_type}: {e}")

                if should_emit_socket:
                    try:
                        socketio.emit(ev_type, {
                            'examId': str(exam_id),
                            'userId': str(user_id),
                            'eventType': ev_type,
                            'details': details,
                            'metrics': metrics,
                            'severity': severity,
                            'timestamp': now_dt.isoformat() + 'Z',
                            'message': (details or {}).get('message', ev_type),
                            'frameEvidence': frame_evidence,
                        }, room=str(exam_id), namespace='/proctor')
                    except Exception as e:
                        print(f"[PROCTOR-SOCKET] Error emitting {ev_type}: {e}")

                st['last_emit'][ev_type] = now_dt

            # =====================================================================
            # CONTEXT AWARENESS: if exam inactive, do not emit camera-related events.
            # =====================================================================
            if not exam_active:
                PROCTOR_STATE[key] = st
                return jsonify(results), 200

            # Compute rolling totals (10s window) for score-based evaluation
            total_gaze_away_10s = _sum_flag_duration(st['samples'], now_ts, window_seconds=10.0, flag_index=4)
            total_head_turned_10s = _sum_flag_duration(st['samples'], now_ts, window_seconds=10.0, flag_index=3)
            total_mouth_open_10s = _sum_flag_duration(st['samples'], now_ts, window_seconds=10.0, flag_index=2)

            # Continuous durations
            face_absent_dur = (now_ts - float(st['face_absent_since'])) if st.get('face_absent_since') else 0.0
            gaze_away_dur = (now_ts - float(st['gaze_away_since'])) if st.get('gaze_away_since') else 0.0
            head_turned_dur = (now_ts - float(st['head_turned_since'])) if st.get('head_turned_since') else 0.0
            mouth_open_dur = (now_ts - float(st['mouth_open_since'])) if st.get('mouth_open_since') else 0.0
            dark_dur = (now_ts - float(st['dark_since'])) if st.get('dark_since') else 0.0

            # ========================================
            # MULTIPLE FACES (HIGH)
            # ========================================
            if face_count and int(face_count) > 1:
                emit_event('multiple_faces', {
                    'count': int(face_count),
                    'message': f'{int(face_count)} people detected in frame',
                    'risk_score': 95,
                }, 'high', risk_score=95)

            # ========================================
            # CAMERA BLOCKED (HIGH; CRITICAL if paired with fullscreen_exit)
            # ========================================
            if now_ts >= float(st.get('suppress_camera_until') or 0.0):
                # Require sustained darkness + face absence consensus for >=3s
                if st.get('dark_since') and dark_dur >= 3.0:
                    # Require at least 3 samples in the last ~3.5s meeting consensus
                    recent = [x for x in list(st['samples']) if float(x[0]) >= now_ts - 3.5]
                    dark_votes = 0
                    for x in recent:
                        b = x[5]
                        if (b is not None and float(b) < DARK_THR) and (not bool(x[1])):
                            dark_votes += 1
                    if len(recent) >= 3 and (dark_votes / max(1, len(recent))) >= 0.7:
                        sev = 'high'
                        # Escalate to critical if a fullscreen_exit occurred very recently
                        try:
                            fe_ts = float(st.get('recent_activity', {}).get('fullscreen_exit') or 0.0)
                            if fe_ts and (now_ts - fe_ts) <= 10.0:
                                sev = 'critical'
                        except Exception:
                            pass

                        emit_event('camera_blocked', {
                            'brightness': brightness_val,
                            'duration_seconds': float(dark_dur),
                            'message': 'Camera appears blocked/covered (sustained darkness + face absence)',
                            'risk_score': 95 if sev in ('high', 'critical') else 80,
                        }, sev, risk_score=95 if sev in ('high', 'critical') else 80)

            # ========================================
            # FACE MISSING (MEDIUM/HIGH) - long absence only
            # ========================================
            if now_ts >= float(st.get('suppress_camera_until') or 0.0):
                if face_absent_dur >= 10.0:
                    emit_event('face_missing', {
                        'message': 'Face not detected for an extended period',
                        'duration_seconds': float(face_absent_dur),
                        'risk_score': 70,
                    }, 'high', risk_score=70)
                elif face_absent_dur >= 6.0:
                    emit_event('face_missing', {
                        'message': 'Face not detected (sustained)',
                        'duration_seconds': float(face_absent_dur),
                        'risk_score': 55,
                    }, 'medium', risk_score=55)

            # ========================================
            # TALKING (MEDIUM) - require sustained mouth activity
            # ========================================
            # Ignore natural mouth movements; require sustained activity.
            if mouth_open_dur >= 4.0 or total_mouth_open_10s >= 6.0:
                emit_event('talking', {
                    'status': mouth_status,
                    'message': 'Sustained mouth movement detected',
                    'duration_seconds': float(max(mouth_open_dur, total_mouth_open_10s)),
                    'risk_score': 45,
                }, 'medium', risk_score=45)

            # ========================================
            # GAZE + HEAD POSE (LOW/MEDIUM) - conservative
            # ========================================
            # Short glance away (<2s) => ignore
            # Repeated gaze away (>5s total in 10s) => low (do not store unless risk threshold exceeded)
            # Long continuous away (>=6s) => medium
            if gaze_away_dur >= 6.0:
                emit_event('gaze_sustained', {
                    'direction': gaze_direction,
                    'direction_norm': gaze_norm,
                    'message': f'Sustained looking away ({gaze_direction})',
                    'duration_seconds': float(gaze_away_dur),
                    'risk_score': 55,
                }, 'medium', risk_score=55)
            elif total_gaze_away_10s >= 5.0:
                emit_event('gaze_frequency', {
                    'direction': gaze_direction,
                    'direction_norm': gaze_norm,
                    'message': 'Repeated gaze aversion detected (cumulative in window)',
                    'window_seconds': 10.0,
                    'away_total_seconds': float(total_gaze_away_10s),
                    'risk_score': 20,
                }, 'low', risk_score=20)

            # Head pose: ignore brief turns; medium only for sustained down/away posture
            if head_turned_dur >= 6.0 and head_norm in ('down', 'down_left', 'down_right'):
                emit_event('head_pose', {
                    'pose': head_pose,
                    'pose_norm': head_norm,
                    'message': f'Sustained head-down posture ({head_pose})',
                    'duration_seconds': float(head_turned_dur),
                    'risk_score': 45,
                }, 'medium', risk_score=45)
            elif total_head_turned_10s >= 6.0:
                emit_event('head_pose', {
                    'pose': head_pose,
                    'pose_norm': head_norm,
                    'message': 'Repeated head turns detected (cumulative in window)',
                    'window_seconds': 10.0,
                    'turned_total_seconds': float(total_head_turned_10s),
                    'risk_score': 15,
                }, 'low', risk_score=15)

            # ========================================
            # BACKGROUND CHANGE (SOFTENED) - persistence required
            # ========================================
            if st.get('reference_background') is not None and frame is not None and (mean_brightness is None or mean_brightness >= 30):
                try:
                    current_small = cv2.resize(frame, (160, 120))
                    diff = cv2.absdiff(st['reference_background'], current_small)
                    diff_score = float(np.mean(diff))
                    try:
                        ref_gray = cv2.cvtColor(st['reference_background'], cv2.COLOR_BGR2GRAY)
                        cur_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
                        lighting_delta = float(abs(np.mean(cur_gray) - np.mean(ref_gray)))
                    except Exception:
                        lighting_delta = 0.0

                    BACKGROUND_THRESHOLD = 35.0
                    if diff_score > BACKGROUND_THRESHOLD and lighting_delta < 25.0:
                        if st.get('background_change_since') is None:
                            st['background_change_since'] = now_ts
                        bg_dur = now_ts - float(st.get('background_change_since') or now_ts)
                        if bg_dur >= 5.0:
                            emit_event('background_change', {
                                'change_score': float(diff_score),
                                'lighting_delta': float(lighting_delta),
                                'message': 'Sustained background change detected',
                                'duration_seconds': float(bg_dur),
                                'risk_score': 55,
                            }, 'medium', risk_score=55)
                    else:
                        st['background_change_since'] = None
                except Exception as e:
                    print(f"[PROCTOR] Error in background detection: {e}")

            # ========================================
            # IDENTITY MISMATCH (HIGH) - stable, interval-based
            # ========================================
            ident = st.get('identity') or {}
            last_check = float(ident.get('last_check_ts') or 0.0)
            head_turned_now = bool(head_norm not in ('forward', 'unknown'))
            if (
                (now_ts - last_check) >= 2.5 and
                (not head_turned_now) and
                int(face_count or 0) == 1 and
                face_absent_dur < 6.0
            ):
                ident['last_check_ts'] = now_ts
                identity_threshold = float(get_face_threshold())

                # Cache stored embeddings per session to reduce DB load and improve determinism.
                stored_embeddings = ident.get('cached_embeddings')
                try:
                    cached_at = float(ident.get('cached_embeddings_at') or 0.0)
                except Exception:
                    cached_at = 0.0

                if not stored_embeddings or (now_ts - cached_at) > 60.0:
                    stored_embeddings = []
                    try:
                        user = users_collection.find_one({'_id': ObjectId(user_id)}) if users_collection is not None else None
                        if user:
                            if user.get('faceEmbeddingAvg') and isinstance(user.get('faceEmbeddingAvg'), (list, tuple)):
                                stored_embeddings.append(user.get('faceEmbeddingAvg'))
                            if user.get('faceEmbeddings') and isinstance(user.get('faceEmbeddings'), list):
                                stored_embeddings.extend([e for e in user['faceEmbeddings'] if isinstance(e, (list, tuple))])
                            if not stored_embeddings and user.get('faceEmbedding'):
                                stored_embeddings = [user.get('faceEmbedding')]
                    except Exception:
                        stored_embeddings = stored_embeddings or []
                    ident['cached_embeddings'] = stored_embeddings
                    ident['cached_embeddings_at'] = now_ts

                similarity_score = None
                identity_verified = False
                if stored_embeddings:
                    try:
                        verify_payload = {'imageDataUrl': f"data:image/jpeg;base64,{frame_base64}"}
                        ok_verify, verify_result = call_ml_service('/verify-face', verify_payload, timeout=12)
                        if ok_verify and isinstance(verify_result, dict) and 'embedding' in verify_result:
                            sims = []
                            for stored in stored_embeddings:
                                sim = _cosine_similarity(verify_result['embedding'], stored)
                                if sim is not None:
                                    sims.append(sim)
                            if sims:
                                similarity_score = float(max(sims))
                    except Exception as e:
                        print(f"[PROCTOR] Identity check error: {e}")

                # Maintain a 15s similarity window and decide based on max similarity in-window.
                try:
                    rs = ident.get('recent_sims')
                    if not isinstance(rs, deque):
                        rs = deque()
                    if similarity_score is not None:
                        rs.append((now_ts, float(similarity_score)))
                    cutoff = now_ts - 15.0
                    while rs and float(rs[0][0]) < cutoff:
                        rs.popleft()
                    ident['recent_sims'] = rs
                    max_recent = max([v for _, v in rs], default=None)
                except Exception:
                    max_recent = similarity_score

                if max_recent is not None:
                    identity_verified = float(max_recent) >= float(identity_threshold)

                # Update response + session cache
                try:
                    st['last_identity'] = {
                        'similarity': float(max_recent) if max_recent is not None else None,
                        'verified': bool(identity_verified),
                        'threshold': float(identity_threshold),
                        'checked_at': now_dt.isoformat() + 'Z',
                    }
                    results['identityVerified'] = bool(identity_verified)
                    results['similarity'] = float(max_recent) if max_recent is not None else None
                except Exception:
                    pass

                if max_recent is None:
                    # If we couldn't compute similarity, do not punish.
                    ident['pass_count'] = int(ident.get('pass_count') or 0) + 1
                    ident['fail_count'] = 0
                else:
                    if identity_verified:
                        ident['pass_count'] = int(ident.get('pass_count') or 0) + 1
                        if int(ident['pass_count']) >= 2:
                            ident['fail_count'] = 0
                    else:
                        ident['fail_count'] = int(ident.get('fail_count') or 0) + 1
                        ident['pass_count'] = 0

                        # Require multiple consecutive mismatches (stability)
                        if int(ident['fail_count']) >= 2:
                            emit_event('identity_mismatch', {
                                'similarity': float(max_recent),
                                'threshold': float(identity_threshold),
                                'message': 'Face does not match registered student (sustained mismatches)',
                                'consecutive_failures': int(ident['fail_count']),
                                'risk_score': 90,
                            }, 'high', risk_score=90)

                st['identity'] = ident

            # Persist state
            PROCTOR_STATE[key] = st
    except Exception as e:
        app.logger.debug('Proctor logic error: %s', e)

    return jsonify(results), 200


@app.route('/api/proctor/reset', methods=['POST'])
def reset_proctor_state():
    """Reset proctoring state for a user-exam pair (called when exam starts)"""
    data = request.get_json()
    exam_id = data.get('examId')
    user_id = data.get('userId')
    
    if not exam_id or not user_id:
        return jsonify({"error": "examId and userId are required"}), 400
    
    key = (str(exam_id), str(user_id))
    if key in PROCTOR_STATE:
        del PROCTOR_STATE[key]
    
    return jsonify({"message": "Proctor state reset successfully"}), 200


@app.route('/api/proctor/audio', methods=['POST'])
def proctor_audio():
    """Process audio chunk and emit event only if human voice is detected.
    Note: Audio processing not yet implemented - returns Unknown status."""
    data = request.get_json()
    audio_b64 = data.get('audioData')
    exam_id = data.get('examId')
    user_id = data.get('userId')
    
    if not audio_b64:
        return jsonify({"error": "Audio data is required"}), 400
    
    # Note: Audio processing is not implemented in this service build.
    result = "Unknown"  # Placeholder until audio processing is implemented
    
    # Record proctoring event only if voice is detected
    if result == "Voice detected" and exam_id and user_id:
        try:
            now = datetime.datetime.utcnow()
            # Check cooldown (10 seconds for audio to prevent spam)
            recent = proctor_events_collection.find_one({
                'examId': str(exam_id),
                'userId': str(user_id),
                'eventType': 'audio_voice',
                'timestamp': {'$gt': now - datetime.timedelta(seconds=10)}
            })
            
            if not recent:
                proctor_events_collection.insert_one({
                    'examId': str(exam_id),
                    'userId': str(user_id),
                    'eventType': 'audio_voice',
                    'details': {'status': 'Voice detected', 'message': 'Human voice detected during exam'},
                    'severity': 'high',  # Voice during exam is high severity
                    'timestamp': now
                })
                print(f"[PROCTOR-EVENT] audio_voice (high) for user {user_id} in exam {exam_id}")
        except Exception as e:
            print(f"Error recording audio event: {e}")

    return jsonify({"audioStatus": result}), 200

# ✅ LOG SUSPICIOUS ACTIVITY - Browser Lock Violations (CRITICAL)
@app.route('/api/log-activity', methods=['POST'])
@limiter.limit("200 per hour")
def log_suspicious_activity():
    """
    CRITICAL: Log suspicious activities like tab switching, fullscreen exit, dev tools.
    
    Tracks student attempts to bypass exam security measures including:
    - Exiting fullscreen mode
    - Switching tabs/windows
    - Opening developer tools
    - Right-click attempts
    - Copy/paste attempts
    - Window blur/focus changes
    
    Request Body:
        examId (str): Exam identifier
        userId (str): User identifier
        activityType (str): Type of activity (fullscreen_exit, tab_switch, dev_tools, etc.)
        timestamp (str): ISO timestamp (optional, server will use current time if not provided)
        details (dict): Optional additional information
    
    Returns:
        200 OK: {message, eventId}
        400 Bad Request: Missing parameters
        500 Internal Error: Database error
    """
    try:
        # 1. Get and validate input
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        exam_id = data.get('examId')
        user_id = data.get('userId')
        activity_type = data.get('activityType')
        details = data.get('details', {})
        
        if not exam_id:
            return jsonify({"error": "examId is required"}), 400
        
        if not user_id:
            return jsonify({"error": "userId is required"}), 400
        
        if not activity_type:
            return jsonify({"error": "activityType is required"}), 400

        # Best-effort: mirror recent browser-lock activity into in-memory proctor state.
        # Used for context-aware escalation (e.g., fullscreen_exit + camera_blocked => critical).
        try:
            key = (str(exam_id), str(user_id))
            st = PROCTOR_STATE.get(key) or {'last_emit': {}, 'recent_activity': {}}
            if 'recent_activity' not in st or not isinstance(st.get('recent_activity'), dict):
                st['recent_activity'] = {}
            st['recent_activity'][str(activity_type)] = datetime.datetime.utcnow().timestamp()
            PROCTOR_STATE[key] = st
        except Exception:
            pass
        
        # 2. Determine severity and risk score based on activity type (ProctorU-style)
        # Risk scoring aligns with commercial proctoring platforms:
        # - Critical (80-100): Definitive cheating indicators
        # - High (60-79): Strong suspicious behavior
        # - Medium (40-59): Moderate concern, needs review
        # - Low (20-39): Minor infractions, natural behavior
        severity_map = {
            'fullscreen_exit': ('critical', 85),
            'tab_switch': ('high', 70),
            'tab_unfocused': ('high', 65),
            'window_blur': ('medium', 50),
            'dev_tools_opened': ('critical', 95),
            'dev_tools_attempt': ('high', 75),
            'right_click': ('low', 25),
            'copy_attempted': ('medium', 45),
            'paste_attempted': ('medium', 45),
            'print_screen': ('high', 70),
            'screenshot_attempt': ('high', 70),
            'multiple_monitors': ('medium', 55),
            'browser_resize': ('low', 20)
        }
        
        severity, risk_score = severity_map.get(activity_type, ('medium', 40))
        
        # 3. Store risk score in details for aggregation
        details['risk_score'] = risk_score
        details['severity'] = severity
        
        # 4. Create activity record
        activity_record = {
            'examId': str(exam_id),
            'userId': str(user_id),
            'type': 'suspicious_activity',
            'activityType': activity_type,
            'severity': severity,
            'risk_score': risk_score,
            'details': details,
            'timestamp': datetime.datetime.utcnow()
        }
        
        # 5. Store in database
        try:
            result = proctor_events_collection.insert_one(activity_record)
            event_id = str(result.inserted_id)
        except Exception as e:
            app.logger.error(f"Failed to store activity record: {e}")
            return jsonify({"error": "Failed to store activity"}), 500
        
        # 6. Broadcast to proctors via WebSocket if critical
        if severity in ['critical', 'high']:
            try:
                violation_data = {
                    'type': activity_type,
                    'severity': severity,
                    'risk_score': risk_score,
                    'message': f"Suspicious activity: {activity_type.replace('_', ' ')}"
                }
                broadcast_violation(exam_id, user_id, violation_data)
            except Exception as e:
                app.logger.error(f"Failed to broadcast activity: {e}")
        
        # 7. Return success
        return jsonify({
            "message": "Activity logged successfully",
            "eventId": event_id,
            "severity": severity,
            "risk_score": risk_score
        }), 200
        
    except Exception as e:
        app.logger.exception("Error in log_suspicious_activity")
        return jsonify({"error": "Internal processing error", "detail": str(e)}), 500

# --- Exam Routes ---
@app.route('/api/exams', methods=['POST'])
def create_exam():
    req_user = _get_authenticated_user_doc()
    if not req_user or req_user.get('role') != 'lecturer':
        return jsonify({'error': 'Forbidden: lecturer role required'}), 403

    data = request.get_json()
    required_fields = ['title', 'courseCode', 'lecturerId', 'institution', 'department', 'targetYear', 'questions']
    if not all(k in data for k in required_fields):
        return jsonify({"error": "Missing required fields for exam"}), 400

    # Prevent client-side spoofing: lecturer identity comes from auth.
    try:
        data['lecturerId'] = str(req_user.get('_id'))
        data['lecturerName'] = req_user.get('name') or data.get('lecturerName')
    except Exception:
        pass
    
    questions_with_ids = []
    for q in data.get('questions', []):
        q_with_id = q.copy()
        q_with_id['_id'] = ObjectId()
        
        # Normalize correctAnswer type for multiple choice questions
        if q_with_id.get('type') == 'multiple-choice' and 'correctAnswer' in q_with_id:
            try:
                # Ensure it's stored as an integer
                q_with_id['correctAnswer'] = int(q_with_id['correctAnswer'])
            except (ValueError, TypeError):
                pass  # Keep as-is if conversion fails
        
        questions_with_ids.append(q_with_id)

    new_exam = {
        "title": data['title'], "courseCode": data['courseCode'], "description": data.get('description', ''),
        "scheduledDate": data.get('scheduledDate'), "startTime": data.get('startTime'), "endTime": data.get('endTime'),
        "duration": data.get('duration'), "institution": data['institution'], "department": data['department'],
        "targetYear": data['targetYear'], "status": 'Scheduled', "lecturerId": data['lecturerId'],
        "lecturerName": data['lecturerName'], "questions": questions_with_ids, "createdAt": datetime.datetime.utcnow()
    }
    result = exams_collection.insert_one(new_exam)
    new_exam['_id'] = str(result.inserted_id)
    new_exam = serialize_doc(new_exam)
    
    # Emit real-time notification to students in the same institution/department/year
    try:
        socketio.emit('exam-created', {
            'examId': str(result.inserted_id),
            'title': data['title'],
            'institution': data['institution'],
            'department': data['department'],
            'targetYear': data['targetYear'],
            'scheduledDate': data.get('scheduledDate'),
            'startTime': data.get('startTime'),
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
        }, namespace='/proctor')
        app.logger.info(f'[EXAM-CREATE] Broadcasted exam-created for exam {result.inserted_id}')
    except Exception as e:
        app.logger.error(f'[EXAM-CREATE] Failed to emit exam-created: {e}')
    
    return jsonify({"message": "Exam created successfully", "exam": new_exam}), 201

@app.route('/api/exams/<exam_id>/status', methods=['PUT'])
def update_exam_status(exam_id):
    req_user = _get_authenticated_user_doc()
    if not req_user or req_user.get('role') != 'lecturer':
        return jsonify({'error': 'Forbidden: lecturer role required'}), 403

    # Only the owner lecturer can update their exam.
    try:
        ex = exams_collection.find_one({'_id': ObjectId(exam_id)}) if ObjectId.is_valid(exam_id) else None
    except Exception:
        ex = None
    if not ex:
        return jsonify({"error": "Exam not found"}), 404
    if str(ex.get('lecturerId')) != str(req_user.get('_id')):
        return jsonify({'error': 'Forbidden'}), 403

    data = request.get_json()
    new_status = data.get('status')
    if not new_status:
        return jsonify({"error": "New status is required"}), 400
    
    result = exams_collection.update_one(
        {'_id': ObjectId(exam_id)},
        {'$set': {'status': new_status}}
    )

    if result.modified_count == 1:
        return jsonify({"message": f"Exam status updated to {new_status}"}), 200
    else:
        return jsonify({"error": "Exam not found or status not updated"}), 404

# ✅ GET EXAM STATUS - Time Enforcement and Duration Limits (CRITICAL)
@app.route('/api/exams/<exam_id>/status', methods=['GET'])
@limiter.limit("200 per hour")
def get_exam_status(exam_id):
    """
    CRITICAL: Check exam status and enforce time limits.
    
    Returns current exam status (not_started, active, expired) based on start time,
    duration, and current time. Used by frontend to show countdown timer and
    auto-submit when time expires.
    
    Frontend should poll this endpoint every 5 seconds during exam to:
    - Update countdown timer
    - Warn user when time is running out
    - Auto-submit answers when exam expires
    
    URL Parameters:
        exam_id (str): Exam identifier
    
    Returns:
        200 OK with status:
            - not_started: {status, startTime, timeUntilStart}
            - active: {status, startTime, endTime, timeRemaining, percentTimeRemaining, duration}
            - expired: {status, endTime, message}
        404 Not Found: Exam not found
        400 Bad Request: Exam start time not set
        500 Internal Error: Time calculation error
    """
    try:
        # 1. Validate and retrieve exam
        try:
            exam = exams_collection.find_one({'_id': ObjectId(exam_id)}) if ObjectId.is_valid(exam_id) else None
            if not exam:
                return jsonify({"error": "Exam not found"}), 404
        except Exception as e:
            return jsonify({"error": "Invalid exam ID format"}), 400
        
        # 2. Get exam timing parameters
        start_time = exam.get('startTime')
        duration = exam.get('duration')  # in minutes
        
        # Validate required fields
        if not start_time:
            return jsonify({"error": "Exam start time not set"}), 400
        
        if not duration:
            return jsonify({"error": "Exam duration not set"}), 400
        
        # 3. Parse start time if it's a string (ISO format)
        if isinstance(start_time, str):
            try:
                # Try parsing ISO format with 'Z' or without timezone
                if start_time.endswith('Z'):
                    start_time = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                else:
                    start_time = datetime.datetime.fromisoformat(start_time)
                
                # Convert to UTC if timezone-aware
                if start_time.tzinfo is not None:
                    start_time = start_time.replace(tzinfo=None)
            except Exception as e:
                app.logger.error(f"Failed to parse start time: {e}")
                return jsonify({"error": "Invalid start time format"}), 400
        
        # 4. Calculate end time
        try:
            duration_minutes = int(duration)
            end_time = start_time + datetime.timedelta(minutes=duration_minutes)
        except Exception as e:
            return jsonify({"error": "Invalid duration format"}), 400
        
        # 5. Get current UTC time
        now = datetime.datetime.utcnow()
        
        # 6. Determine exam status based on time comparison
        
        # Case 1: Exam hasn't started yet
        if now < start_time:
            time_until_start = (start_time - now).total_seconds()
            return jsonify({
                "status": "not_started",
                "startTime": start_time.isoformat() + 'Z',
                "timeUntilStart": int(time_until_start)
            }), 200
        
        # Case 2: Exam has expired
        elif now > end_time:
            return jsonify({
                "status": "expired",
                "endTime": end_time.isoformat() + 'Z',
                "message": "Exam time exceeded"
            }), 200
        
        # Case 3: Exam is active
        else:
            # Calculate remaining time
            time_remaining = (end_time - now).total_seconds()
            
            # Calculate percentage of time remaining
            total_duration_seconds = duration_minutes * 60
            percent_time_remaining = (time_remaining / total_duration_seconds) * 100 if total_duration_seconds > 0 else 0
            
            # Determine warning level (optional bonus feature)
            warning = None
            if time_remaining <= 60:  # 1 minute or less
                warning = "critical"
            elif time_remaining <= 300:  # 5 minutes or less
                warning = "low"
            
            response_data = {
                "status": "active",
                "startTime": start_time.isoformat() + 'Z',
                "endTime": end_time.isoformat() + 'Z',
                "timeRemaining": int(time_remaining),
                "percentTimeRemaining": round(percent_time_remaining, 2),
                "duration": duration_minutes
            }
            
            # Add warning if applicable
            if warning:
                response_data["warning"] = warning
            
            return jsonify(response_data), 200
        
    except Exception as e:
        app.logger.exception("Error in get_exam_status")
        return jsonify({"error": "Time calculation error", "detail": str(e)}), 500

@app.route('/api/lecturer/exams/<exam_id>/monitor', methods=['GET'])
@limiter.limit("60 per minute")
def get_exam_monitoring_data(exam_id):
    """
    Get real-time monitoring data for all students taking this exam.
    
    Returns list of active students with their proctoring status, violations, and metrics.
    Used by lecturer live monitoring dashboard.
    
    URL Parameters:
        exam_id (str): Exam identifier
    
    Returns:
        200 OK with monitoring data for all active students
        404 Not Found: Exam not found
        500 Internal Error: Database error
    """
    try:
        # Validate exam exists
        exam = exams_collection.find_one({'_id': ObjectId(exam_id)}) if ObjectId.is_valid(exam_id) else None
        if not exam:
            return jsonify({"error": "Exam not found"}), 404
        
        # Get all exam attempts for this exam that are in progress
        attempts = list(exam_attempts_collection.find({
            'exam_id': exam_id,
            'status': 'in_progress'
        }))
        
        # Build student monitoring data
        students_data = []
        total_violations = 0
        
        for attempt in attempts:
            user_id = attempt.get('user_id')
            user = users_collection.find_one({'_id': ObjectId(user_id)}) if ObjectId.is_valid(user_id) else None
            if not user:
                continue
            
            # Get proctoring logs for violation count
            logs = list(proctoring_logs_collection.find({
                'exam_id': exam_id,
                'user_id': user_id
            }))
            
            # Filter out browser lock violations (they're already handled by security warnings)
            # Only count ML-detected violations for action review
            violation_count = len([
                log for log in logs 
                if log.get('violation_type') and 
                not log.get('violation_type', '').startswith('browser_lock')
            ])
            total_violations += violation_count
            
            # Determine status based on violations and risk
            risk_score = attempt.get('risk_score', 0)
            status = 'normal'
            if violation_count >= 5 or risk_score >= 80:
                status = 'critical'
            elif violation_count >= 3 or risk_score >= 60:
                status = 'suspicious'
            elif violation_count >= 1 or risk_score >= 40:
                status = 'warning'
            
            # Get latest violation
            latest_violation = None
            violation_time = None
            latest_violation_event_id = None
            if logs:
                sorted_logs = sorted(logs, key=lambda x: x.get('timestamp', datetime.datetime.min), reverse=True)
                if sorted_logs:
                    latest_log = sorted_logs[0]
                    latest_violation = latest_log.get('violation_type', 'Unknown violation')
                    violation_time = latest_log.get('timestamp')
                    latest_violation_event_id = latest_log.get('eventId')
                    if violation_time:
                        if isinstance(violation_time, str):
                            violation_time = violation_time
                        else:
                            violation_time = violation_time.isoformat() + 'Z'
            
            # Calculate time remaining
            start_time = attempt.get('start_time')
            duration = exam.get('duration', 60)  # default 60 minutes
            time_remaining = duration * 60  # convert to seconds
            
            if start_time:
                if isinstance(start_time, str):
                    start_time = datetime.datetime.fromisoformat(start_time.replace('Z', ''))
                elapsed = (datetime.datetime.utcnow() - start_time).total_seconds()
                time_remaining = max(0, (duration * 60) - int(elapsed))
            
            # Get last known proctoring metrics
            face_detected = True
            gaze = 'forward'
            head_pose = 'normal'
            
            if logs:
                for log in reversed(logs):
                    metrics = log.get('metrics', {})
                    if metrics:
                        face_detected = metrics.get('face_detected', True)
                        # Estimate gaze based on violations
                        vtype = log.get('violation_type', '').lower()
                        if 'gaze' in vtype or 'looking' in vtype:
                            gaze = 'away'
                        elif 'head' in vtype or 'pose' in vtype:
                            head_pose = 'tilted'
                        break
            
            # Fetch video evidence for action review panel
            # Prefer exact eventId match (1:1 mapping). Fall back to violationType match.
            latest_video_url = None
            violation_video_url = None
            try:
                if latest_violation_event_id and isinstance(latest_violation_event_id, str):
                    vq = {
                        'examId': str(exam_id),
                        'userId': str(user_id),
                        'type': 'evidence',
                        'evidenceType': 'video',
                        'eventId': latest_violation_event_id,
                    }
                    match_vid = proctor_events_collection.find(vq).sort('timestamp', -1).limit(1)
                    match_vid_list = list(match_vid)
                    if match_vid_list:
                        violation_video_url = f"/api/evidence/{str(match_vid_list[0]['_id'])}"

                if (
                    not violation_video_url and
                    latest_violation and isinstance(latest_violation, str) and
                    latest_violation.lower() not in ('unknown', 'unknown violation')
                ):
                    vq = {
                        'examId': str(exam_id),
                        'userId': str(user_id),
                        'type': 'evidence',
                        'evidenceType': 'video',
                        'violationType': latest_violation
                    }
                    match_vid = proctor_events_collection.find(vq).sort('timestamp', -1).limit(1)
                    match_vid_list = list(match_vid)
                    if match_vid_list:
                        violation_video_url = f"/api/evidence/{str(match_vid_list[0]['_id'])}"

                latest_vid = proctor_events_collection.find({
                    'examId': str(exam_id),
                    'userId': str(user_id),
                    'type': 'evidence',
                    'evidenceType': 'video'
                }).sort('timestamp', -1).limit(1)
                latest_vid_list = list(latest_vid)
                if latest_vid_list:
                    latest_video_url = f"/api/evidence/{str(latest_vid_list[0]['_id'])}"
            except Exception as _e:
                pass

            students_data.append({
                'userId': str(user_id),
                'studentId': user.get('studentId', user.get('email', '')),
                'name': user.get('name', 'Unknown'),
                'faceDetected': face_detected,
                'gaze': gaze,
                'headPose': head_pose,
                'violations': violation_count,
                'status': status,
                'timeRemaining': time_remaining,
                'latestViolation': latest_violation,
                'violationTime': violation_time,
                'latestViolationEventId': latest_violation_event_id,
                'violationVideoUrl': violation_video_url,
                'latestVideoUrl': latest_video_url
            })
        
        # Calculate ID verified count (students with successful face verification)
        id_verified_count = len([s for s in students_data if s['faceDetected']])
        
        # Calculate average risk
        total_risk = sum(s['violations'] * 10 for s in students_data)  # Simple risk calculation
        avg_risk = (total_risk / len(students_data)) if students_data else 0
        
        # Calculate compliance rate
        compliant_students = len([s for s in students_data if s['status'] == 'normal'])
        compliance = (compliant_students / len(students_data) * 100) if students_data else 100
        
        return jsonify({
            'students': students_data,
            'stats': {
                'activeExams': 1,
                'studentsOnline': len(students_data),
                'idVerified': id_verified_count,
                'activeViolations': total_violations,
                'avgRisk': round(avg_risk, 1),
                'compliance': round(compliance, 1)
            }
        }), 200
        
    except Exception as e:
        app.logger.exception("Error in get_exam_monitoring_data")
        return jsonify({"error": "Failed to fetch monitoring data", "detail": str(e)}), 500

@app.route('/api/exams/<exam_id>/report', methods=['GET', 'POST'])
@limiter.limit("30 per minute")
def get_exam_report(exam_id):
    """
    Get comprehensive exam report with student results, violations, and statistics.
    
    Returns detailed report for completed/in-progress exam including:
    - Overall statistics (attendance, average score, pass rate)
    - Individual student results with scores and violations
    - Incident breakdown by type
    
    URL Parameters:
        exam_id (str): Exam identifier
    
    Returns:
        200 OK with exam report data
        404 Not Found: Exam not found
        500 Internal Error: Database error
    """
    print(f"[REPORT] Fetching report for exam {exam_id}")
    
    try:
        # Validate exam exists
        if not ObjectId.is_valid(exam_id):
            print(f"[REPORT] Invalid exam ID format: {exam_id}")
            return jsonify({"error": "Invalid exam ID"}), 400
            
        exam = exams_collection.find_one({'_id': ObjectId(exam_id)})
        if not exam:
            print(f"[REPORT] Exam not found: {exam_id}")
            return jsonify({"error": "Exam not found"}), 404
        
        print(f"[REPORT] Found exam: {exam.get('title')}")
        
        # Get all exam attempts for this exam
        attempts = list(exam_attempts_collection.find({'exam_id': exam_id}))
        print(f"[REPORT] Found {len(attempts)} attempts")
        
        # Initialize default values for empty exam
        total_questions = len(exam.get('questions', []))
        print(f"[REPORT] Exam has {total_questions} questions")
        
        # Calculate statistics
        total_students = len(attempts)
        completed_students = len([a for a in attempts if a.get('status') == 'completed'])
        
        # Get student results
        students_data = []
        total_score = 0
        passed_students = 0
        total_incidents = 0
        high_risk_count = 0
        
        incident_breakdown = {
            'identityMismatch': 0,
            'multipleFaces': 0,
            'phoneDetected': 0,
            'tabSwitch': 0,
            'gazeAway': 0,
            'audioViolation': 0
        }
        
        for attempt in attempts:
            user_id = attempt.get('user_id')
            user = users_collection.find_one({'_id': ObjectId(user_id)}) if ObjectId.is_valid(user_id) else None
            if not user:
                continue
            
            score = attempt.get('score', 0)
            total_questions = len(exam.get('questions', []))
            percentage = round((score / total_questions * 100), 2) if total_questions > 0 else 0
            risk_score = attempt.get('risk_score', 0)
            
            # Get proctoring logs for incident count
            logs = list(proctoring_logs_collection.find({
                'exam_id': exam_id,
                'user_id': user_id
            }))
            
            incident_count = len([log for log in logs if log.get('violation_type')])
            
            # Count incident types
            for log in logs:
                violation_type = log.get('violation_type', '')
                if 'identity' in violation_type.lower() or 'face_mismatch' in violation_type.lower():
                    incident_breakdown['identityMismatch'] += 1
                elif 'multiple' in violation_type.lower():
                    incident_breakdown['multipleFaces'] += 1
                elif 'phone' in violation_type.lower() or 'mobile' in violation_type.lower():
                    incident_breakdown['phoneDetected'] += 1
                elif 'tab' in violation_type.lower() or 'window' in violation_type.lower():
                    incident_breakdown['tabSwitch'] += 1
                elif 'gaze' in violation_type.lower() or 'eye' in violation_type.lower():
                    incident_breakdown['gazeAway'] += 1
                elif 'audio' in violation_type.lower() or 'voice' in violation_type.lower():
                    incident_breakdown['audioViolation'] += 1
            
            # Calculate duration
            start_time = attempt.get('start_time')
            end_time = attempt.get('end_time')
            duration = 0
            if start_time and end_time:
                if isinstance(start_time, str):
                    start_time = datetime.datetime.fromisoformat(start_time.replace('Z', ''))
                if isinstance(end_time, str):
                    end_time = datetime.datetime.fromisoformat(end_time.replace('Z', ''))
                duration = int((end_time - start_time).total_seconds() / 60)  # in minutes
            
            total_score += score
            if percentage >= 50:  # Pass threshold
                passed_students += 1
            
            total_incidents += incident_count
            if risk_score >= 60:
                high_risk_count += 1
            
            students_data.append({
                'userId': user_id,
                'studentId': user.get('studentId', user.get('email', '')),
                'name': user.get('name', 'Unknown'),
                'score': score,
                'percentage': percentage,
                'riskScore': risk_score,
                'incidentCount': incident_count,
                'duration': duration,
                'status': attempt.get('status', 'in_progress')
            })
        
        # Calculate overall metrics
        total_questions = len(exam.get('questions', []))
        average_score = round(total_score / total_students, 2) if total_students > 0 else 0
        average_percentage = round((average_score / total_questions * 100), 2) if total_questions > 0 else 0
        pass_rate = round((passed_students / total_students * 100), 1) if total_students > 0 else 0
        attendance_rate = round((total_students / total_students * 100), 1) if total_students > 0 else 0
        
        # Format date properly
        exam_date = 'N/A'
        scheduled_date = exam.get('scheduledDate')
        if scheduled_date:
            if isinstance(scheduled_date, str):
                try:
                    # Try to parse and format the date
                    dt = datetime.datetime.fromisoformat(scheduled_date.replace('Z', ''))
                    exam_date = dt.strftime('%Y-%m-%d')
                except:
                    # If it's already in YYYY-MM-DD format, use as-is
                    exam_date = scheduled_date.split('T')[0] if 'T' in scheduled_date else scheduled_date
            elif isinstance(scheduled_date, datetime.datetime):
                exam_date = scheduled_date.strftime('%Y-%m-%d')
        
        report_data = {
            'examId': exam_id,
            'title': exam.get('title', 'Untitled Exam'),
            'courseCode': exam.get('courseCode', 'N/A'),
            'date': exam_date,
            'duration': exam.get('duration', 0),
            'totalStudents': total_students,
            'attendanceRate': attendance_rate,
            'averageScore': average_percentage,  # Show as percentage for consistency
            'passRate': pass_rate,
            'totalIncidents': total_incidents,
            'highRiskStudents': high_risk_count,
            'students': students_data,
            'incidentBreakdown': incident_breakdown
        }
        
        print(f"[REPORT] Generated report for exam {exam_id}: {total_students} students, avg score: {average_percentage}%")
        
        return jsonify(report_data), 200
        
    except Exception as e:
        app.logger.exception("Error in get_exam_report")
        return jsonify({"error": "Failed to fetch exam report", "detail": str(e)}), 500

@app.route('/api/exams/<exam_id>', methods=['DELETE'])
def delete_exam(exam_id):
    try:
        result = exams_collection.delete_one({'_id': ObjectId(exam_id)})
        if result.deleted_count == 1:
            return jsonify({"message": "Exam deleted successfully"}), 200
        else:
            return jsonify({"error": "Exam not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/exams/<exam_id>/start', methods=['POST'])
def start_exam(exam_id):
    """Track when a student starts an exam to calculate duration and participation."""
    try:
        data = request.get_json() or {}
        user_id = data.get('userId')
        
        if not user_id:
            return jsonify({"error": "userId is required"}), 400
        
        # Validate exam exists
        exam = exams_collection.find_one({'_id': ObjectId(exam_id)}) if ObjectId.is_valid(exam_id) else None
        if not exam:
            return jsonify({"error": "Exam not found"}), 404
        
        # Get user info
        user = users_collection.find_one({'_id': ObjectId(user_id)}) if ObjectId.is_valid(user_id) else None
        
        # Create or update exam attempt record
        result = exam_attempts_collection.update_one(
            {
                'exam_id': exam_id,
                'user_id': user_id
            },
            {
                '$setOnInsert': {
                    'exam_id': exam_id,
                    'user_id': user_id,
                    'start_time': datetime.datetime.utcnow(),
                    'status': 'in_progress',
                    'score': 0,
                    'total_marks': len(exam.get('questions', [])),
                    'percentage': 0,
                    'correct_count': 0,
                    'risk_score': 0
                }
            },
            upsert=True
        )
        
        # Emit real-time notification to lecturers monitoring this exam
        if result.upserted_id or result.matched_count > 0:
            try:
                socketio.emit('student-joined', {
                    'examId': exam_id,
                    'userId': user_id,
                    'studentId': user.get('studentId', user.get('email', '')) if user else 'Unknown',
                    'name': user.get('name', 'Unknown') if user else 'Unknown',
                    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
                }, room=exam_id, namespace='/proctor')
                app.logger.info(f'[EXAM-START] Broadcasted student-joined for user {user_id} in exam {exam_id}')
            except Exception as e:
                app.logger.error(f'[EXAM-START] Failed to emit student-joined: {e}')
        
        return jsonify({"message": "Exam started successfully"}), 200
        
    except Exception as e:
        app.logger.exception("Error in start_exam")
        return jsonify({"error": "Failed to start exam", "detail": str(e)}), 500

@app.route('/api/exams/<exam_id>/submit', methods=['POST', 'OPTIONS'])
def submit_exam(exam_id):
    # Handle OPTIONS preflight request explicitly
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Get JSON data
        data = request.get_json(force=True)
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        user_id = data.get('userId')
        answers = data.get('answers')

        if not user_id or not answers:
            return jsonify({"error": "User ID and answers are required"}), 400
        
        # CRITICAL: Check if exam was terminated by proctor - enforce zero score
        existing_attempt = exam_attempts_collection.find_one({
            'exam_id': exam_id,
            'user_id': user_id
        })
        
        if existing_attempt and existing_attempt.get('status') == 'terminated_by_proctor':
            app.logger.warning(f'[SUBMIT] Student {user_id} attempted to submit terminated exam - enforcing zero score')
            return jsonify({
                'score': 0,
                'totalMarks': existing_attempt.get('total_marks', 0),
                'percentage': 0,
                'correctCount': 0,
                'terminated': True,
                'message': 'Score: 0 (Exam terminated by invigilator)',
                'perQuestion': []
            }), 200
        
        exam = exams_collection.find_one({'_id': ObjectId(exam_id)})
            
        if not exam:
            return jsonify({"error": "Exam not found"}), 404
        
        total_marks = 0
        score = 0
        correct_count = 0
        per_question = []

        for question in exam.get('questions', []):
            q_id = str(question['_id'])
            marks = 1  # Each question is worth 1 mark
            total_marks += marks
            user_answer = answers.get(q_id)
            correct = False
            
            print(f"[GRADING] Processing question {q_id}, type={question.get('type')}, user_answer={user_answer}")

            # Robust comparison by question type
            qtype = question.get('type')
            correct_answer = question.get('correctAnswer')

            try:
                if user_answer is not None:
                    # Multiple choice: both stored as integers (1, 2, 3, 4)
                    if qtype == 'multiple-choice':
                        print(f"[GRADING] Q{q_id}: user={user_answer} (type={type(user_answer)}), correct={correct_answer} (type={type(correct_answer)})")
                        
                        # Convert both to integers for comparison
                        try:
                            user_answer_int = int(user_answer)
                            correct_answer_int = int(correct_answer)
                            
                            if user_answer_int == correct_answer_int:
                                correct = True
                                print(f"[GRADING] Q{q_id}: CORRECT - user answered {user_answer_int}, correct is {correct_answer_int}")
                            else:
                                print(f"[GRADING] Q{q_id}: WRONG - user answered {user_answer_int}, correct is {correct_answer_int}")
                        except (ValueError, TypeError) as e:
                            print(f"[GRADING] Q{q_id}: Conversion error: {e}")

                    elif qtype == 'true-false':
                        # Coerce user answer to boolean
                        ua_bool = None
                        if isinstance(user_answer, bool):
                            ua_bool = user_answer
                        else:
                            s = str(user_answer).lower()
                            if s in ('true', '1', 'yes'): ua_bool = True
                            elif s in ('false', '0', 'no'): ua_bool = False
                        
                        # Coerce correct answer to boolean
                        ca_bool = None
                        if isinstance(correct_answer, bool):
                            ca_bool = correct_answer
                        else:
                            s = str(correct_answer).lower()
                            if s in ('true', '1', 'yes'): ca_bool = True
                            elif s in ('false', '0', 'no'): ca_bool = False
                        
                        # Compare booleans
                        if ua_bool is not None and ca_bool is not None and ua_bool == ca_bool:
                            correct = True

                    else:
                        # Short answer / essay: perform trimmed case-insensitive match
                        if correct_answer is not None and str(correct_answer).strip() != '':
                            if str(user_answer).strip().lower() == str(correct_answer).strip().lower():
                                correct = True
                        else:
                            correct = False
                            
            except Exception as e:
                print(f"[SUBMIT] Error grading question {q_id}: {e}")

            if correct:
                score += marks
                correct_count += 1
                print(f"[GRADING] Q{q_id}: MARKED CORRECT, score now={score}")
            else:
                print(f"[GRADING] Q{q_id}: MARKED WRONG")

            per_question.append({
                'questionId': q_id,
                'question': question.get('question'),
                'given': user_answer,
                'expected': correct_answer,
                'marks': marks,
                'correct': correct
            })

        percentage = round((score / total_marks) * 100, 2) if total_marks > 0 else 0
        
        print(f"[GRADING] FINAL RESULTS: score={score}, total_marks={total_marks}, percentage={percentage}, correct_count={correct_count}")

        attempt_record = {
            'userId': user_id,
            'score': score,  # Actual score (number of correct answers)
            'totalMarks': total_marks,
            'percentage': percentage,
            'correctCount': correct_count,
            'completedAt': datetime.datetime.utcnow().isoformat(),
            'perQuestion': per_question
        }

        # Store attempt in exams collection
        exams_collection.update_one(
            {'_id': ObjectId(exam_id)},
            {
                '$push': {'attempts': attempt_record},
                '$addToSet': {'completedBy': user_id}
            }
        )
        
        # Also store in exam_attempts collection for easier querying
        exam_attempts_collection.update_one(
            {
                'exam_id': exam_id,
                'user_id': user_id
            },
            {
                '$set': {
                    'exam_id': exam_id,
                    'user_id': user_id,
                    'score': score,
                    'total_marks': total_marks,
                    'percentage': percentage,
                    'correct_count': correct_count,
                    'status': 'completed',
                    'end_time': datetime.datetime.utcnow(),
                    'per_question': per_question,
                    'risk_score': 0  # Will be updated by proctoring
                }
            },
            upsert=True
        )

        return jsonify({
            'score': score,
            'totalMarks': total_marks,
            'percentage': percentage,
            'correctCount': correct_count,
            'perQuestion': per_question
        }), 200

    except Exception as e:
        print(f"[SUBMIT] Error processing submission: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to submit exam"}), 500
        return jsonify({
            "message": "Exam submitted successfully!",
            "score": percentage,
            "totalMarks": total_marks,
            "perQuestion": per_question
        }), 200
        
    except Exception as e:
        print(f"[SUBMIT] CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ✅ CALCULATE SCORE - Server-Side Answer Verification (CRITICAL)
@app.route('/api/exams/<exam_id>/calculate-score', methods=['POST'])
@limiter.limit("50 per hour")
def calculate_score(exam_id):
    """
    CRITICAL: Server-side answer verification and score calculation.
    
    Frontend grading can be bypassed - this endpoint provides authoritative scoring.
    Compares user answers against correct answers, calculates marks, stores attempt
    record, and returns detailed results with grade.
    
    Request Body:
        userId (str): User identifier submitting answers
        answers (list): Array of answer objects with questionId and answer fields
            Example: [{"questionId": "q1", "answer": "A"}, {"questionId": "q2", "answer": true}]
    
    Returns:
        200 OK: {score, totalMarks, percentage, correctCount, totalCount, grade}
        400 Bad Request: Missing parameters, exam has no questions, or question not found
        404 Not Found: Exam not found
        500 Internal Error: Database or processing error
    """
    try:
        # 1. Validate input parameters
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        user_id = data.get('userId')
        user_answers = data.get('answers')
        
        if not user_id:
            return jsonify({"error": "userId is required"}), 400
        
        if not user_answers or not isinstance(user_answers, list):
            return jsonify({"error": "answers array is required"}), 400
        
        # 2. Validate exam exists
        try:
            exam = exams_collection.find_one({'_id': ObjectId(exam_id)}) if ObjectId.is_valid(exam_id) else None
            if not exam:
                return jsonify({"error": "Exam not found"}), 404
        except Exception as e:
            return jsonify({"error": "Invalid exam ID format"}), 400
        
        # 3. Validate exam has questions
        questions = exam.get('questions', [])
        if not questions:
            return jsonify({"error": "Exam has no questions"}), 400
        
        # 4. Initialize scoring variables
        score = 0
        total_marks = 0
        correct_count = 0
        total_count = len(questions)
        
        # Create answer lookup for faster access
        answer_map = {ans.get('questionId'): ans.get('answer') for ans in user_answers if ans.get('questionId')}
        
        # 5. Process each question and compare answers
        for question in questions:
            try:
                # Get question details
                question_id = str(question.get('_id', ''))
                correct_answer = question.get('correctAnswer')
                marks = question.get('marks', 0) or 0
                question_type = question.get('type', 'short-answer')
                
                # Add to total marks
                total_marks += marks
                
                # Get user's answer for this question
                user_answer = answer_map.get(question_id)
                
                # Skip if user didn't answer
                if user_answer is None:
                    continue
                
                # 6. Compare answers based on question type
                is_correct = False
                
                if question_type == 'multiple-choice':
                    # Multiple choice: exact match (case-sensitive for option letters)
                    if str(user_answer) == str(correct_answer):
                        is_correct = True
                
                elif question_type == 'true-false':
                    # Boolean: handle various true/false representations
                    user_bool = None
                    correct_bool = None
                    
                    # Convert user answer to boolean
                    if isinstance(user_answer, bool):
                        user_bool = user_answer
                    else:
                        user_str = str(user_answer).lower().strip()
                        if user_str in ('true', '1', 'yes', 't', 'y'):
                            user_bool = True
                        elif user_str in ('false', '0', 'no', 'f', 'n'):
                            user_bool = False
                    
                    # Convert correct answer to boolean
                    if isinstance(correct_answer, bool):
                        correct_bool = correct_answer
                    else:
                        correct_str = str(correct_answer).lower().strip()
                        if correct_str in ('true', '1', 'yes', 't', 'y'):
                            correct_bool = True
                        elif correct_str in ('false', '0', 'no', 'f', 'n'):
                            correct_bool = False
                    
                    # Compare booleans
                    if user_bool is not None and correct_bool is not None and user_bool == correct_bool:
                        is_correct = True
                
                else:
                    # Short answer / essay: case-insensitive, trimmed comparison
                    if correct_answer is not None:
                        user_text = str(user_answer).strip().lower()
                        correct_text = str(correct_answer).strip().lower()
                        if user_text == correct_text:
                            is_correct = True
                
                # 7. Update score if correct
                if is_correct:
                    score += marks
                    correct_count += 1
                    
            except Exception as e:
                app.logger.error(f"Error grading question {question_id}: {e}")
                # Continue processing other questions even if one fails
                continue
        
        # 8. Calculate percentage (handle division by zero)
        percentage = round((score / total_marks) * 100, 2) if total_marks > 0 else 0
        
        # 9. Determine grade based on percentage
        if percentage >= 90:
            grade = 'A'
        elif percentage >= 80:
            grade = 'B'
        elif percentage >= 70:
            grade = 'C'
        elif percentage >= 60:
            grade = 'D'
        else:
            grade = 'F'
        
        # 10. Store attempt record in database
        attempt_record = {
            'userId': user_id,
            'score': score,
            'totalMarks': total_marks,
            'percentage': percentage,
            'correctCount': correct_count,
            'totalCount': total_count,
            'grade': grade,
            'submittedAt': datetime.datetime.utcnow(),
            'answers': user_answers
        }
        
        try:
            # Push attempt to attempts array and add user to completedBy set
            exams_collection.update_one(
                {'_id': ObjectId(exam_id)},
                {
                    '$push': {'attempts': attempt_record},
                    '$addToSet': {'completedBy': user_id}
                }
            )
        except Exception as e:
            app.logger.error(f"Failed to store attempt record: {e}")
            return jsonify({"error": "Failed to save exam results"}), 500
        
        # 11. Return results (DO NOT include correct answers or answer keys)
        return jsonify({
            "score": score,
            "totalMarks": total_marks,
            "percentage": percentage,
            "correctCount": correct_count,
            "totalCount": total_count,
            "grade": grade
        }), 200
        
    except Exception as e:
        app.logger.exception("Error in calculate_score")
        return jsonify({"error": "Internal processing error", "detail": str(e)}), 500

@app.route('/api/exams', methods=['GET'])
def get_exams():
    # Optional ?userId= to include per-user attempt info (helps client hide Start button after submission)
    user_id = request.args.get('userId')
    all_exams_cursor = exams_collection.find()
    all_exams = []
    for exam in all_exams_cursor:
        exam_safe = serialize_doc(exam)
        # Add meta about attempts for the requesting user if provided
        if user_id:
            attempts = exam.get('attempts', []) or []
            # find latest attempt for this user
            latest = None
            for a in reversed(attempts):
                if a.get('userId') == user_id or str(a.get('userId')) == str(user_id):
                    latest = a
                    break
            if latest:
                # Ensure completedAt is visible as iso
                la = latest.copy()
                if isinstance(la.get('completedAt'), datetime.datetime):
                    la['completedAt'] = la['completedAt'].isoformat()
                exam_safe['attemptForUser'] = la
            else:
                exam_safe['attemptForUser'] = None

            # Whether this user has completed the exam
            completed_by = exam.get('completedBy', []) or []
            exam_safe['completedByUser'] = str(user_id) in [str(x) for x in completed_by]

            # compute canStart for this user: only if not completed and within scheduled window or status Available/Live
            def within_window(ex):
                try:
                    date = datetime.datetime.fromisoformat(ex.get('scheduledDate')) if ex.get('scheduledDate') else None
                    if not date:
                        return ex.get('status') in ['Available', 'Live']
                    sh, sm = (ex.get('startTime') or '00:00').split(':')
                    eh, em = (ex.get('endTime') or '23:59').split(':')
                    start = datetime.datetime(date.year, date.month, date.day, int(sh or 0), int(sm or 0))
                    end = datetime.datetime(date.year, date.month, date.day, int(eh or 23), int(em or 59))
                    now = datetime.datetime.now()  # Use local time instead of UTC
                    return now >= start and now <= end
                except Exception:
                    return False

            exam_safe['canStartForUser'] = (exam_safe.get('status') in ['Available', 'Live'] or within_window(exam)) and not exam_safe['completedByUser']

        all_exams.append(exam_safe)

    return jsonify({"exams": all_exams}), 200


@app.route('/api/exams/<exam_id>', methods=['PUT'])
def update_exam(exam_id):
    req_user = _get_authenticated_user_doc()
    if not req_user or req_user.get('role') != 'lecturer':
        return jsonify({'error': 'Forbidden: lecturer role required'}), 403

    # Only the owner lecturer can update their exam.
    try:
        ex = exams_collection.find_one({'_id': ObjectId(exam_id)}) if ObjectId.is_valid(exam_id) else None
    except Exception:
        ex = None
    if not ex:
        return jsonify({'error': 'Exam not found'}), 404
    if str(ex.get('lecturerId')) != str(req_user.get('_id')):
        return jsonify({'error': 'Forbidden'}), 403

    """Update exam fields and questions. Expects fields similar to creation payload."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    allowed = {'title', 'courseCode', 'description', 'scheduledDate', 'startTime', 'endTime', 'duration', 'institution', 'department', 'targetYear', 'questions'}
    update = {k: v for k, v in data.items() if k in allowed}
    if 'questions' in update:
        # Ensure each question has an _id as ObjectId
        new_questions = []
        for q in update['questions']:
            q_copy = q.copy()
            qid = q_copy.get('_id')
            try:
                if qid:
                    q_copy['_id'] = ObjectId(qid)
                else:
                    q_copy['_id'] = ObjectId()
            except Exception:
                q_copy['_id'] = ObjectId()
            new_questions.append(q_copy)
        update['questions'] = new_questions

    try:
        result = exams_collection.update_one({'_id': ObjectId(exam_id)}, {'$set': update})
        if result.matched_count == 0:
            return jsonify({'error': 'Exam not found'}), 404
        exam = exams_collection.find_one({'_id': ObjectId(exam_id)})
        exam = serialize_doc(exam)
        return jsonify({'message': 'Exam updated', 'exam': exam}), 200
    except Exception as e:
        print(f"Error updating exam {exam_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/proctor/event', methods=['POST'])
def record_proctor_event():
    """Record a proctoring event emitted by the student's client (suspicious detections)."""
    data = request.get_json()
    
    required = ['examId', 'userId', 'eventType']
    if not data or not all(k in data for k in required):
        print(f"[PROCTOR-EVENT] ERROR: Missing required fields")
        return jsonify({'error': 'Missing required fields'}), 400

    requester = _get_authenticated_user_id()
    if not requester:
        return jsonify({'error': 'Authentication required'}), 403
    if str(requester) != str(data.get('userId')):
        return jsonify({'error': 'Forbidden'}), 403

    # Normalize eventType to a consistent lower_snake format
    et = data.get('eventType', '') or ''
    # Normalize eventType: convert CamelCase and spaces to lower_snake
    import re
    s = str(et).strip()
    # Insert underscores before camelCase transitions, replace spaces and hyphens
    s2 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s2)
    # Use raw string for regex to avoid invalid escape sequence warnings
    event_type = re.sub(r'[\s\-]+', '_', s2).lower()

    # Assign a simple severity mapping (can be tuned later)
    severity_map = {
        'audio': 'high',
        'identity': 'high',
        'multiple_faces': 'high',
        'object_detected': 'high',
        'head_pose': 'medium',
        'gaze': 'medium',
        'gaze_extreme': 'critical',
        'gaze_frequency': 'critical',
        'gaze_sustained': 'high',
        'blink': 'low',
        # Advanced events
        'head_pose_excess': 'medium',
        'gaze_aversion': 'medium',
        'talking': 'medium',
        'face_missing': 'medium',
        'environment_change': 'info',
        'tab_switch': 'warning'
    }
    severity = severity_map.get(event_type, 'low')

    snapshot = data.get('snapshot')  # Optional small data URL image for evidence
    event = {
        'examId': str(data['examId']),
        'userId': str(data['userId']),
        'eventType': event_type,
        'details': data.get('details', {}),
        'severity': severity,
        'timestamp': datetime.datetime.utcnow()
    }
    if snapshot and isinstance(snapshot, str):
        # Store snapshot both under details and as unified frameEvidence for consistency
        try:
            event['details'] = event.get('details', {})
            event['details']['snapshot'] = snapshot
            event['frameEvidence'] = snapshot
        except Exception:
            pass
    try:
        # Store in proctor_events_collection (main collection)
        result = proctor_events_collection.insert_one(event)
        event_id = str(result.inserted_id)

        # Also store in proctoring_logs_collection for report queries
        # Create a compatible format for reports
        log_entry = {
            'exam_id': str(data['examId']),
            'user_id': str(data['userId']),
            'violation_type': event_type,
            'details': event.get('details', {}),
            'severity': severity,
            'timestamp': datetime.datetime.utcnow(),
            # Link log -> proctor event for 1:1 evidence mapping
            'eventId': event_id,
        }
        proctoring_logs_collection.insert_one(log_entry)

        return jsonify({'message': 'Event recorded', 'eventId': event_id}), 201
    except Exception as e:
        print(f"[PROCTOR-EVENT] ERROR saving event: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/exams/<exam_id>/proctoring', methods=['GET'])
def get_proctoring_summary(exam_id):
    """Return a summary of proctoring events for an exam, grouped by student."""
    req_user = _get_authenticated_user_doc()
    if not req_user:
        return jsonify({'error': 'Authentication required'}), 403
    if req_user.get('role') != 'lecturer':
        return jsonify({'error': 'Forbidden: lecturer role required'}), 403

    try:
        # Fetch all events for the exam and group by userId in Python so we can
        # build per-event-type counts and ensure safe JSON conversion.
        cursor = proctor_events_collection.find({'examId': str(exam_id)})
        events_by_user = {}
        for ev in cursor:
            uid = str(ev.get('userId'))
            events_by_user.setdefault(uid, []).append(ev)

        summary = []
        for uid, evs in events_by_user.items():
            # counts by eventType
            counts = {}
            last_event = None
            for ev in evs:
                et = ev.get('eventType', 'unknown')
                counts[et] = counts.get(et, 0) + 1
                # determine last event by timestamp (timestamp stored as datetime)
                ts = ev.get('timestamp')
                if isinstance(ts, datetime.datetime):
                    if last_event is None or ts > last_event.get('timestamp'):
                        last_event = ev
                else:
                    # fallback if timestamp is string
                    last_event = ev

            # fetch user display name if possible
            user_doc = None
            try:
                user_doc = users_collection.find_one({'_id': ObjectId(uid)}) if ObjectId.is_valid(uid) else None
            except Exception:
                user_doc = None

            # Prepare lastEvent with serializable types
            last_ev_safe = None
            if last_event:
                last_ev_safe = last_event.copy()
                if isinstance(last_ev_safe.get('_id'), ObjectId):
                    last_ev_safe['_id'] = str(last_ev_safe['_id'])
                ts = last_ev_safe.get('timestamp')
                if isinstance(ts, datetime.datetime):
                    last_ev_safe['timestamp'] = ts.isoformat()

            summary.append({
                'userId': uid,
                'name': user_doc.get('name') if user_doc else uid,
                'count': len(evs),
                'countsByType': counts,
                'lastEvent': last_ev_safe
            })

        return jsonify({'summary': summary}), 200
    except Exception as e:
        print(f"Error getting proctoring summary: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/exams/<exam_id>/proctoring/<user_id>', methods=['GET'])
def get_proctoring_details(exam_id, user_id):
    """Return detailed proctor events for a student in an exam."""
    # Require lecturer role
    req_user = _get_authenticated_user_doc()
    if not req_user:
        return jsonify({'error': 'Authentication required'}), 403
    if req_user.get('role') != 'lecturer':
        return jsonify({'error': 'Forbidden: lecturer role required'}), 403

    try:
        # Sort by timestamp descending - newest events first
        docs = list(proctor_events_collection.find({'examId': str(exam_id), 'userId': str(user_id)}).sort('timestamp', -1))
        
        events = []
        for d in docs:
            ev = d.copy()
            ev['_id'] = str(ev.get('_id'))
            ts = ev.get('timestamp')
            if isinstance(ts, datetime.datetime):
                ev['timestamp'] = ts.isoformat()  # ISO format for frontend parsing
            events.append(ev)
        
        return jsonify({'events': events}), 200
    except Exception as e:
        print(f"[PROCTOR-FETCH] ERROR: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/exams/<exam_id>/proctoring/recent', methods=['GET'])
def get_proctoring_recent(exam_id):
    """Return proctoring events for an exam since a given ISO timestamp (query param since=).
    Useful for frequent polling by lecturer dashboard to get fast updates.
    """
    # Require lecturer role
    req_user = _get_authenticated_user_doc()
    if not req_user:
        return jsonify({'error': 'Authentication required'}), 403
    if req_user.get('role') != 'lecturer':
        return jsonify({'error': 'Forbidden: lecturer role required'}), 403

    since = request.args.get('since')
    limit = int(request.args.get('limit', 100))
    q = {'examId': str(exam_id)}
    if since:
        try:
            since_dt = datetime.datetime.fromisoformat(since)
            q['timestamp'] = {'$gt': since_dt}
        except Exception:
            pass

    try:
        # Sort by timestamp descending - newest events first
        docs = list(proctor_events_collection.find(q).sort('timestamp', -1).limit(limit))
        events = []
        for d in docs:
            ev = d.copy()
            ev['_id'] = str(ev.get('_id'))
            ts = ev.get('timestamp')
            if isinstance(ts, datetime.datetime):
                ev['timestamp'] = ts.isoformat()
            events.append(ev)
        return jsonify({'events': events}), 200
    except Exception as e:
        print(f"Error fetching recent proctor events: {e}")
        return jsonify({'error': str(e)}), 500

# ✅ PAGINATED PROCTOR EVENTS - Performance Optimization (CRITICAL)
@app.route('/api/proctor-events', methods=['GET'])
@limiter.limit("100 per hour")
def get_proctor_events():
    """
    CRITICAL: Paginated proctor events to prevent memory/performance issues.
    
    Returns paginated list of proctoring events for an exam, preventing memory
    and performance issues when dealing with thousands of events.
    
    Query Parameters:
        examId (str): Exam identifier (required)
        userId (str): User identifier to filter by specific student (optional)
        page (int): Page number starting at 1 (default: 1)
        limit (int): Events per page, max 200 (default: 50)
    
    Returns:
        200 OK: {events, page, limit, total, pages, hasNext, hasPrev}
        400 Bad Request: Missing examId or invalid parameters
        404 Not Found: Exam not found
        500 Internal Error: Database error
    """
    try:
        # 1. Get and validate query parameters
        exam_id = request.args.get('examId')
        user_id = request.args.get('userId')
        
        if not exam_id:
            return jsonify({"error": "examId query parameter is required"}), 400
        
        # Get pagination parameters
        try:
            page = int(request.args.get('page', 1))
            if page < 1:
                return jsonify({"error": "page must be >= 1"}), 400
        except ValueError:
            return jsonify({"error": "Invalid page parameter"}), 400
        
        try:
            limit = int(request.args.get('limit', 50))
            if limit < 1:
                return jsonify({"error": "limit must be >= 1"}), 400
            if limit > 200:
                return jsonify({"error": "limit cannot exceed 200"}), 400
        except ValueError:
            return jsonify({"error": "Invalid limit parameter"}), 400
        
        # 2. Validate exam exists
        try:
            exam = exams_collection.find_one({'_id': ObjectId(exam_id)}) if ObjectId.is_valid(exam_id) else None
            if not exam:
                return jsonify({"error": "Exam not found"}), 404
        except Exception as e:
            return jsonify({"error": "Invalid exam ID format"}), 400
        
        # 3. Build MongoDB query
        query = {'examId': str(exam_id)}
        if user_id:
            query['userId'] = str(user_id)
        
        # 4. Calculate skip for pagination
        skip = (page - 1) * limit
        
        # 5. Get total count for pagination info
        try:
            total = proctor_events_collection.count_documents(query)
        except Exception as e:
            app.logger.error(f"Error counting proctor events: {e}")
            return jsonify({"error": "Failed to count events"}), 500
        
        # 6. Query database with pagination (newest first)
        try:
            cursor = proctor_events_collection.find(query).sort('timestamp', -1).skip(skip).limit(limit)
            docs = list(cursor)
        except Exception as e:
            app.logger.error(f"Error fetching proctor events: {e}")
            return jsonify({"error": "Failed to fetch events"}), 500
        
        # 7. Convert ObjectId and datetime to JSON-serializable format
        events = []
        for doc in docs:
            event = doc.copy()
            
            # Convert ObjectId to string
            if '_id' in event and isinstance(event['_id'], ObjectId):
                event['_id'] = str(event['_id'])
            
            # Convert datetime to ISO format
            if 'timestamp' in event and isinstance(event['timestamp'], datetime.datetime):
                event['timestamp'] = event['timestamp'].isoformat() + 'Z'
            
            events.append(event)
        
        # 8. Calculate pagination metadata
        import math
        total_pages = math.ceil(total / limit) if total > 0 else 1
        has_next = page < total_pages
        has_prev = page > 1
        
        LIGHT_MODE = os.getenv('INVIGILO_LIGHT_MODE', '0') == '1'
        return jsonify({
            "events": events,
            "page": page,
            "limit": limit,
            "total": total,
            "pages": total_pages,
            "hasNext": has_next,
            "hasPrev": has_prev
        }), 200
        
    except Exception as e:
        app.logger.exception("Error in get_proctor_events")
        return jsonify({"error": "Internal processing error", "detail": str(e)}), 500


# ✅ UPLOAD EVIDENCE - Screen Recording and Screenshot Upload (CRITICAL)
@app.route('/api/upload-evidence', methods=['POST'])
@limiter.limit("100 per hour")
def upload_evidence():
    """
    CRITICAL: Upload screen recording, screenshot, or audio evidence for violations.
    
    Stores evidence files with references in database for dispute resolution and proof.
    Files are saved locally with organized directory structure by exam and user.
    
    Form Data:
        file (binary): Video, image, or audio file
        examId (str): Exam identifier
        userId (str): User who triggered violation
        evidenceType (str): Type of evidence (screenshot, video, audio)
        violationType (str): Optional - associated violation type
        violationScore (int): Optional - violation score
    
    Returns:
        200 OK: {message, url, fileId, filePath}
        400 Bad Request: Missing file or parameters
        404 Not Found: Exam not found
        500 Internal Error: File storage error
    """
    try:
        def _bytes_from_mb(mb: float) -> int:
            try:
                return int(float(mb) * 1024 * 1024)
            except Exception:
                return int(10 * 1024 * 1024)

        def _env_mb(name: str, default_mb: float) -> float:
            try:
                return float(os.getenv(name, str(default_mb)))
            except Exception:
                return float(default_mb)

        # 1. Validate required parameters
        exam_id = request.form.get('examId')
        user_id = request.form.get('userId')
        evidence_type = request.form.get('evidenceType', 'screenshot')
        violation_type = request.form.get('violationType', 'unknown')
        violation_score = request.form.get('violationScore', 0)
        event_id = request.form.get('eventId')
        
        if not exam_id:
            return jsonify({"error": "examId is required"}), 400
        
        if not user_id:
            return jsonify({"error": "userId is required"}), 400

        # Best-effort: if the client authenticates, enforce it matches the payload userId.
        requester = (_get_authenticated_user_id() or '').strip()
        if requester and requester != str(user_id):
            return jsonify({"error": "Forbidden"}), 403
        
        # 2. Get uploaded file
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # 3. Validate exam exists
        try:
            exam = exams_collection.find_one({'_id': ObjectId(exam_id)}) if ObjectId.is_valid(exam_id) else None
            if not exam:
                return jsonify({"error": "Exam not found"}), 404
        except Exception as e:
            return jsonify({"error": "Invalid exam ID format"}), 400
        
        evidence_type = str(evidence_type or 'screenshot').strip().lower()
        allowed_types = {'screenshot', 'image', 'video', 'audio'}
        if evidence_type not in allowed_types:
            return jsonify({"error": f"Invalid evidenceType: {evidence_type}"}), 400

        # 4. Validate request size early (best-effort; relies on Content-Length)
        max_mb_map = {
            'screenshot': _env_mb('EVIDENCE_MAX_MB_SCREENSHOT', 2.0),
            'image': _env_mb('EVIDENCE_MAX_MB_IMAGE', 2.0),
            'audio': _env_mb('EVIDENCE_MAX_MB_AUDIO', 6.0),
            'video': _env_mb('EVIDENCE_MAX_MB_VIDEO', 20.0),
        }
        max_bytes = _bytes_from_mb(max_mb_map.get(evidence_type, 2.0))
        try:
            content_len = int(request.content_length or 0)
        except Exception:
            content_len = 0
        if content_len and content_len > (max_bytes + (512 * 1024)):
            return jsonify({"error": "Evidence upload too large", "maxBytes": max_bytes}), 413

        # 5. Determine and validate file extension based on evidence type
        allowed_ext_map = {
            'screenshot': {'.jpg', '.jpeg', '.png'},
            'image': {'.jpg', '.jpeg', '.png'},
            'video': {'.webm', '.mp4'},
            'audio': {'.wav', '.mp3', '.ogg'},
        }
        default_ext_map = {
            'screenshot': '.jpg',
            'image': '.jpg',
            'video': '.webm',
            'audio': '.wav',
        }
        extension = default_ext_map.get(evidence_type, '.bin')

        original_ext = ''
        if '.' in file.filename:
            original_ext = os.path.splitext(file.filename)[1].lower()

        if original_ext and original_ext in allowed_ext_map.get(evidence_type, set()):
            extension = original_ext

        # Best-effort mimetype validation: reject only when it's clearly wrong.
        mimetype = (getattr(file, 'mimetype', None) or '').lower()
        if mimetype and mimetype not in ('application/octet-stream', 'binary/octet-stream'):
            if evidence_type in ('screenshot', 'image') and not mimetype.startswith('image/'):
                return jsonify({"error": "Invalid file type for image evidence"}), 415
            if evidence_type == 'video' and not mimetype.startswith('video/'):
                return jsonify({"error": "Invalid file type for video evidence"}), 415
            if evidence_type == 'audio' and not mimetype.startswith('audio/'):
                return jsonify({"error": "Invalid file type for audio evidence"}), 415
        
        # 6. Create directory structure: evidence/{examId}/{userId}/
        evidence_base = os.path.join(os.path.dirname(__file__), 'evidence')
        evidence_dir = os.path.join(evidence_base, str(exam_id), str(user_id))
        
        try:
            os.makedirs(evidence_dir, exist_ok=True)
        except Exception as e:
            app.logger.error(f"Failed to create evidence directory: {e}")
            return jsonify({"error": "Failed to create storage directory"}), 500
        
        # 7. Generate unique filename with timestamp
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        filename = f"{timestamp}_{evidence_type}{extension}"
        filepath = os.path.join(evidence_dir, filename)
        
        # 8. Save file to disk
        try:
            file.save(filepath)
            file_size = os.path.getsize(filepath)
        except Exception as e:
            app.logger.error(f"Failed to save evidence file: {e}")
            return jsonify({"error": "Failed to save file"}), 500

        # Enforce max size after-save too (in case Content-Length was missing).
        if file_size and int(file_size) > int(max_bytes):
            try:
                os.remove(filepath)
            except Exception:
                pass
            return jsonify({"error": "Evidence upload too large", "maxBytes": max_bytes}), 413
        
        # 9. Create relative path for URL and database storage
        relative_path = os.path.join('evidence', str(exam_id), str(user_id), filename)
        relative_path = relative_path.replace('\\', '/')  # Normalize for web URLs
        
        # 10. Store evidence record in database
        evidence_record = {
            'examId': str(exam_id),
            'userId': str(user_id),
            'type': 'evidence',
            'evidenceType': evidence_type,
            'violationType': violation_type,
            'filePath': relative_path,
            'fileSize': file_size,
            'timestamp': datetime.datetime.utcnow(),
            'violationScore': int(violation_score) if violation_score else 0
        }

        # Optional: link evidence to a specific proctor event (1:1 mapping)
        if event_id and isinstance(event_id, str):
            evidence_record['eventId'] = event_id
        
        try:
            result = proctor_events_collection.insert_one(evidence_record)
            file_id = str(result.inserted_id)
        except Exception as e:
            app.logger.error(f"Failed to store evidence record: {e}")
            # File is saved but database record failed - log warning
            app.logger.warning(f"Evidence file saved but DB record failed: {filepath}")
            return jsonify({"error": "Failed to store evidence record"}), 500
        
        # 11. Return success response
        return jsonify({
            "message": "Evidence uploaded successfully",
            "url": f"/api/evidence/{file_id}",
            "fileId": file_id,
            "filePath": relative_path,
            "fileSize": file_size,
            "evidenceType": evidence_type,
            "eventId": event_id
        }), 200
        
    except Exception as e:
        app.logger.exception("Error in upload_evidence")
        return jsonify({"error": "Internal processing error", "detail": str(e)}), 500


# ✅ GET EVIDENCE FILE - Retrieve Evidence for Proctors
@app.route('/api/evidence/<evidence_id>', methods=['GET'])
@limiter.limit("200 per hour")
def get_evidence(evidence_id):
    """
    Retrieve evidence file for authorized proctors.
    
    Returns the actual file (image, video, audio) for viewing/download.
    Only accessible by lecturers who have access to the exam.
    
    URL Parameters:
        evidence_id (str): Evidence record ID from database
    
    Returns:
        200 OK: File content with appropriate mime type
        403 Forbidden: Not authorized
        404 Not Found: Evidence not found or file missing
        500 Internal Error: File retrieval error
    """
    try:
        # 1. Require lecturer role (basic security)
        req_user = _get_authenticated_user_doc()
        if not req_user:
            return jsonify({'error': 'Authentication required'}), 403
        if req_user.get('role') != 'lecturer':
            return jsonify({'error': 'Forbidden: lecturer role required'}), 403
        
        # 2. Get evidence record from database
        try:
            evidence = proctor_events_collection.find_one({'_id': ObjectId(evidence_id)}) if ObjectId.is_valid(evidence_id) else None
            if not evidence:
                return jsonify({"error": "Evidence not found"}), 404
        except Exception as e:
            return jsonify({"error": "Invalid evidence ID format"}), 400
        
        # 3. Verify evidence type
        if evidence.get('type') != 'evidence':
            return jsonify({"error": "Invalid evidence record"}), 400
        
        # 4. Get file path
        file_path = evidence.get('filePath')
        if not file_path:
            return jsonify({"error": "File path not found in record"}), 404
        
        # 5. Construct absolute file path
        absolute_path = os.path.join(os.path.dirname(__file__), file_path)
        
        # 6. Verify file exists
        if not os.path.exists(absolute_path):
            app.logger.error(f"Evidence file not found: {absolute_path}")
            return jsonify({"error": "Evidence file not found on disk"}), 404
        
        # 7. Determine mime type based on file extension
        mime_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webm': 'video/webm',
            '.mp4': 'video/mp4',
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.ogg': 'audio/ogg'
        }
        
        file_ext = os.path.splitext(absolute_path)[1].lower()
        mime_type = mime_type_map.get(file_ext, 'application/octet-stream')
        
        # 8. Send file
        from flask import send_file
        return send_file(
            absolute_path,
            mimetype=mime_type,
            as_attachment=False,  # Display inline in browser
            download_name=os.path.basename(absolute_path)
        )
        
    except Exception as e:
        app.logger.exception("Error in get_evidence")
        return jsonify({"error": "Internal processing error", "detail": str(e)}), 500


# ✅ LIST EVIDENCE - Get All Evidence for an Exam/User
@app.route('/api/evidence', methods=['GET'])
@limiter.limit("100 per hour")
def list_evidence():
    """
    List all evidence records for an exam or user.
    
    Query Parameters:
        examId (str): Filter by exam (required)
        userId (str): Filter by specific user (optional)
        evidenceType (str): Filter by type (screenshot, video, audio) (optional)
    
    Returns:
        200 OK: {evidence: [...]}
        400 Bad Request: Missing examId
        403 Forbidden: Not authorized
        500 Internal Error: Database error
    """
    try:
        # 1. Require lecturer role
        req_user = _get_authenticated_user_doc()
        if not req_user:
            return jsonify({'error': 'Authentication required'}), 403
        if req_user.get('role') != 'lecturer':
            return jsonify({'error': 'Forbidden: lecturer role required'}), 403
        
        # 2. Get query parameters
        exam_id = request.args.get('examId')
        user_id = request.args.get('userId')
        evidence_type = request.args.get('evidenceType')
        
        if not exam_id:
            return jsonify({"error": "examId query parameter is required"}), 400
        
        # 3. Build query
        query = {
            'examId': str(exam_id),
            'type': 'evidence'
        }
        
        if user_id:
            query['userId'] = str(user_id)
        
        if evidence_type:
            query['evidenceType'] = evidence_type
        
        # 4. Query database
        try:
            cursor = proctor_events_collection.find(query).sort('timestamp', -1)
            docs = list(cursor)
        except Exception as e:
            app.logger.error(f"Error querying evidence: {e}")
            return jsonify({"error": "Failed to query evidence"}), 500
        
        # 5. Format results
        evidence_list = []
        for doc in docs:
            evidence = {
                'id': str(doc.get('_id')),
                'examId': doc.get('examId'),
                'userId': doc.get('userId'),
                'evidenceType': doc.get('evidenceType'),
                'violationType': doc.get('violationType'),
                'eventId': doc.get('eventId'),
                'fileSize': doc.get('fileSize'),
                'violationScore': doc.get('violationScore'),
                'url': f"/api/evidence/{str(doc.get('_id'))}",
                'timestamp': doc.get('timestamp').isoformat() + 'Z' if isinstance(doc.get('timestamp'), datetime.datetime) else doc.get('timestamp')
            }
            evidence_list.append(evidence)
        
        return jsonify({
            "evidence": evidence_list,
            "count": len(evidence_list)
        }), 200
        
    except Exception as e:
        app.logger.exception("Error in list_evidence")
        return jsonify({"error": "Internal processing error", "detail": str(e)}), 500


# --- Recent Global Proctor Events ---
@app.route('/api/proctoring/recent-global', methods=['GET'])
def get_proctoring_recent_global():
    """Return newest proctoring events across all exams since a given ISO timestamp.
    Requires lecturer role. Sorted newest-first and limited by ?limit=.
    """
    req_user = _get_authenticated_user_doc()
    if not req_user:
        return jsonify({'error': 'Authentication required'}), 403
    if req_user.get('role') != 'lecturer':
        return jsonify({'error': 'Forbidden: lecturer role required'}), 403

    since = request.args.get('since')
    limit = int(request.args.get('limit', 100))
    q = {}
    if since:
        try:
            since_dt = datetime.datetime.fromisoformat(since)
            q['timestamp'] = {'$gt': since_dt}
        except Exception:
            pass
    try:
        docs = list(proctor_events_collection.find(q).sort('timestamp', -1).limit(limit))
        events = []
        for d in docs:
            ev = d.copy()
            ev['_id'] = str(ev.get('_id'))
            ts = ev.get('timestamp')
            if isinstance(ts, datetime.datetime):
                ev['timestamp'] = ts.isoformat()
            events.append(ev)
        return jsonify({'events': events}), 200
    except Exception as e:
        print(f"Error fetching recent global proctor events: {e}")
        return jsonify({'error': str(e)}), 500


# --- Recent Proctor Events for a specific user (student view) ---
@app.route('/api/proctoring/recent', methods=['GET'])
def get_proctoring_recent_user():
    """Return proctoring events for a specific user across all their exams.
    Query params: userId=<id>, limit=<num>, since=<iso_timestamp>
    Students can view their own events.
    """
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'userId is required'}), 400
    
    # Verify the requester is either the user themselves or a lecturer
    requester = _get_authenticated_user_id()
    if not requester:
        return jsonify({'error': 'Authentication required'}), 403
    req_user = _get_authenticated_user_doc()
    
    if not req_user:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Allow if requester is the user themselves or a lecturer
    is_self = str(requester) == str(user_id)
    is_lecturer = req_user.get('role') == 'lecturer'
    
    if not (is_self or is_lecturer):
        return jsonify({'error': 'Forbidden'}), 403

    since = request.args.get('since')
    limit = int(request.args.get('limit', 100))
    q = {'userId': str(user_id)}
    if since:
        try:
            since_dt = datetime.datetime.fromisoformat(since)
            q['timestamp'] = {'$gt': since_dt}
        except Exception:
            pass

    try:
        docs = list(proctor_events_collection.find(q).sort('timestamp', -1).limit(limit))
        events = []
        for d in docs:
            ev = d.copy()
            ev['_id'] = str(ev.get('_id'))
            ts = ev.get('timestamp')
            if isinstance(ts, datetime.datetime):
                ev['timestamp'] = ts.isoformat()
            events.append(ev)
        return jsonify({'events': events}), 200
    except Exception as e:
        print(f"Error fetching recent user proctor events: {e}")
        return jsonify({'error': str(e)}), 500


# --- Student Attempt Retrieval ---
@app.route('/api/exams/<exam_id>/attempt', methods=['GET'])
def get_student_attempt(exam_id):
    """Return the attempt for a specific user on a given exam.
    Query param: userId=<id>
    """
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'userId is required'}), 400
    try:
        exam = exams_collection.find_one({'_id': ObjectId(exam_id)})
        if not exam:
            return jsonify({'error': 'Exam not found'}), 404
        attempts = exam.get('attempts', []) or []
        for a in reversed(attempts):
            if str(a.get('userId')) == str(user_id):
                att = a.copy()
                if isinstance(att.get('completedAt'), datetime.datetime):
                    att['completedAt'] = att['completedAt'].isoformat()
                return jsonify({'attempt': att}), 200
        return jsonify({'attempt': None}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- AI Question Generation ---
@app.route('/api/ai-generate-questions', methods=['POST'])
def ai_generate_questions():
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY is not configured on the server."}), 500

    data = request.get_json()
    topic = data.get('topic')
    difficulty = data.get('difficulty')
    num_questions = data.get('num_questions')
    question_type = data.get('question_type')

    if not all([topic, difficulty, num_questions, question_type]):
        return jsonify({"error": "Missing parameters for AI question generation"}), 400

    # Build detailed prompt for better AI generation
    print(f"[AI_GEN] Generating {num_questions} {question_type} questions about '{topic}' at {difficulty} difficulty")
    
    if question_type == 'multiple-choice':
        prompt = f"""You are an expert exam question writer. Generate {num_questions} {difficulty} difficulty multiple-choice questions specifically about "{topic}".

CRITICAL REQUIREMENTS:
- Each question MUST be a real, meaningful question about {topic}, NOT generic placeholders
- BAD EXAMPLE (DO NOT DO THIS): "Medium question 2 about Data science?"
- GOOD EXAMPLE: "Which algorithm is most suitable for classification tasks in supervised learning?"
- Write 4 DISTINCT answer options that are plausible and specific to the question
- BAD EXAMPLE (DO NOT DO THIS): "Option 1", "Option 2", "Option 3", "Option 4"
- GOOD EXAMPLE: "Decision Trees", "K-Means Clustering", "Linear Regression", "Random Forest"
- Set correctAnswer to the index (1-4) of the correct option
- Assign marks: Easy=1, Medium=2, Hard=3

Generate questions that demonstrate real knowledge of {topic}."""
    
    elif question_type == 'true-false':
        prompt = f"""You are an expert exam question writer. Generate {num_questions} {difficulty} difficulty true/false questions specifically about "{topic}".

CRITICAL REQUIREMENTS:
- Write clear, factual statements about {topic}, NOT generic placeholders
- Each statement should test real knowledge of {topic}
- Set correctAnswer to boolean true or false
- Assign 1 mark per question

Generate questions that demonstrate real knowledge of {topic}."""
    
    elif question_type == 'short-answer':
        prompt = f"""You are an expert exam question writer. Generate {num_questions} {difficulty} difficulty short-answer questions specifically about "{topic}".

CRITICAL REQUIREMENTS:
- Write clear questions requiring brief answers (1-2 sentences)
- Questions must test real knowledge of {topic}, NOT be generic
- Provide the expected correct answer
- Assign marks: Easy=2, Medium=3, Hard=4

Generate questions that demonstrate real knowledge of {topic}."""
    
    else:  # essay
        prompt = f"""You are an expert exam question writer. Generate {num_questions} {difficulty} difficulty essay questions specifically about "{topic}".

CRITICAL REQUIREMENTS:
- Write thought-provoking questions requiring detailed answers
- Questions must test deep understanding of {topic}, NOT be generic
- Provide key points expected in a good answer
- Assign marks: Easy=5, Medium=7, Hard=10

Generate questions that demonstrate real knowledge of {topic}."""
    
    print(f"[AI_GEN] Prompt length: {len(prompt)} characters")
    
    schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "type": {"type": "string", "enum": ["multiple-choice", "true-false", "short-answer", "essay"]},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "correctAnswer": {"type": "string"},
                        "marks": {"type": "integer"}
                    },
                    "required": ["question", "type", "correctAnswer", "marks"]
                }
            }
        },
        "required": ["questions"]
    }

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema
        }
    }
    # Helper: cache ListModels for a short time to avoid rate-limiting
    MODEL_CACHE = getattr(app, '_model_cache', {'ts': 0, 'models': []})
    CACHE_TTL = 300

    def get_listmodels():
        now = int(datetime.datetime.utcnow().timestamp())
        if MODEL_CACHE.get('models') and now - MODEL_CACHE.get('ts', 0) < CACHE_TTL:
            return MODEL_CACHE['models']
        list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        resp = requests.get(list_url)
        resp.raise_for_status()
        models = resp.json().get('models', [])
        MODEL_CACHE['models'] = models
        MODEL_CACHE['ts'] = now
        setattr(app, '_model_cache', MODEL_CACHE)
        return models

    def select_candidate_models(models):
        # Prefer models that advertise supportedMethods. Fallback to name heuristics.
        candidates = []
        for m in models:
            name = m.get('name')
            methods = m.get('supportedMethods') or []
            candidates.append({'name': name, 'methods': methods})

        # Sort: models with supportedMethods first
        candidates.sort(key=lambda x: 0 if x['methods'] else 1)

        # If supportedMethods is empty for most, add heuristic ordering for known text models
        heuristics = [
            r'gemini-2.5-pro', r'gemini-2.5-pro', r'gemini-2.5', r'gemini-pro', r'gemini-2.0-pro',
            r'gemini-2.5-flash', r'text-bison', r'gemma-3', r'gemini-flash'
        ]

        # Build final ordered list
        ordered = []
        seen = set()
        # first add those with supportedMethods listing generateText/generateContent
        for c in candidates:
            if 'generateText' in c['methods'] or 'generateContent' in c['methods']:
                ordered.append(c)
                seen.add(c['name'])

        # then add heuristic matches
        import re
        for pattern in heuristics:
            rx = re.compile(pattern, re.IGNORECASE)
            for c in candidates:
                if c['name'] in seen:
                    continue
                if rx.search(c['name'] or ''):
                    ordered.append(c)
                    seen.add(c['name'])

        # lastly append any remaining
        for c in candidates:
            if c['name'] not in seen:
                ordered.append(c)
                seen.add(c['name'])

        return ordered

    def try_call_model(candidate):
        name = candidate['name']
        methods = candidate.get('methods') or []

        # Attempt generateText (v1beta2) first if methods advertise it or if name looks like a newer model
        tried = []
        # build attempts: try generateText then generateContent
        attempts = []
        if 'generateText' in methods:
            attempts.append(('v1beta2', 'generateText'))
        if 'generateContent' in methods:
            attempts.append(('v1beta', 'generateContent'))
        # if no methods advertised, try generateText first for gemini-like names
        if not attempts:
            attempts = [('v1beta2', 'generateText'), ('v1beta', 'generateContent')]

        for ver, method in attempts:
            try:
                if method == 'generateText':
                    url = f"https://generativelanguage.googleapis.com/{ver}/{name}:generateText?key={GEMINI_API_KEY}"
                    body = {'prompt': {'text': prompt + "\n\nReturn only JSON that matches the requested schema."}, 'maxOutputTokens': 800}
                    r = requests.post(url, json=body)
                    print(f"[AI_GEN] Trying {name} {method} -> {r.status_code}")
                    r.raise_for_status()
                    j = r.json()
                    text_content = None
                    if isinstance(j, dict) and j.get('candidates'):
                        text_content = j['candidates'][0].get('output') or j['candidates'][0].get('content') or r.text
                    else:
                        text_content = r.text
                    print(f"[AI_GEN] SUCCESS with {name} via {method}, response length: {len(text_content) if text_content else 0}")
                    return text_content
                else:
                    url = f"https://generativelanguage.googleapis.com/{ver}/{name}:generateContent?key={GEMINI_API_KEY}"
                    r = requests.post(url, json=payload)
                    print(f"[AI_GEN] Trying {name} {method} -> {r.status_code}")
                    r.raise_for_status()
                    j = r.json()
                    try:
                        text_content = j['candidates'][0]['content']['parts'][0]['text']
                    except Exception:
                        text_content = r.text
                    print(f"[AI_GEN] SUCCESS with {name} via {method}, response length: {len(text_content) if text_content else 0}")
                    return text_content
            except requests.exceptions.RequestException as e:
                print(f"[AI_GEN] Model {name} via {method} failed: {e}")
                tried.append((name, method, str(e)))
                continue
        return None

    try:
        # If a selected model was previously saved (manual or auto), try it first
        SELECTED = getattr(app, '_selected_model', None)
        SELECTED_TTL = 60 * 60 * 24  # 24 hours
        if SELECTED:
            sel_ts = SELECTED.get('ts', 0)
            now_ts = int(datetime.datetime.utcnow().timestamp())
            if now_ts - sel_ts < SELECTED_TTL:
                print(f"Trying cached selected model: {SELECTED.get('name')}")
                candidate = {'name': SELECTED.get('name'), 'methods': SELECTED.get('methods', [])}
                text = try_call_model(candidate)
                if text:
                    try:
                        questions_json = json.loads(text)
                        return jsonify(questions_json), 200
                    except json.JSONDecodeError:
                        print("Cached selected model returned non-JSON; will fall back to selection process.")
                        # fall through to selection
                else:
                    print("Cached selected model failed; clearing cached selection.")
                    try:
                        delattr(app, '_selected_model')
                    except Exception:
                        app._selected_model = None

        models = get_listmodels()
        print(f"[AI_GEN] Found {len(models)} available Gemini models")
        candidates = select_candidate_models(models)
        print(f"[AI_GEN] Selected {len(candidates)} candidate models to try")
        # Try candidates in order until one returns JSON we can parse
        for cand in candidates:
            if not cand.get('name'):
                continue
            text = try_call_model(cand)
            if not text:
                continue
            try:
                questions_json = json.loads(text)
                print(f"[AI_GEN] Successfully parsed JSON response from {cand.get('name')}")
                # cache successful candidate for subsequent calls
                try:
                    app._selected_model = {'name': cand.get('name'), 'methods': cand.get('methods', []), 'ts': int(datetime.datetime.utcnow().timestamp())}
                except Exception:
                    pass
                return jsonify(questions_json), 200
            except json.JSONDecodeError:
                print(f"[AI_GEN] Model {cand.get('name')} returned non-JSON text; skipping. Preview: {text[:200]}")
                continue

        # If all remote attempts failed, fall back to the local generator
        print("[AI_GEN] WARNING: All Gemini API attempts failed, using local fallback (this will generate placeholder text)")
        def local_generate_questions(topic, num_questions, qtype, difficulty):
            print(f"[AI_GEN] Local fallback generating {num_questions} {qtype} questions")
            questions = []
            for i in range(int(num_questions)):
                q_text = f"{difficulty} question {i+1} about {topic}?"
                if qtype == 'multiple-choice':
                    opts = [f"Option {chr(65+j)} for {topic}" for j in range(4)]
                    questions.append({
                        'question': q_text,
                        'type': 'multiple-choice',
                        'options': opts,
                        'correctAnswer': 1,
                        'marks': 1
                    })
                elif qtype == 'true-false':
                    questions.append({
                        'question': q_text,
                        'type': 'true-false',
                        'correctAnswer': True,
                        'marks': 1
                    })
                else:
                    questions.append({
                        'question': q_text,
                        'type': qtype,
                        'correctAnswer': '',
                        'marks': 2
                    })
            return {'questions': questions}

        placeholder = local_generate_questions(topic, num_questions, question_type, difficulty)
        return jsonify(placeholder), 200

    except requests.exceptions.RequestException as e:
        print(f"[AI_GEN] ERROR: ListModels / API request exception during selection: {e}")
        # As a last-resort return local placeholder
        placeholder = {
            'questions': [{
                'question': f'{difficulty} question about {topic}?',
                'type': question_type,
                'correctAnswer': '',
                'marks': 1
            } for _ in range(int(num_questions))]
        }
        return jsonify(placeholder), 200
    except Exception as e:
        print(f"Unhandled exception in ai_generate_questions selection: {e}")
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

# Note: server start is performed at the end of the file after all route definitions.


@app.route('/api/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    """Update user profile fields. Allowed fields: name, phoneNumber, institution, department, year, studentId, lecturerId."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    allowed = {'name', 'phoneNumber', 'institution', 'department', 'year', 'studentId', 'lecturerId'}
    update = {k: v for k, v in data.items() if k in allowed}
    if not update:
        return jsonify({'error': 'No updatable fields provided'}), 400

    try:
        result = users_collection.update_one({'_id': ObjectId(user_id)}, {'$set': update})
        if result.matched_count == 0:
            return jsonify({'error': 'User not found'}), 404
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        user = serialize_doc(user)
        user = sanitize_user_response(user)  # Remove all sensitive fields
        return jsonify({'message': 'User updated', 'user': user}), 200
    except Exception as e:
        print(f"Error updating user {user_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<user_id>/face-samples', methods=['PUT'])
def add_face_samples(user_id):
    """Add one or more face samples to the user's enrollment. Enforces max samples.
    Body: { imageDataUrls: [dataUrl, ...] }
    """
    data = request.get_json()
    urls = data.get('imageDataUrls') if data else None
    if not urls or not isinstance(urls, list):
        return jsonify({'error': 'imageDataUrls (array) is required'}), 400
    max_samples = int(os.getenv('FACE_SAMPLES_MAX', '5'))
    try:
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        current = user.get('faceEmbeddings', []) or []
        if len(current) >= max_samples:
            return jsonify({'error': f'Max samples reached ({max_samples})'}), 400

        new_vecs = []
        # Embed each using ML service
        for u in urls:
            img = decode_base64_image(u)
            if img is None:
                continue
            
            # Convert to base64 for ML service
            try:
                _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Call ML service to generate embedding
                verify_payload = {'imageDataUrl': f"data:image/jpeg;base64,{img_base64}"}
                ok_verify, verify_result = call_ml_service('/verify-face', verify_payload, timeout=15)

                if ok_verify and isinstance(verify_result, dict) and 'embedding' in verify_result:
                    emb = verify_result['embedding']
                    new_vecs.append(emb)
                else:
                    app.logger.warning('ML service failed to generate embedding for image')
            except Exception as e:
                app.logger.warning(f'Error processing face sample: {e}')
            
            if len(current) + len(new_vecs) >= max_samples:
                break

        if not new_vecs:
            return jsonify({'error': 'No valid face samples were added. Please ensure your face is clearly visible.'}), 400

        updated = current + new_vecs
        # Ensure first faceEmbedding is set for backcompat if missing
        update = {'faceEmbeddings': updated}
        if not user.get('faceEmbedding') and len(updated) > 0:
            update['faceEmbedding'] = updated[0]
        users_collection.update_one({'_id': ObjectId(user_id)}, {'$set': update})
        return jsonify({'message': 'Samples added', 'count': len(updated), 'added': len(new_vecs), 'max': max_samples}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ================================================================================
# WebSocket Event Handlers for Real-Time Proctor Updates
# ================================================================================

@socketio.on('connect', namespace='/proctor')
def handle_proctor_connect():
    """
    Handle proctor client connection to WebSocket.
    
    Clients connect to the /proctor namespace for real-time violation updates.
    Connection is established before joining specific exam rooms.
    """
    try:
        client_id = request.sid

        # Optional but recommended: require signed token for Socket.IO.
        # Clients should pass ?token=<BearerToken> in the namespace connection query.
        require_auth = _bool_env('SOCKET_REQUIRE_AUTH', '1')
        token = (request.args.get('token') or '').strip()
        user_doc = None
        authed = False
        if token:
            ok, uid = _verify_token(token)
            if ok and uid:
                try:
                    user_doc = users_collection.find_one({'_id': ObjectId(uid)}) if ObjectId.is_valid(uid) else None
                except Exception:
                    user_doc = None
                if user_doc:
                    authed = True
                    SOCKET_AUTH[client_id] = {
                        'userId': str(user_doc.get('_id')),
                        'role': str(user_doc.get('role') or ''),
                    }

        if require_auth and not authed:
            emit('error', {
                'message': 'Unauthorized: token required',
                'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
            })
            try:
                disconnect()
            except Exception:
                pass
            return

        app.logger.info(f'[WEBSOCKET] Proctor client connected: {client_id}')
        emit('status', {
            'message': 'Connected to proctor updates',
            'clientId': client_id,
            'authenticated': bool(authed),
            'role': (SOCKET_AUTH.get(client_id) or {}).get('role') if authed else None,
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
        })
    except Exception as e:
        app.logger.error(f'[WEBSOCKET] Error in connect handler: {e}')


@socketio.on('disconnect', namespace='/proctor')
def handle_proctor_disconnect():
    """
    Handle proctor client disconnection from WebSocket.
    
    Automatically removes client from all rooms they joined.
    """
    try:
        client_id = request.sid
        try:
            SOCKET_AUTH.pop(client_id, None)
        except Exception:
            pass
        app.logger.info(f'[WEBSOCKET] Proctor client disconnected: {client_id}')
    except Exception as e:
        app.logger.error(f'[WEBSOCKET] Error in disconnect handler: {e}')


@socketio.on('join_exam', namespace='/proctor')
def handle_join_exam(data):
    """
    Join an exam room to receive real-time violation updates.
    
    Args:
        data (dict): Must contain 'examId' field
        
    Emits:
        - 'status': Success message when joined
        - 'error': Error message if exam not found or invalid
    """
    try:
        if not data or not isinstance(data, dict):
            emit('error', {'message': 'Invalid data format'})
            return
        
        exam_id = data.get('examId')
        if not exam_id:
            emit('error', {'message': 'examId is required'})
            return
        
        # Validate exam exists
        try:
            exam = exams_collection.find_one({'_id': ObjectId(exam_id)}) if ObjectId.is_valid(exam_id) else None
            if not exam:
                emit('error', {'message': 'Exam not found'})
                return
        except Exception as e:
            emit('error', {'message': 'Invalid exam ID format'})
            return

        # Only lecturers can join the exam broadcast room.
        # This prevents students from subscribing to other students' frames.
        allow_unauth = _bool_env('ALLOW_UNAUTH_JOIN_EXAM', '0')
        if not allow_unauth:
            sid = request.sid
            auth = SOCKET_AUTH.get(sid) or {}
            if not auth or auth.get('role') != 'lecturer':
                emit('error', {'message': 'Forbidden: lecturer role required'})
                return
        
        # Join the exam room
        join_room(exam_id, namespace='/proctor')
        
        client_id = request.sid
        app.logger.info(f'[WEBSOCKET] Client {client_id} joined exam room: {exam_id}')
        
        emit('status', {
            'message': f'Joined exam {exam_id}',
            'examId': exam_id,
            'examTitle': exam.get('title', 'Unknown'),
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
        })
        
    except Exception as e:
        app.logger.error(f'[WEBSOCKET] Error in join_exam: {e}')
        emit('error', {'message': 'Failed to join exam room'})


@socketio.on('leave_exam', namespace='/proctor')
def handle_leave_exam(data):
    """
    Leave an exam room to stop receiving violation updates.
    
    Args:
        data (dict): Must contain 'examId' field
        
    Emits:
        - 'status': Success message when left
        - 'error': Error message if invalid
    """
    try:
        if not data or not isinstance(data, dict):
            emit('error', {'message': 'Invalid data format'})
            return
        
        exam_id = data.get('examId')
        if not exam_id:
            emit('error', {'message': 'examId is required'})
            return
        
        # Leave the exam room
        leave_room(exam_id, namespace='/proctor')
        
        client_id = request.sid
        app.logger.info(f'[WEBSOCKET] Client {client_id} left exam room: {exam_id}')
        
        emit('status', {
            'message': f'Left exam {exam_id}',
            'examId': exam_id,
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
        })
        
    except Exception as e:
        app.logger.error(f'[WEBSOCKET] Error in leave_exam: {e}')
        emit('error', {'message': 'Failed to leave exam room'})


@socketio.on('ping', namespace='/proctor')
def handle_ping():
    """
    Handle ping from client to keep connection alive.
    
    Emits:
        - 'pong': Response to ping
    """
    emit('pong', {'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'})


@app.route('/api/exams/<exam_id>/students/<user_id>/violations', methods=['GET'])
def get_student_violations(exam_id, user_id):
    """
    Get all violations with frame evidence for a specific student.
    Used by lecturer to review incidents with captured images.
    
    Returns:
        200: List of violations with images, sorted by timestamp (newest first)
    """
    try:
        # Find all proctoring events for this student
        violations = list(proctor_events_collection.find({
            'examId': str(exam_id),
            'userId': str(user_id),
            'eventType': {'$in': ['gaze_extreme', 'gaze_frequency', 'gaze_sustained', 'identity_mismatch', 'multiple_faces', 'face_missing']}
        }).sort('timestamp', -1).limit(50))  # Last 50 violations
        
        # Format response
        formatted = []
        for v in violations:
            formatted.append({
                '_id': str(v['_id']),
                'eventType': v.get('eventType'),
                'severity': v.get('severity'),
                'timestamp': v.get('timestamp').isoformat() + 'Z' if v.get('timestamp') else None,
                'details': v.get('details', {}),
                'frameEvidence': v.get('frameEvidence'),  # Base64 image
                'reviewStatus': v.get('reviewStatus', 'pending'),  # pending, allowed, rejected
                'reviewedBy': v.get('reviewedBy'),
                'reviewedAt': v.get('reviewedAt').isoformat() + 'Z' if v.get('reviewedAt') else None
            })
        
        return jsonify({'violations': formatted}), 200
    
    except Exception as e:
        app.logger.error(f'Error fetching violations: {e}')
        return jsonify({'error': 'Failed to fetch violations'}), 500


@app.route('/api/exams/<exam_id>/students/<user_id>/violations/<violation_id>/review', methods=['POST'])
def review_violation(exam_id, user_id, violation_id):
    """
    Allow or reject a specific violation incident.
    
    Request Body:
        action (str): 'allow' or 'reject'
        reviewerId (str): Lecturer ID performing the review
        notes (str, optional): Additional review notes
    
    Returns:
        200: Success message
    """
    try:
        data = request.get_json()
        action = data.get('action')  # 'allow' or 'reject'
        reviewer_id = data.get('reviewerId')
        notes = data.get('notes', '')
        
        if action not in ['allow', 'reject']:
            return jsonify({'error': 'Invalid action. Must be "allow" or "reject"'}), 400
        
        if not reviewer_id:
            return jsonify({'error': 'reviewerId is required'}), 400
        
        # Update violation with review decision
        result = proctor_events_collection.update_one(
            {'_id': ObjectId(violation_id)},
            {
                '$set': {
                    'reviewStatus': 'allowed' if action == 'allow' else 'rejected',
                    'reviewedBy': str(reviewer_id),
                    'reviewedAt': datetime.datetime.utcnow(),
                    'reviewNotes': notes
                }
            }
        )
        
        if result.matched_count == 0:
            return jsonify({'error': 'Violation not found'}), 404
        
        # Adjust student risk score based on action
        if action == 'allow':
            # Reduce risk score when violation is approved/allowed
            violation = proctor_events_collection.find_one({'_id': ObjectId(violation_id)})
            if violation:
                risk_reduction = violation.get('details', {}).get('risk_score', 10)
                exam_attempts_collection.update_one(
                    {'exam_id': str(exam_id), 'user_id': str(user_id)},
                    {'$inc': {'risk_score': -risk_reduction}}
                )
                app.logger.info(f'[VIOLATION-REVIEW] Reduced risk score by {risk_reduction} for user {user_id}')
        elif action == 'reject':
            # Keep risk score as is and log confirmed violation
            proctor_events_collection.insert_one({
                'examId': str(exam_id),
                'userId': str(user_id),
                'eventType': 'violation_confirmed',
                'severity': 'high',
                'details': {
                    'originalViolationId': str(violation_id),
                    'reviewedBy': str(reviewer_id),
                    'notes': notes,
                    'message': 'Violation confirmed by lecturer review'
                },
                'timestamp': datetime.datetime.utcnow()
            })
        
        # Broadcast update to lecturer dashboard
        try:
            review_status = 'allowed' if action == 'allow' else 'rejected'
            socketio.emit('violation_reviewed', {
                'examId': str(exam_id),
                'userId': str(user_id),
                'violationId': str(violation_id),
                'reviewStatus': review_status,
                'reviewedBy': str(reviewer_id),
                'reviewedAt': datetime.datetime.utcnow().isoformat() + 'Z'
            }, room=str(exam_id), namespace='/proctor')
        except Exception as e:
            app.logger.error(f'Failed to broadcast violation review: {e}')
        
        review_status = 'allowed' if action == 'allow' else 'rejected'
        
        return jsonify({
            'message': f'Violation {action}ed successfully',
            'reviewStatus': review_status,
            'reviewedBy': str(reviewer_id),
            'reviewedAt': datetime.datetime.utcnow().isoformat() + 'Z',
            'violationId': str(violation_id)
        }), 200
    
    except Exception as e:
        app.logger.error(f'Error reviewing violation: {e}')
        return jsonify({'error': 'Failed to review violation'}), 500


@socketio.on('student-video-frame', namespace='/proctor')
def handle_student_video_frame(data):
    """
    Receive video frame from student and broadcast to lecturers monitoring this exam.
    
    Args:
        data (dict): Contains examId, userId, frame (base64 image), timestamp
    """
    try:
        exam_id = data.get('examId')
        user_id = data.get('userId')
        frame = data.get('frame')
        
        if not exam_id or not user_id or not frame:
            app.logger.warning('[VIDEO] Missing required fields in video frame')
            return

        # Require authenticated sender, and require sender matches declared userId.
        allow_unauth = _bool_env('ALLOW_UNAUTH_STUDENT_ROOMS', '0')
        if not allow_unauth:
            auth = SOCKET_AUTH.get(request.sid) or {}
            sender_user_id = auth.get('userId')
            if not sender_user_id or str(sender_user_id) != str(user_id):
                app.logger.warning('[VIDEO] Dropping frame: unauth or user mismatch')
                return
        
        # Broadcast to all lecturers monitoring this exam (send to exam room)
        socketio.emit(
            'video-frame',
            {
                'userId': user_id,
                'frame': frame,
                'timestamp': data.get('timestamp', int(datetime.datetime.utcnow().timestamp() * 1000))
            },
            room=exam_id,
            namespace='/proctor'
        )
        
    except Exception as e:
        app.logger.error(f'[VIDEO] Error handling video frame: {e}')


@socketio.on('student-screen-frame', namespace='/proctor')
def handle_student_screen_frame(data):
    """
    Receive screen capture frame from student and broadcast to lecturers monitoring this exam.
    
    Args:
        data (dict): Contains examId, userId, frame (base64 image), timestamp
    """
    try:
        exam_id = data.get('examId')
        user_id = data.get('userId')
        frame = data.get('frame')
        
        if not exam_id or not user_id or not frame:
            app.logger.warning('[SCREEN] Missing required fields in screen frame')
            return

        allow_unauth = _bool_env('ALLOW_UNAUTH_STUDENT_ROOMS', '0')
        if not allow_unauth:
            auth = SOCKET_AUTH.get(request.sid) or {}
            sender_user_id = auth.get('userId')
            if not sender_user_id or str(sender_user_id) != str(user_id):
                app.logger.warning('[SCREEN] Dropping frame: unauth or user mismatch')
                return
        
        # Broadcast to all lecturers monitoring this exam
        socketio.emit(
            'screen-frame',
            {
                'userId': user_id,
                'frame': frame,
                'timestamp': data.get('timestamp', int(datetime.datetime.utcnow().timestamp() * 1000))
            },
            room=exam_id,
            namespace='/proctor'
        )
        
    except Exception as e:
        app.logger.error(f'[SCREEN] Error handling screen frame: {e}')


def broadcast_violation(exam_id, user_id, violation_data):
    """
    Broadcast violation to all proctors monitoring this exam.
    
    Args:
        exam_id (str): Exam identifier
        user_id (str): User who triggered violation
        violation_data (dict): Violation details (type, severity, score, message)
    """
    try:
        socketio.emit(
            'violation_detected',
            {
                'userId': user_id,
                'examId': exam_id,
                'violationType': violation_data.get('type'),
                'severity': violation_data.get('severity'),
                'score': violation_data.get('score'),
                'message': violation_data.get('message'),
                'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
            },
            room=exam_id,
            namespace='/proctor'
        )
        app.logger.debug(f'[WEBSOCKET] Broadcasted violation to exam room {exam_id}')
    except Exception as e:
        app.logger.error(f'[WEBSOCKET] Error broadcasting violation: {e}')


@socketio.on('join_student', namespace='/proctor')
def handle_join_student(data):
    """Join a student-specific room (examId:userId) for targeted proctor decisions.

    NOTE: For now this is best-effort and not strongly authenticated.
    """
    try:
        if not data or not isinstance(data, dict):
            emit('error', {'message': 'Invalid data format'})
            return

        exam_id = data.get('examId')
        user_id = data.get('userId')
        if not exam_id or not user_id:
            emit('error', {'message': 'examId and userId are required'})
            return

        # Only the student themselves (or a lecturer) may join examId:userId rooms.
        allow_unauth = _bool_env('ALLOW_UNAUTH_STUDENT_ROOMS', '0')
        if not allow_unauth:
            auth = SOCKET_AUTH.get(request.sid) or {}
            requester_id = auth.get('userId')
            requester_role = auth.get('role')
            if not requester_id:
                emit('error', {'message': 'Unauthorized'})
                return
            if requester_role != 'lecturer' and str(requester_id) != str(user_id):
                emit('error', {'message': 'Forbidden'})
                return

        room_name = f"{exam_id}:{user_id}"
        join_room(room_name, namespace='/proctor')
        emit('status', {
            'message': 'Joined student room',
            'room': room_name,
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
        })

        # Immediately emit current status so student UI can sync on connect
        try:
            current = get_proctor_decision(exam_id, user_id)
            emit('student_paused', {
                'examId': str(exam_id),
                'userId': str(user_id),
                'status': current.get('status', 'active'),
                'reason': current.get('reason'),
                'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
            })
        except Exception:
            pass

    except Exception as e:
        app.logger.error(f'[WEBSOCKET] Error in join_student: {e}')
        emit('error', {'message': 'Failed to join student room'})


# ================================================================================
# Proctor Decisions / Student Pausing
# ================================================================================

# In-memory cache for fast reads (best-effort; DB is source of truth)
# Key: (examId, userId) -> {status, reason, updatedAt}
PROCTOR_DECISIONS = {}


def _decision_key(exam_id: str, user_id: str):
    return (str(exam_id), str(user_id))


def get_proctor_decision(exam_id: str, user_id: str):
    """Return current proctor decision for a student in an exam.

    Status values:
      - active: student can continue
      - paused: student must pause (awaiting lecturer)
      - terminated: student must stop/submit
    """
    k = _decision_key(exam_id, user_id)
    cached = PROCTOR_DECISIONS.get(k)
    if cached:
        return cached

    try:
        doc = proctor_events_collection.find_one(
            {
                'type': 'proctor_decision',
                'examId': str(exam_id),
                'userId': str(user_id),
            },
            sort=[('updatedAt', -1), ('timestamp', -1)]
        )
    except Exception:
        doc = None

    if not doc:
        dec = {
            'examId': str(exam_id),
            'userId': str(user_id),
            'status': 'active',
            'reason': None,
            'updatedAt': datetime.datetime.utcnow().isoformat() + 'Z'
        }
        PROCTOR_DECISIONS[k] = dec
        return dec

    updated_at = doc.get('updatedAt') or doc.get('timestamp')
    if isinstance(updated_at, datetime.datetime):
        updated_at = updated_at.isoformat() + 'Z'

    dec = {
        'examId': str(exam_id),
        'userId': str(user_id),
        'status': doc.get('status') or 'active',
        'reason': doc.get('reason'),
        'updatedAt': updated_at
    }
    PROCTOR_DECISIONS[k] = dec
    return dec


def set_proctor_decision(exam_id: str, user_id: str, status: str, reason: str = None, actor_id: str = None):
    k = _decision_key(exam_id, user_id)
    now = datetime.datetime.utcnow()
    dec = {
        'examId': str(exam_id),
        'userId': str(user_id),
        'status': status,
        'reason': reason,
        'updatedAt': now.isoformat() + 'Z'
    }
    PROCTOR_DECISIONS[k] = dec

    # Persist an audit trail entry
    try:
        proctor_events_collection.insert_one({
            'type': 'proctor_decision',
            'examId': str(exam_id),
            'userId': str(user_id),
            'status': status,
            'reason': reason,
            'actorId': str(actor_id) if actor_id else None,
            'updatedAt': now,
            'timestamp': now
        })
    except Exception as e:
        app.logger.error(f"[PROCTOR-DECISION] Failed to persist decision: {e}")

    # Broadcast to lecturer dashboard(s) and the student client.
    try:
        socketio.emit(
            'proctor_decision',
            {
                'examId': str(exam_id),
                'userId': str(user_id),
                'status': status,
                'reason': reason,
                'timestamp': now.isoformat() + 'Z'
            },
            room=str(exam_id),
            namespace='/proctor'
        )
        # Student-specific room (best-effort; requires student to join it)
        socketio.emit(
            'student_paused',
            {
                'examId': str(exam_id),
                'userId': str(user_id),
                'status': status,
                'reason': reason,
                'timestamp': now.isoformat() + 'Z'
            },
            room=f"{exam_id}:{user_id}",
            namespace='/proctor'
        )
    except Exception as e:
        app.logger.error(f"[PROCTOR-DECISION] Failed to broadcast decision: {e}")

    return dec


@app.route('/api/exams/<exam_id>/students/<user_id>/proctor-status', methods=['GET'])
def api_get_proctor_status(exam_id, user_id):
    """Get current proctor decision status for a student.

    Authorization:
      - lecturer can query any student
      - student can query themselves
    """
    requester = _get_authenticated_user_id()
    if not requester:
        return jsonify({'error': 'Authentication required'}), 403
    req_user = _get_authenticated_user_doc()
    if not req_user:
        return jsonify({'error': 'Unauthorized'}), 403

    is_self = str(requester) == str(user_id)
    is_lecturer = req_user.get('role') == 'lecturer'
    if not (is_self or is_lecturer):
        return jsonify({'error': 'Forbidden'}), 403

    dec = get_proctor_decision(exam_id, user_id)
    return jsonify({'status': dec}), 200


@app.route('/api/exams/<exam_id>/students/<user_id>/proctor-status', methods=['POST'])
def api_set_proctor_status(exam_id, user_id):
    """Set proctor decision status for a student (lecturer only)."""
    requester = _get_authenticated_user_id()
    if not requester:
        return jsonify({'error': 'Authentication required'}), 403
    req_user = _get_authenticated_user_doc()
    if not req_user or req_user.get('role') != 'lecturer':
        return jsonify({'error': 'Forbidden: lecturer role required'}), 403

    body = request.get_json() or {}
    status = (body.get('status') or '').strip().lower()
    reason = body.get('reason')

    allowed = {'active', 'paused', 'terminated'}
    if status not in allowed:
        return jsonify({'error': f"Invalid status. Allowed: {sorted(list(allowed))}"}), 400

    dec = set_proctor_decision(exam_id, user_id, status=status, reason=reason, actor_id=requester)
    return jsonify({'status': dec}), 200


@socketio.on('pause_student', namespace='/proctor')
def handle_pause_student(data):
    """Lecturer pauses a student's exam."""
    try:
        exam_id = data.get('examId')
        user_id = data.get('userId')
        
        if not exam_id or not user_id:
            emit('error', {'message': 'examId and userId are required'})
            return
        
        app.logger.info(f'[PROCTOR] Pausing student {user_id} in exam {exam_id}')
        
        # Update decision
        set_proctor_decision(
            exam_id=exam_id,
            user_id=user_id,
            status='paused',
            reason='Paused by lecturer',
            actor_id=None
        )
        
        # Immediately notify the student via their personal room
        student_room = f"{exam_id}:{user_id}"
        socketio.emit('student_paused', {
            'examId': exam_id,
            'userId': user_id,
            'status': 'paused',
            'reason': 'Paused by lecturer',
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
        }, room=student_room, namespace='/proctor')
        
        emit('status', {
            'message': f'Student {user_id} paused successfully',
            'examId': exam_id,
            'userId': user_id,
            'status': 'paused'
        })
        
    except Exception as e:
        app.logger.error(f'[PROCTOR] Error pausing student: {e}')
        emit('error', {'message': 'Failed to pause student'})


@socketio.on('stop_student', namespace='/proctor')
def handle_stop_student(data):
    """Lecturer stops/terminates a student's exam - ENFORCES ZERO SCORE."""
    try:
        exam_id = data.get('examId')
        user_id = data.get('userId')
        reason = data.get('reason', 'Removed due to suspicious behavior')
        
        if not exam_id or not user_id:
            emit('error', {'message': 'examId and userId are required'})
            return
        
        app.logger.info(f'[PROCTOR] Stopping student {user_id} in exam {exam_id}')
        
        # Update decision
        set_proctor_decision(
            exam_id=exam_id,
            user_id=user_id,
            status='terminated',
            reason=reason,
            actor_id=None
        )
        
        # CRITICAL: Force final score to ZERO in attempt record
        try:
            exam_attempts_collection.update_one(
                {
                    'exam_id': exam_id,
                    'user_id': user_id
                },
                {
                    '$set': {
                        'status': 'terminated_by_proctor',
                        'score': 0,
                        'percentage': 0,
                        'correct_count': 0,
                        'final_score': 0,
                        'score_overridden': True,
                        'override_reason': reason,
                        'terminated_at': datetime.datetime.utcnow(),
                        'terminated_by': 'lecturer',
                        'end_time': datetime.datetime.utcnow()  # Mark as completed
                    }
                },
                upsert=True
            )
            app.logger.info(f'[PROCTOR] Forced zero score for terminated student {user_id}')
        except Exception as e:
            app.logger.error(f'[PROCTOR] Error setting zero score: {e}')
        
        # Immediately notify the student via their personal room
        student_room = f"{exam_id}:{user_id}"
        socketio.emit('student_paused', {
            'examId': exam_id,
            'userId': user_id,
            'status': 'terminated',
            'reason': reason,
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
        }, room=student_room, namespace='/proctor')
        
        emit('status', {
            'message': f'Student {user_id} stopped successfully',
            'examId': exam_id,
            'userId': user_id,
            'status': 'terminated'
        })
        
        app.logger.info(f'[PROCTOR] Student {user_id} terminated and notified')
        
    except Exception as e:
        app.logger.error(f'[PROCTOR] Error stopping student: {e}')
        emit('error', {'message': 'Failed to stop student'})


@socketio.on('allow_student', namespace='/proctor')
def handle_allow_student(data):
    """Lecturer allows a paused student to continue."""
    try:
        exam_id = data.get('examId')
        user_id = data.get('userId')
        
        if not exam_id or not user_id:
            emit('error', {'message': 'examId and userId are required'})
            return
        
        app.logger.info(f'[PROCTOR] Allowing student {user_id} to continue in exam {exam_id}')
        
        # Update decision
        set_proctor_decision(
            exam_id=exam_id,
            user_id=user_id,
            status='active',
            reason='Allowed by lecturer',
            actor_id=None
        )
        
        # Immediately notify the student via their personal room
        student_room = f"{exam_id}:{user_id}"
        socketio.emit('student_paused', {
            'examId': exam_id,
            'userId': user_id,
            'status': 'active',
            'reason': 'Allowed by lecturer',
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
        }, room=student_room, namespace='/proctor')
        
        emit('status', {
            'message': f'Student {user_id} allowed to continue',
            'examId': exam_id,
            'userId': user_id,
            'status': 'active'
        })
        
    except Exception as e:
        app.logger.error(f'[PROCTOR] Error allowing student: {e}')
        emit('error', {'message': 'Failed to allow student'})
        actor_id=None
        
        emit('status', {
            'message': f'Student {user_id} allowed to continue',
            'examId': exam_id,
            'userId': user_id,
            'status': 'active'
        })
        
    except Exception as e:
        app.logger.error(f'[PROCTOR] Error allowing student: {e}')
        emit('error', {'message': 'Failed to allow student'})


# --- Internal tuning dashboard (protected) ---
@app.route('/internal/proctor-events', methods=['GET'])
def internal_proctor_events_dashboard():
        """Minimal internal dashboard to inspect recent proctor events and their metrics.

        Query params:
            examId, userId, eventType, severity, limit (default 200)
            token (optional, if INTERNAL_DASHBOARD_TOKEN is set)
        """
        if not _internal_allowed():
                return jsonify({"error": "forbidden"}), 403

        exam_id = (request.args.get('examId') or '').strip()
        user_id = (request.args.get('userId') or '').strip()
        event_type = (request.args.get('eventType') or '').strip()
        severity = (request.args.get('severity') or '').strip()

        try:
                limit = int(request.args.get('limit') or 200)
        except Exception:
                limit = 200
        limit = max(1, min(1000, limit))

        q = {}
        if exam_id:
                q['examId'] = exam_id
        if user_id:
                q['userId'] = user_id
        if event_type:
                q['eventType'] = event_type
        if severity:
                q['severity'] = severity

        try:
                cur = proctor_events_collection.find(q).sort('timestamp', -1).limit(limit)
                events = list(cur)
        except Exception as e:
                return jsonify({"error": "query_failed", "detail": str(e)}), 500

        counts = Counter([str(ev.get('eventType') or '') for ev in events])
        top_counts = counts.most_common(12)

        def esc(s: str) -> str:
                return (s or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

        rows = []
        for ev in events:
                ts = ev.get('timestamp')
                ts_s = ts.isoformat() + 'Z' if hasattr(ts, 'isoformat') else str(ts)
                det = ev.get('details') if isinstance(ev.get('details'), dict) else {}
                msg = det.get('message') if isinstance(det, dict) else ''
                metrics = ev.get('metrics') if isinstance(ev.get('metrics'), dict) else {}
                rows.append(
                        "<tr>"
                        f"<td>{esc(ts_s)}</td>"
                        f"<td>{esc(str(ev.get('eventType')))}</td>"
                        f"<td>{esc(str(ev.get('severity')))}</td>"
                        f"<td>{esc(str(ev.get('examId')))}</td>"
                        f"<td>{esc(str(ev.get('userId')))}</td>"
                        f"<td>{esc(str(msg))}</td>"
                        f"<td><pre style='margin:0;white-space:pre-wrap'>{esc(json.dumps(metrics, default=str, sort_keys=True))}</pre></td>"
                        "</tr>"
                )

        counts_html = " ".join([f"<span style='margin-right:10px'><b>{esc(k)}</b>: {v}</span>" for k, v in top_counts])

        token_val = (request.args.get('token') or '').strip()
        token_input = (
                f"<input name=\"token\" value=\"{esc(token_val)}\" type=\"hidden\">"
                if os.getenv('INTERNAL_DASHBOARD_TOKEN', '').strip() else ''
        )

        html = f"""<!doctype html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Invigilo - Proctor Events</title>
</head>
<body style='font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; padding: 16px;'>
    <h2 style='margin:0 0 8px 0;'>Proctor Events (internal)</h2>
    <div style='margin: 8px 0 12px 0; color: #333;'>Top event counts: {counts_html}</div>

    <form method='get' style='margin: 0 0 12px 0;'>
        <label>examId <input name='examId' value='{esc(exam_id)}' style='width: 260px;'></label>
        <label>userId <input name='userId' value='{esc(user_id)}' style='width: 260px;'></label>
        <label>eventType <input name='eventType' value='{esc(event_type)}' style='width: 160px;'></label>
        <label>severity <input name='severity' value='{esc(severity)}' style='width: 120px;'></label>
        <label>limit <input name='limit' value='{limit}' style='width: 80px;'></label>
        {token_input}
        <button type='submit'>Refresh</button>
    </form>

    <table border='1' cellpadding='6' cellspacing='0' style='border-collapse: collapse; width: 100%; font-size: 12px;'>
        <thead>
            <tr>
                <th>timestamp</th>
                <th>eventType</th>
                <th>severity</th>
                <th>examId</th>
                <th>userId</th>
                <th>message</th>
                <th>metrics</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
</body>
</html>"""

        return html, 200, {'Content-Type': 'text/html; charset=utf-8'}


# ================================================================================
# Main Application Entry Point
# ================================================================================

if __name__ == '__main__':
    # DEV-friendly server start. Set DEV_MODE=true in .env to enable Flask debug.
    # By default run without Flask debug/reloader to avoid child process behavior on Windows.
    DEV_MODE = os.getenv('DEV_MODE', 'false').lower() in ('1', 'true', 'yes')
    port = int(os.getenv('PORT', 5000))
    # Configure basic file logging to capture uncaught exceptions and server errors
    import logging
    from logging.handlers import RotatingFileHandler
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    log_file = os.path.join(os.path.dirname(__file__), 'server_error.log')
    handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    handler.setFormatter(log_formatter)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    # also configure root logger
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    print(f"Starting Invigilo server on 0.0.0.0:{port} (DEV_MODE={DEV_MODE}, ML_SERVICE_URL={os.getenv('ML_SERVICE_URL', 'Not set')})")

    # Install a global exception hook to log uncaught exceptions to file so the terminal doesn't silently close
    import sys, traceback
    def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # allow keyboard interrupts to exit normally
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        app.logger.error('Uncaught exception', exc_info=(exc_type, exc_value, exc_traceback))
        # also print to console for immediate visibility
        print('Uncaught exception:', exc_value)

    sys.excepthook = log_uncaught_exceptions

    print(f"About to start Flask server with threading enabled...")
    print(f"Flask app object: {app}")
    print(f"Flask routes registered: {[str(rule) for rule in app.url_map.iter_rules()][:5]}")  # Show first 5 routes
    
    # Run the app inside a restart loop so transient errors don't silently close the terminal.
    max_restarts = 3
    restarts = 0
    while True:
        try:
            # Enable threaded mode to handle multiple concurrent requests + WebSocket connections
            # This is critical for exam submissions to work while proctoring is active
            print(f"Starting Flask-SocketIO on http://0.0.0.0:{port} with threading={True}")
            import sys
            sys.stdout.flush()  # Force flush output

            # Configure Flask-SocketIO to use threaded mode
            print("Calling socketio.run() with threading support...")
            socketio.run(
                app,
                host='0.0.0.0',
                port=port,
                debug=False,
                use_reloader=False,
                allow_unsafe_werkzeug=True  # Allow for development
            )
            print("Flask has exited normally")
            break
        except Exception as e:
            app.logger.exception('Flask server crashed with exception: %s', e)
            print(f"Flask crashed: {e}")
            import traceback
            traceback.print_exc()
            restarts += 1
            if restarts >= max_restarts:
                print(f"Server crashed {restarts} times; giving up. See {log_file} for details.")
                break
            print(f"Server crashed, restarting ({restarts}/{max_restarts}) in 1s...")
            time.sleep(1)


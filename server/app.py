import os
import re
from flask import Flask, jsonify, request, redirect, url_for
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_socketio import SocketIO, emit, join_room, leave_room
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

def _bool_env(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


# ============================================================================
# ML SERVICE CLIENT (Hugging Face Spaces)
# ============================================================================
# Backend delegates heavy ML to separate service on Hugging Face Spaces
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "")
ML_SERVICE_TIMEOUT = 30  # seconds

def call_ml_service(endpoint: str, payload: dict, timeout: int = ML_SERVICE_TIMEOUT):
    """
    Call ML service endpoint via HTTP.
    
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
    
    url = ML_SERVICE_URL.rstrip('/') + endpoint
    try:
        print(f"[ML-CLIENT] Calling {url}")
        response = requests.post(url, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            print(f"[ML-CLIENT] ERROR: {response.status_code} - {response.text}")
            return False, {"error": f"ML service error: {response.status_code}"}
    
    except requests.Timeout:
        print(f"[ML-CLIENT] ERROR: Request timeout after {timeout}s")
        return False, {"error": "ML service timeout"}
    except Exception as e:
        print(f"[ML-CLIENT] ERROR: {e}")
        return False, {"error": str(e)}


# ============================================================================
# NO LOCAL ML - All ML processing delegated to ml-service
# ============================================================================
# This backend is ML-free for lightweight deployment on Render.
# All face recognition, proctoring analysis happens via HTTP calls to ML service.

# --- Setup ---
load_dotenv()
app = Flask(__name__)

app.secret_key = os.getenv('FLASK_SECRET_KEY', os.getenv('SECRET_KEY', 'dev-secret-change-me'))

def _sign_token(user_id: str, ttl_seconds: int = 3600):
    """Create a simple HMAC-signed token (no external deps).

    Format: v1.<user_id>.<exp_ts>.<sig>
    """
    exp = int(time.time()) + int(ttl_seconds)
    msg = f"v1.{user_id}.{exp}".encode('utf-8')
    key = app.secret_key.encode('utf-8')
    sig = hmac.new(key, msg, hashlib.sha256).hexdigest()
    return f"v1.{user_id}.{exp}.{sig}"

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
socketio = SocketIO(
    app,
    cors_allowed_origins=ALLOWED_ORIGINS,
    # Use gevent for async mode (compatible with modern Python + PyMongo)
    async_mode='gevent',
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
# ADMIN ROLE ENFORCEMENT
# ================================================================================
def require_admin():
    """
    Verifies that the current request is from an admin user.
    Checks X-User-Id header and validates admin role.
    
    Returns:
        tuple: (is_admin: bool, user: dict or None, error_response: tuple or None)
            - is_admin: True if user is admin, False otherwise
            - user: User document if found, None otherwise
            - error_response: (jsonify, status_code) tuple if not admin, None if admin
    
    Usage:
        is_admin, user, error = require_admin()
        if not is_admin:
            return error  # Returns 403 Forbidden or 401 Unauthorized
        # Continue with admin action...
    """
    # Get user ID from header
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return False, None, (jsonify({
            'error': 'Unauthorized',
            'message': 'X-User-Id header required'
        }), 401)
    
    # Validate ObjectId format
    if not ObjectId.is_valid(user_id):
        return False, None, (jsonify({
            'error': 'Unauthorized',
            'message': 'Invalid user ID format'
        }), 401)
    
    # Fetch user from database
    try:
        user = users_collection.find_one({'_id': ObjectId(user_id)})
    except Exception as e:
        app.logger.error(f"Error fetching user for admin check: {e}")
        return False, None, (jsonify({
            'error': 'Internal error',
            'message': 'Failed to verify user'
        }), 500)
    
    # Check if user exists
    if not user:
        return False, None, (jsonify({
            'error': 'Unauthorized',
            'message': 'User not found'
        }), 401)
    
    # Check if user has admin role
    if user.get('role') != 'admin':
        app.logger.warning(f"Non-admin user {user_id} attempted to access admin endpoint: {request.path}")
        return False, user, (jsonify({
            'error': 'Forbidden',
            'message': 'Only admins can perform this action'
        }), 403)
    
    # User is admin
    return True, user, None

def log_admin_action(admin_id, action, details=None):
    """
    Logs admin actions to audit_logs collection for security and compliance.
    
    Args:
        admin_id (str): The ObjectId of the admin user
        action (str): The action performed (e.g., 'delete_lecturer', 'update_settings')
        details (dict): Additional details about the action (e.g., deleted user ID, old/new values)
    
    Example:
        log_admin_action(admin_id, 'delete_lecturer', {
            'lecturer_id': lect_id,
            'lecturer_email': 'john@example.com'
        })
    """
    try:
        log_entry = {
            'admin_id': admin_id,
            'action': action,
            'details': details or {},
            'timestamp': datetime.datetime.utcnow(),
            'ip_address': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'endpoint': request.path
        }
        audit_logs_collection.insert_one(log_entry)
        app.logger.info(f"Admin action logged: {action} by {admin_id}")
    except Exception as e:
        # Don't fail the request if logging fails, but log the error
        app.logger.error(f"Failed to log admin action: {e}")

# ================================================================================
# END ADMIN ROLE ENFORCEMENT
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

# Simple health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "timestamp": datetime.datetime.utcnow().isoformat()}), 200

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
audit_logs_collection = None  # For admin action logging

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
    audit_logs_collection = db['audit_logs']  # For admin action logging
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
@app.route('/api/register', methods=['POST'])
#@limiter.limit("5 per hour")
def register_user():
    print('[REGISTER] Received registration request')
    db_ok, db_err = require_db()
    if not db_ok:
        print('[REGISTER] ERROR: DB not configured (MONGO_URI missing)')
        return db_err

    data = request.get_json()
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

        hashed_pw = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())

        new_user = {
            "name": data['fullName'],
            "email": data['email'],
            "phoneNumber": data['phoneNumber'],
            "role": data['role'],
            "password": hashed_pw,
            "institution": data['institution'],
            "department": data['department'],
            # Backcompat: keep the first embedding in faceEmbedding; store all in faceEmbeddings
            "faceEmbedding": (face_vectors[0] if face_vectors else None),
            "faceEmbeddings": face_vectors,
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
    except Exception as e:
        print(f"[REGISTER] Registration error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

# ✅ LOGIN
@app.route('/api/login', methods=['POST'])
#@limiter.limit("10 per hour")
def login_user():
    data = request.get_json()
    identifier, password, role = data.get('identifier'), data.get('password'), data.get('role')
    if not all([identifier, password, role]):
        return jsonify({"error": "Missing fields"}), 400

    user = users_collection.find_one({
        "role": role,
        "$or": [{"email": identifier}, {"phoneNumber": identifier},
                {"studentId": identifier}, {"lecturerId": identifier}]
    })

    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        user = serialize_doc(user)
        user = sanitize_user_response(user)  # Remove all sensitive fields
        return jsonify({"message": "Login successful", "user": user}), 200
    return jsonify({"error": "Invalid credentials"}), 401

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
        
        # 6. Call ML service for comprehensive frame analysis
        ml_payload = {
            'image': image_base64,
            'stored_embeddings': stored_embeddings,
            'face_threshold': get_face_threshold(0.56)
        }
        
        ml_result = call_ml_service('/analyze-frame', ml_payload, timeout=30)
        
        if not ml_result or 'violations' not in ml_result:
            app.logger.error(f"ML service returned invalid response: {ml_result}")
            return jsonify({"error": "ML service failed to analyze frame"}), 500
        
        violations = ml_result['violations']
        face_count = ml_result.get('face_count', 0)
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
def get_face_threshold(default_val=0.58):
    try:
        cfg = settings_collection.find_one({'key': 'FACE_SIMILARITY_THRESHOLD'})
        if cfg and isinstance(cfg.get('value'), (int, float)):
            return float(cfg['value'])
    except Exception:
        pass
    try:
        return float(os.getenv('FACE_SIMILARITY_THRESHOLD', str(default_val)))
    except Exception:
        return float(default_val)

# ✅ FACE VERIFICATION
@app.route('/api/verify-face', methods=['POST'])
#@limiter.limit("20 per hour")
def verify_face():
    print('[FACE-VERIFY] Received face verification request')
    data = request.get_json()
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
        # Convert image to base64 for ML service
        if ',' in image_data_url:
            image_base64 = image_data_url.split(',', 1)[1]
        else:
            image_base64 = image_data_url
        
        # Get stored embeddings
        stored_embeddings = []
        if user.get('faceEmbeddings') and isinstance(user.get('faceEmbeddings'), list):
            stored_embeddings = [e for e in user['faceEmbeddings'] if isinstance(e, (list, tuple))]
        if not stored_embeddings and user.get('faceEmbedding'):
            stored_embeddings = [user.get('faceEmbedding')]
        
        if not stored_embeddings:
            print('[FACE-VERIFY] ERROR: No stored embeddings found')
            return jsonify({"error": "No face data stored for user"}), 404
        
        # First, verify the face and get embedding from ML service
        verify_payload = {
            'image': image_base64
        }
        verify_result = call_ml_service('/verify-face', verify_payload, timeout=10)
        
        if not verify_result or 'embedding' not in verify_result:
            print('[FACE-VERIFY] ERROR: ML service failed to generate embedding')
            return jsonify({"error": "Failed to process face image"}), 400
        
        new_embedding = verify_result['embedding']
        
        # Now match against stored embeddings
        match_payload = {
            'embedding1': new_embedding,
            'stored_embeddings': stored_embeddings
        }
        match_result = call_ml_service('/match-face', match_payload, timeout=10)
        
        if not match_result or 'similarity' not in match_result:
            print('[FACE-VERIFY] ERROR: ML service failed to match faces')
            return jsonify({"error": "Failed to verify face"}), 500
        
        max_sim = match_result['similarity']
        sims = match_result.get('similarities', [max_sim])
        THRESHOLD = get_face_threshold(0.56)
        
        print(f'[FACE-VERIFY] Face verify for {identifier}: similarities={sims}, max={max_sim} thr={THRESHOLD}')
        
        if max_sim >= THRESHOLD:
            return jsonify({
                "message": "Face verified successfully",
                "verified": True,
                "similarity": float(max_sim),
                "similarities": sims
            }), 200
        else:
            return jsonify({
                "message": "Face verification failed",
                "verified": False,
                "similarity": float(max_sim),
                "similarities": sims
            }), 401

    except ValueError:
        return jsonify({"error": "No face detected"}), 400
    except Exception as e:
        app.logger.exception('Verification error')
        return jsonify({"error": "Internal verification error", "detail": str(e)}), 500

# --- (Keep all your other routes exactly as before) ---
# exams, proctoring, ai_generate_questions, etc. remain unchanged.



@app.route('/api/proctor', methods=['POST'])
@limiter.limit("100 per hour")
def proctor_activity():
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
    
    # Check frame brightness for camera blocking detection
    try:
        mean_brightness = np.mean(frame)
        
        # CRITICAL SECURITY CHECK: Distinguish between legitimate and suspicious blank frames
        if mean_brightness < 10:  # Nearly black frame detected
            
            if not exam_active:
                # LEGITIMATE: Exam submission in progress - camera stopped legitimately
                print(f"[PROCTOR] Skipping blank frame - exam inactive (brightness: {mean_brightness:.2f})")
                return jsonify({
                    "faceCount": 0,
                    "identityVerified": False,
                    "similarity": None,
                    "blinkStatus": "Unknown",
                    "gazeDirection": "Unknown",
                    "mouthStatus": "Unknown",
                    "headPose": "Unknown",
                    "message": "Blank frame - proctoring stopped"
                }), 200
            
            else:
                # SUSPICIOUS: Exam is ACTIVE but frame is blank - camera likely covered!
                print(f"[PROCTOR] ⚠️ CAMERA COVERED/BLOCKED during active exam! (brightness: {mean_brightness:.2f})")
                
                # Record HIGH severity event - this is cheating behavior
                try:
                    now = datetime.datetime.utcnow()
                    recent = proctor_events_collection.find_one({
                        'examId': str(exam_id),
                        'userId': str(user_id),
                        'eventType': 'camera_blocked',
                        'timestamp': {'$gt': now - datetime.timedelta(seconds=5)}
                    })
                    
                    if not recent:
                        frame_evidence = f"data:image/jpeg;base64,{frame_base64}"
                        
                        proctor_events_collection.insert_one({
                            'examId': str(exam_id),
                            'userId': str(user_id),
                            'eventType': 'camera_blocked',
                            'details': {
                                'brightness': float(mean_brightness),
                                'message': 'Camera blocked/covered during exam - possible cheating attempt'
                            },
                            'severity': 'high',
                            'timestamp': now,
                            'frameEvidence': frame_evidence
                        })
                        print(f"[PROCTOR-EVENT] camera_blocked (high) for user {user_id} in exam {exam_id}")
                except Exception as e:
                    print(f"[PROCTOR] Error recording camera_blocked event: {e}")
                
                # Return response indicating blocked camera
                return jsonify({
                    "faceCount": 0,
                    "identityVerified": False,
                    "similarity": None,
                    "blinkStatus": "Unknown",
                    "gazeDirection": "Unknown",
                    "mouthStatus": "Camera Blocked",
                    "headPose": "Camera Blocked",
                    "message": "Camera appears to be blocked or covered"
                }), 200
                
    except Exception as e:
        print(f"[PROCTOR] Error checking frame brightness: {e}")

    # Call ML service for full proctoring analysis
    try:
        ml_payload = {
            'image': frame_base64,
            'check_blink': True,
            'check_gaze': True,
            'check_mouth': True,
            'check_head_pose': True
        }
        
        ml_result = call_ml_service('/analyze-frame', ml_payload, timeout=30)
        
        if not ml_result:
            print("[PROCTOR] ML service unavailable")
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
        
        # Extract ML results
        face_count = ml_result.get('face_count', 0)
        blink_status = ml_result.get('blink_status', 'Unknown')
        gaze_direction = ml_result.get('gaze_direction', 'Unknown')
        mouth_status = ml_result.get('mouth_status', 'Unknown')
        head_pose = ml_result.get('head_pose', 'Unknown')
        
    except Exception as e:
        print(f"[PROCTOR] Error calling ML service: {e}")
        return jsonify({"error": "ML service error"}), 500

    # Identity verification using ML service
    identity_verified = False
    similarity_score = None
    try:
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if user and (user.get('faceEmbeddings') or user.get('faceEmbedding')):
            
            # Get stored embeddings
            stored_embeddings = []
            if user.get('faceEmbeddings') and isinstance(user.get('faceEmbeddings'), list):
                stored_embeddings = [e for e in user['faceEmbeddings'] if isinstance(e, (list, tuple))]
            elif user.get('faceEmbedding'):
                stored_embeddings = [user['faceEmbedding']]
            
            if stored_embeddings:
                # Generate embedding from current frame via ML service
                verify_payload = {'image': frame_base64}
                verify_result = call_ml_service('/verify-face', verify_payload, timeout=10)
                
                if verify_result and 'embedding' in verify_result:
                    # Match against stored embeddings
                    match_payload = {
                        'embedding1': verify_result['embedding'],
                        'stored_embeddings': stored_embeddings
                    }
                    match_result = call_ml_service('/match-face', match_payload, timeout=10)
                    
                    if match_result and 'similarity' in match_result:
                        similarity_score = float(match_result['similarity'])
                        THRESH_P = get_face_threshold(0.56)
                        identity_verified = similarity_score >= THRESH_P
                        app.logger.info(f'Proctor identity for {user_id}: similarity={similarity_score} threshold={THRESH_P}')
                    else:
                        app.logger.warning('ML service face matching failed')
                else:
                    app.logger.warning('ML service face verification failed')
            else:
                app.logger.warning(f'No stored embeddings for user {user_id}')
    except Exception as e:
        print(f"Error during identity verification: {e}")

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
    # Compute avg brightness to approximate environment changes
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_b = float(np.mean(hsv[...,2]))
    except Exception:
        avg_b = None

    # Immediate event recording - no tolerance thresholds
    try:
        if exam_id and user_id:
            now = datetime.datetime.utcnow()
            key = (str(exam_id), str(user_id))
            st = PROCTOR_STATE.get(key) or {
                'last_emit': {},  # Track last event time to prevent spam (5 second cooldown per event type)
                'reference_background': None
            }
            
            # Store reference background on first check (for background change detection)
            if st['reference_background'] is None and frame is not None:
                try:
                    ref_small = cv2.resize(frame, (160, 120))
                    st['reference_background'] = ref_small.copy()
                except Exception as e:
                    print(f"[PROCTOR] Error storing reference background: {e}")

            def emit(ev_type: str, details: dict, severity: str):
                # Short cooldown to prevent spam (5 seconds per event type)
                last_time = st['last_emit'].get(ev_type)
                if last_time and (now - last_time).total_seconds() < 5:
                    return
                
                # Capture frame evidence as base64 image
                frame_evidence = None
                if frame is not None:
                    try:
                        # Check if frame is blank/black/very dark - increased threshold to 30
                        mean_brightness = np.mean(frame)
                        if mean_brightness < 30:
                            print(f"[PROCTOR] Skipping event {ev_type} - blank/dark frame detected (brightness: {mean_brightness:.2f})")
                            return  # Don't record events with blank/dark frames
                        
                        # Encode frame to JPEG format
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        frame_evidence = f"data:image/jpeg;base64,{frame_base64}"
                    except Exception as e:
                        print(f"[PROCTOR] Error encoding frame: {e}")
                        return  # Don't record event if frame capture fails
                
                proctor_events_collection.insert_one({
                    'examId': str(exam_id),
                    'userId': str(user_id),
                    'eventType': ev_type,
                    'details': details,
                    'severity': severity,
                    'timestamp': now,
                    'frameEvidence': frame_evidence  # Store captured frame
                })
                st['last_emit'][ev_type] = now
                print(f"[PROCTOR-EVENT] {ev_type} ({severity}) for user {user_id} in exam {exam_id}")


            # IMMEDIATE EVENT DETECTION - Record all suspicious activity
            
            # 1. IDENTITY VERIFICATION - High severity
            if not identity_verified and similarity_score is not None:
                emit('identity_mismatch', {
                    'similarity': float(similarity_score),
                    'message': 'Face does not match registered student'
                }, 'high')
            
            # 2. MULTIPLE FACES - High severity (immediate detection)
            if face_count and face_count > 1:
                emit('multiple_faces', {
                    'count': face_count,
                    'message': f'{face_count} people detected in frame'
                }, 'high')
            
            # 3. NO FACE DETECTED - Medium severity
            if face_count == 0:
                emit('face_missing', {
                    'message': 'No face detected in frame'
                }, 'medium')
            
            # 4. HEAD POSE - Low severity (looking away)
            head_pose = results.get('headPose')
            if head_pose and str(head_pose).lower() not in ('forward', 'center', 'normal'):
                severity = 'medium' if str(head_pose).lower() in ('down', 'extreme_left', 'extreme_right') else 'low'
                emit('head_pose', {
                    'pose': head_pose,
                    'message': f'Head turned {head_pose}'
                }, severity)
            
            # 5. GAZE DIRECTION - Low severity (eyes looking away)
            gaze = results.get('gazeDirection')
            if gaze and str(gaze).lower() not in ('center', 'normal', 'forward'):
                emit('gaze_aversion', {
                    'direction': gaze,
                    'message': f'Eyes looking {gaze}'
                }, 'low')
            
            # 6. MOUTH STATUS - Medium severity (talking detected)
            mouth = results.get('mouthStatus')
            if mouth and str(mouth).lower() in ('open', 'talking', 'speaking'):
                emit('talking', {
                    'status': mouth,
                    'message': 'Mouth movement detected'
                }, 'medium')
            
            # 7. BACKGROUND CHANGE - High severity (detects hands, objects, people)
            if st['reference_background'] is not None and frame is not None:
                try:
                    current_small = cv2.resize(frame, (160, 120))
                    diff = cv2.absdiff(st['reference_background'], current_small)
                    diff_score = np.mean(diff)
                    
                    BACKGROUND_THRESHOLD = 20.0  # Detect significant background changes
                    if diff_score > BACKGROUND_THRESHOLD:
                        emit('background_change', {
                            'change_score': float(diff_score),
                            'message': 'Background changed - possible external assistance'
                        }, 'high')
                except Exception as e:
                    print(f"[PROCTOR] Error in background detection: {e}")

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
    
    # TODO: Implement audio processing via ML service or separate audio service
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
        
        # 2. Determine severity based on activity type
        severity_map = {
            'fullscreen_exit': 'critical',
            'tab_switch': 'high',
            'tab_unfocused': 'high',
            'window_blur': 'medium',
            'dev_tools_opened': 'critical',
            'dev_tools_attempt': 'high',
            'right_click': 'low',
            'copy_attempted': 'medium',
            'paste_attempted': 'medium',
            'print_screen': 'high',
            'screenshot_attempt': 'high',
            'multiple_monitors': 'medium',
            'browser_resize': 'low'
        }
        
        severity = severity_map.get(activity_type, 'medium')
        
        # 3. Calculate violation score
        score_map = {
            'critical': 50,
            'high': 30,
            'medium': 15,
            'low': 5
        }
        
        score = score_map.get(severity, 10)
        
        # 4. Create activity record
        activity_record = {
            'examId': str(exam_id),
            'userId': str(user_id),
            'type': 'suspicious_activity',
            'activityType': activity_type,
            'severity': severity,
            'score': score,
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
                    'score': score,
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
            "score": score
        }), 200
        
    except Exception as e:
        app.logger.exception("Error in log_suspicious_activity")
        return jsonify({"error": "Internal processing error", "detail": str(e)}), 500

# --- Exam Routes ---
@app.route('/api/exams', methods=['POST'])
def create_exam():
    data = request.get_json()
    required_fields = ['title', 'courseCode', 'lecturerId', 'institution', 'department', 'targetYear', 'questions']
    if not all(k in data for k in required_fields):
        return jsonify({"error": "Missing required fields for exam"}), 400
    
    questions_with_ids = []
    for q in data.get('questions', []):
        q_with_id = q.copy()
        q_with_id['_id'] = ObjectId()
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
    return jsonify({"message": "Exam created successfully", "exam": new_exam}), 201

@app.route('/api/exams/<exam_id>/status', methods=['PUT'])
def update_exam_status(exam_id):
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
        
        exam = exams_collection.find_one({'_id': ObjectId(exam_id)})
            
        if not exam:
            return jsonify({"error": "Exam not found"}), 404
        
        total_marks = 0
        score = 0
        per_question = []

        for question in exam.get('questions', []):
            q_id = str(question['_id'])
            marks = question.get('marks', 0) or 0
            total_marks += marks
            user_answer = answers.get(q_id)
            correct = False

            # Robust comparison by question type
            qtype = question.get('type')
            correct_answer = question.get('correctAnswer')

            try:
                if user_answer is not None:
                    # Multiple choice: support numeric (1-based) or option-text answers
                    if qtype == 'multiple-choice':
                        if isinstance(correct_answer, (int, float)):
                            try:
                                if int(user_answer) == int(correct_answer):
                                    correct = True
                            except Exception:
                                if str(user_answer) == str(correct_answer):
                                    correct = True
                        else:
                            if isinstance(user_answer, (int, float)):
                                try:
                                    idx = int(user_answer) - 1
                                    opts = question.get('options') or []
                                    if 0 <= idx < len(opts) and str(opts[idx]) == str(correct_answer):
                                        correct = True
                                except Exception:
                                    pass
                            else:
                                if str(user_answer) == str(correct_answer):
                                    correct = True

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

            per_question.append({
                'questionId': q_id,
                'question': question.get('question'),
                'given': user_answer,
                'expected': correct_answer,
                'marks': marks,
                'correct': correct
            })

        percentage = round((score / total_marks) * 100) if total_marks > 0 else 0

        attempt_record = {
            'userId': user_id,
            'score': percentage,
            'completedAt': datetime.datetime.utcnow().isoformat(),
            'perQuestion': per_question
        }

        # Store attempt and mark this user as completed for this exam
        exams_collection.update_one(
            {'_id': ObjectId(exam_id)},
            {
                '$push': {'attempts': attempt_record},
                '$addToSet': {'completedBy': user_id}
            }
        )

        return jsonify({
            'score': percentage,
            'totalMarks': total_marks,
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

            # compute canStart for this user: only if not completed and within scheduled window or status Available
            def within_window(ex):
                try:
                    date = datetime.datetime.fromisoformat(ex.get('scheduledDate')) if ex.get('scheduledDate') else None
                    if not date:
                        return ex.get('status') == 'Available'
                    sh, sm = (ex.get('startTime') or '00:00').split(':')
                    eh, em = (ex.get('endTime') or '23:59').split(':')
                    start = datetime.datetime(date.year, date.month, date.day, int(sh or 0), int(sm or 0))
                    end = datetime.datetime(date.year, date.month, date.day, int(eh or 23), int(em or 59))
                    now = datetime.datetime.utcnow()
                    return now >= start and now <= end
                except Exception:
                    return False

            exam_safe['canStartForUser'] = (exam_safe.get('status') == 'Available' or within_window(exam)) and not exam_safe['completedByUser']

        all_exams.append(exam_safe)

    return jsonify({"exams": all_exams}), 200


@app.route('/api/exams/<exam_id>/report', methods=['GET'])
def get_exam_report(exam_id):
    """Return detailed report for an exam: attempts, per-question stats, average score."""
    # Require lecturer role
    requester = request.headers.get('X-User-Id')
    if not requester:
        return jsonify({'error': 'X-User-Id header required'}), 403
    try:
        req_user = users_collection.find_one({'_id': ObjectId(requester)}) if ObjectId.is_valid(requester) else None
    except Exception:
        req_user = None
    if not req_user or req_user.get('role') != 'lecturer':
        return jsonify({'error': 'Forbidden: lecturer role required'}), 403

    try:
        exam = exams_collection.find_one({'_id': ObjectId(exam_id)})
        if not exam:
            return jsonify({'error': 'Exam not found'}), 404

        # Gather attempts (if any)
        attempts = exam.get('attempts', []) or []

        # Compute average score
        scores = [a.get('score', 0) for a in attempts if isinstance(a.get('score', None), (int, float))]
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0

        # Per-question aggregation
        q_stats = {}
        questions = exam.get('questions', [])
        # initialize stats for each question
        for q in questions:
            qid = str(q.get('_id'))
            q_stats[qid] = {
                'questionId': qid,
                'question': q.get('question'),
                'marks': q.get('marks', 0),
                'attempts': 0,
                'correctCount': 0
            }

        for att in attempts:
            perq = att.get('perQuestion', [])
            for pq in perq:
                qid = str(pq.get('questionId'))
                if qid not in q_stats:
                    q_stats[qid] = {
                        'questionId': qid,
                        'question': pq.get('question'),
                        'marks': pq.get('marks', 0),
                        'attempts': 0,
                        'correctCount': 0
                    }
                q_stats[qid]['attempts'] += 1
                if pq.get('correct'):
                    q_stats[qid]['correctCount'] += 1

        per_question_stats = []
        for qid, s in q_stats.items():
            attempts_count = s.get('attempts', 0)
            correct = s.get('correctCount', 0)
            ratio = round((correct / attempts_count) * 100, 2) if attempts_count > 0 else 0.0
            s['correctRatio'] = ratio
            per_question_stats.append(s)

        # Enrich attempts with user display names when possible
        enriched_attempts = []
        for a in attempts:
            a_copy = a.copy()
            uid = a_copy.get('userId')
            name = None
            try:
                if uid and ObjectId.is_valid(str(uid)):
                    user_doc = users_collection.find_one({'_id': ObjectId(uid)})
                    name = user_doc.get('name') if user_doc else None
            except Exception:
                name = None
            a_copy['userName'] = name or uid
            # ensure timestamp iso
            if isinstance(a_copy.get('completedAt'), datetime.datetime):
                a_copy['completedAt'] = a_copy['completedAt'].isoformat()
            enriched_attempts.append(a_copy)

        exam_safe = serialize_doc(exam)
        return jsonify({
            'exam': exam_safe,
            'averageScore': avg_score,
            'perQuestionStats': per_question_stats,
            'attempts': enriched_attempts
        }), 200
    except Exception as e:
        print(f"Error generating exam report for {exam_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/stats', methods=['GET'])
def get_admin_stats():
    """Return small set of stats for the admin dashboard (mock-friendly)."""
    # Require admin role
    is_admin, admin_user, error = require_admin()
    if not is_admin:
        return error

    try:
        total_students = users_collection.count_documents({'role': 'student'})
        live_exams = exams_collection.count_documents({'status': 'Live'})
        # active alerts in last 24 hours
        since = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
        active_alerts = proctor_events_collection.count_documents({'timestamp': {'$gte': since}})

        uptime_delta = datetime.datetime.utcnow() - APP_START
        uptime_hours = uptime_delta.total_seconds() / 3600.0
        # For display, mock uptime percentage as 99.9 if server has been up > 1 minute, else 100%
        uptime_percent = '99.9%'

        return jsonify({
            'totalStudents': total_students,
            'liveExams': live_exams,
            'activeAlerts': active_alerts,
            'systemUptime': uptime_percent,
            'serverUptimeHours': round(uptime_hours, 2)
        }), 200
    except Exception as e:
        print(f"Error getting admin stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/exams/<exam_id>', methods=['PUT'])
def update_exam(exam_id):
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
        # Store snapshot under details to keep schema simple
        try:
            event['details'] = event.get('details', {})
            event['details']['snapshot'] = snapshot
        except Exception:
            pass
    try:
        proctor_events_collection.insert_one(event)
        return jsonify({'message': 'Event recorded'}), 201
    except Exception as e:
        print(f"[PROCTOR-EVENT] ERROR saving event: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/exams/<exam_id>/proctoring', methods=['GET'])
def get_proctoring_summary(exam_id):
    """Return a summary of proctoring events for an exam, grouped by student."""
    # Simple role guard: require X-User-Id header of a lecturer
    requester = request.headers.get('X-User-Id')
    if not requester:
        return jsonify({'error': 'X-User-Id header required'}), 403
    try:
        req_user = users_collection.find_one({'_id': ObjectId(requester)}) if ObjectId.is_valid(requester) else None
    except Exception:
        req_user = None
    if not req_user or req_user.get('role') != 'lecturer':
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
    requester = request.headers.get('X-User-Id')
    if not requester:
        return jsonify({'error': 'X-User-Id header required'}), 403
    try:
        req_user = users_collection.find_one({'_id': ObjectId(requester)}) if ObjectId.is_valid(requester) else None
    except Exception:
        req_user = None
    if not req_user or req_user.get('role') != 'lecturer':
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
    requester = request.headers.get('X-User-Id')
    if not requester:
        return jsonify({'error': 'X-User-Id header required'}), 403
    try:
        req_user = users_collection.find_one({'_id': ObjectId(requester)}) if ObjectId.is_valid(requester) else None
    except Exception:
        req_user = None
    if not req_user or req_user.get('role') != 'lecturer':
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
        # 1. Validate required parameters
        exam_id = request.form.get('examId')
        user_id = request.form.get('userId')
        evidence_type = request.form.get('evidenceType', 'screenshot')
        violation_type = request.form.get('violationType', 'unknown')
        violation_score = request.form.get('violationScore', 0)
        
        if not exam_id:
            return jsonify({"error": "examId is required"}), 400
        
        if not user_id:
            return jsonify({"error": "userId is required"}), 400
        
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
        
        # 4. Determine file extension based on evidence type
        extension_map = {
            'screenshot': '.jpg',
            'video': '.webm',
            'audio': '.wav',
            'image': '.jpg'
        }
        extension = extension_map.get(evidence_type, '.bin')
        
        # Allow extension from original filename if available
        if '.' in file.filename:
            original_ext = os.path.splitext(file.filename)[1]
            if original_ext.lower() in ['.jpg', '.jpeg', '.png', '.webm', '.mp4', '.wav', '.mp3', '.ogg']:
                extension = original_ext.lower()
        
        # 5. Create directory structure: evidence/{examId}/{userId}/
        evidence_base = os.path.join(os.path.dirname(__file__), 'evidence')
        evidence_dir = os.path.join(evidence_base, str(exam_id), str(user_id))
        
        try:
            os.makedirs(evidence_dir, exist_ok=True)
        except Exception as e:
            app.logger.error(f"Failed to create evidence directory: {e}")
            return jsonify({"error": "Failed to create storage directory"}), 500
        
        # 6. Generate unique filename with timestamp
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        filename = f"{timestamp}_{evidence_type}{extension}"
        filepath = os.path.join(evidence_dir, filename)
        
        # 7. Save file to disk
        try:
            file.save(filepath)
            file_size = os.path.getsize(filepath)
        except Exception as e:
            app.logger.error(f"Failed to save evidence file: {e}")
            return jsonify({"error": "Failed to save file"}), 500
        
        # 8. Create relative path for URL and database storage
        relative_path = os.path.join('evidence', str(exam_id), str(user_id), filename)
        relative_path = relative_path.replace('\\', '/')  # Normalize for web URLs
        
        # 9. Store evidence record in database
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
        
        try:
            result = proctor_events_collection.insert_one(evidence_record)
            file_id = str(result.inserted_id)
        except Exception as e:
            app.logger.error(f"Failed to store evidence record: {e}")
            # File is saved but database record failed - log warning
            app.logger.warning(f"Evidence file saved but DB record failed: {filepath}")
            return jsonify({"error": "Failed to store evidence record"}), 500
        
        # 10. Return success response
        return jsonify({
            "message": "Evidence uploaded successfully",
            "url": f"/api/evidence/{file_id}",
            "fileId": file_id,
            "filePath": relative_path,
            "fileSize": file_size,
            "evidenceType": evidence_type
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
        requester = request.headers.get('X-User-Id')
        if not requester:
            return jsonify({'error': 'X-User-Id header required'}), 403
        
        try:
            req_user = users_collection.find_one({'_id': ObjectId(requester)}) if ObjectId.is_valid(requester) else None
        except Exception:
            req_user = None
        
        if not req_user or req_user.get('role') != 'lecturer':
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
        requester = request.headers.get('X-User-Id')
        if not requester:
            return jsonify({'error': 'X-User-Id header required'}), 403
        
        try:
            req_user = users_collection.find_one({'_id': ObjectId(requester)}) if ObjectId.is_valid(requester) else None
        except Exception:
            req_user = None
        
        if not req_user or req_user.get('role') != 'lecturer':
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
    requester = request.headers.get('X-User-Id')
    if not requester:
        return jsonify({'error': 'X-User-Id header required'}), 403
    try:
        req_user = users_collection.find_one({'_id': ObjectId(requester)}) if ObjectId.is_valid(requester) else None
    except Exception:
        req_user = None
    if not req_user or req_user.get('role') != 'lecturer':
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
    requester = request.headers.get('X-User-Id')
    if not requester:
        return jsonify({'error': 'X-User-Id header required'}), 403
    
    try:
        req_user = users_collection.find_one({'_id': ObjectId(requester)}) if ObjectId.is_valid(requester) else None
    except Exception:
        req_user = None
    
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

    prompt = f"Generate {num_questions} {difficulty} level questions for an exam on the topic of '{topic}'. The question type should be '{question_type}'."
    
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
                    print(f"Trying {name} {method} -> {r.status_code}")
                    r.raise_for_status()
                    j = r.json()
                    text_content = None
                    if isinstance(j, dict) and j.get('candidates'):
                        text_content = j['candidates'][0].get('output') or j['candidates'][0].get('content') or r.text
                    else:
                        text_content = r.text
                    return text_content
                else:
                    url = f"https://generativelanguage.googleapis.com/{ver}/{name}:generateContent?key={GEMINI_API_KEY}"
                    r = requests.post(url, json=payload)
                    print(f"Trying {name} {method} -> {r.status_code}")
                    r.raise_for_status()
                    j = r.json()
                    try:
                        text_content = j['candidates'][0]['content']['parts'][0]['text']
                    except Exception:
                        text_content = r.text
                    return text_content
            except requests.exceptions.RequestException as e:
                print(f"Model {name} via {method} failed: {e}")
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
        candidates = select_candidate_models(models)
        # Try candidates in order until one returns JSON we can parse
        for cand in candidates:
            if not cand.get('name'):
                continue
            text = try_call_model(cand)
            if not text:
                continue
            try:
                questions_json = json.loads(text)
                # cache successful candidate for subsequent calls
                try:
                    app._selected_model = {'name': cand.get('name'), 'methods': cand.get('methods', []), 'ts': int(datetime.datetime.utcnow().timestamp())}
                except Exception:
                    pass
                return jsonify(questions_json), 200
            except json.JSONDecodeError:
                print(f"Model {cand.get('name')} returned non-JSON text; skipping. Preview: {text[:200]}")
                continue

        # If all remote attempts failed, fall back to the local generator
        def local_generate_questions(topic, num_questions, qtype, difficulty):
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
        print(f"ListModels / API request exception during selection: {e}")
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


@app.route('/api/debug/models', methods=['GET'])
def debug_list_models():
    """Admin/debug endpoint: returns the ListModels output the server sees."""
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY not configured"}), 400
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        # Return a filtered summary to keep payload small
        summary = []
        for m in data.get('models', []):
            summary.append({
                'name': m.get('name'),
                'displayName': m.get('displayName'),
                'supportedMethods': m.get('supportedMethods')
            })
        return jsonify({'models': summary}), 200
    except Exception as e:
        print(f"Error fetching models: {e}")
        return jsonify({"error": str(e), "raw_response": getattr(e, 'response', None) and getattr(e.response, 'text', None)}), 500


@app.route('/api/debug/selected-model', methods=['GET'])
def debug_get_selected_model():
    sel = getattr(app, '_selected_model', None)
    if not sel:
        return jsonify({'selected': None}), 200
    return jsonify({'selected': sel}), 200


@app.route('/api/debug/selected-model', methods=['POST'])
def debug_set_selected_model():
    data = request.get_json()
    name = data.get('name')
    methods = data.get('methods', [])
    if not name:
        return jsonify({'error': 'Model name is required'}), 400
    try:
        app._selected_model = {'name': name, 'methods': methods, 'ts': int(datetime.datetime.utcnow().timestamp())}
        return jsonify({'selected': app._selected_model}), 200
    except Exception as e:
        print(f"Error setting selected model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/lecturers/<lect_id>', methods=['DELETE'])
def admin_delete_lecturer(lect_id):
    """Delete a lecturer and cascade-delete all their data: exams, attempts, proctor events.
    Note: This is a best-effort cascading delete without multi-document transactions (unless running on a replica set).
    """
    # Require admin role
    is_admin, admin_user, error = require_admin()
    if not is_admin:
        return error

    try:
        user = users_collection.find_one({'_id': ObjectId(lect_id)}) if ObjectId.is_valid(lect_id) else None
        if not user or user.get('role') != 'lecturer':
            return jsonify({'error': 'Lecturer not found'}), 404

        # Log the action before deletion
        log_admin_action(
            admin_id=str(admin_user['_id']),
            action='delete_lecturer',
            details={
                'lecturer_id': lect_id,
                'lecturer_email': user.get('email'),
                'lecturer_name': user.get('name')
            }
        )

        # Find all exams by this lecturer
        exam_docs = list(exams_collection.find({'lecturerId': lect_id}))
        exam_ids = [str(d.get('_id')) for d in exam_docs]

        # Delete proctor events for these exams
        pe_res = {'deletedCount': 0}
        if exam_ids:
            pe_res = proctor_events_collection.delete_many({'examId': {'$in': exam_ids}})

        # Delete exams
        ex_res = exams_collection.delete_many({'lecturerId': lect_id})

        # Finally, delete the lecturer account
        u_res = users_collection.delete_one({'_id': ObjectId(lect_id)})

        return jsonify({
            'message': 'Lecturer and related data deleted',
            'deletedProctorEvents': getattr(pe_res, 'deleted_count', 0),
            'deletedExams': getattr(ex_res, 'deleted_count', 0),
            'deletedLecturer': getattr(u_res, 'deleted_count', 0)
        }), 200
    except Exception as e:
        print(f"Error deleting lecturer {lect_id}: {e}")
        return jsonify({'error': str(e)}), 500


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
                verify_payload = {'image': img_base64}
                verify_result = call_ml_service('/verify-face', verify_payload, timeout=10)
                
                if verify_result and 'embedding' in verify_result:
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


@app.route('/api/admin/settings/face-threshold', methods=['GET'])
def get_face_threshold_setting():
    """Get current face recognition threshold setting. Admin only."""
    # Require admin role
    is_admin, admin_user, error = require_admin()
    if not is_admin:
        return error
    
    return jsonify({'threshold': get_face_threshold()}), 200


@app.route('/api/admin/settings/face-threshold', methods=['PUT'])
def set_face_threshold_setting():
    """Update face recognition threshold setting. Admin only."""
    # Require admin role
    is_admin, admin_user, error = require_admin()
    if not is_admin:
        return error
    
    body = request.get_json() or {}
    try:
        thr = float(body.get('threshold'))
        if thr <= 0 or thr > 1:
            return jsonify({'error': 'Threshold must be between 0 and 1'}), 400
    except Exception:
        return jsonify({'error': 'Invalid threshold'}), 400
    
    try:
        # Log the settings change
        old_threshold = get_face_threshold()
        log_admin_action(
            admin_id=str(admin_user['_id']),
            action='update_face_threshold',
            details={
                'old_threshold': old_threshold,
                'new_threshold': thr
            }
        )
        
        settings_collection.update_one({'key': 'FACE_SIMILARITY_THRESHOLD'}, {'$set': {'key': 'FACE_SIMILARITY_THRESHOLD', 'value': thr}}, upsert=True)
        return jsonify({'message': 'Threshold updated', 'threshold': thr}), 200
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
        app.logger.info(f'[WEBSOCKET] Proctor client connected: {client_id}')
        emit('status', {
            'message': 'Connected to proctor updates',
            'clientId': client_id,
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
    requester = request.headers.get('X-User-Id')
    if not requester:
        return jsonify({'error': 'X-User-Id header required'}), 403
    try:
        req_user = users_collection.find_one({'_id': ObjectId(requester)}) if ObjectId.is_valid(requester) else None
    except Exception:
        req_user = None
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
    requester = request.headers.get('X-User-Id')
    if not requester:
        return jsonify({'error': 'X-User-Id header required'}), 403
    try:
        req_user = users_collection.find_one({'_id': ObjectId(requester)}) if ObjectId.is_valid(requester) else None
    except Exception:
        req_user = None
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


"""
Celery Worker Configuration for Invigilo Proctoring System
============================================================

This module configures Celery for asynchronous background task processing.
Used for offloading heavy ML operations (face detection, identity verification)
from the main Flask request-response cycle.

Architecture:
- Redis as message broker and result backend
- Celery workers process ML tasks asynchronously
- Flask endpoints return 202 Accepted immediately
- Background workers handle frame analysis

Usage:
    # Start Celery worker
    celery -A celery_worker.celery_app worker --loglevel=info --pool=solo
    
    # With concurrency (multiple workers)
    celery -A celery_worker.celery_app worker --loglevel=info --concurrency=4

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
"""

import os
import sys
import json
import base64
import datetime
import traceback
from celery import Celery
from pymongo import MongoClient
from bson import ObjectId
import requests
import hmac
import hashlib

# ============================================================================
# CELERY CONFIGURATION
# ============================================================================

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
MONGO_URI = os.getenv('MONGO_URI')
ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', '')
ML_SHARED_SECRET = os.getenv('ML_SHARED_SECRET', '')

# Initialize Celery app
celery_app = Celery(
    'invigilo_tasks',
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=60,  # 60 seconds max per task
    task_soft_time_limit=50,  # Soft limit at 50 seconds
    worker_prefetch_multiplier=1,  # One task at a time per worker
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks (prevent memory leaks)
)

# ============================================================================
# DATABASE CONNECTION (Per-Task)
# ============================================================================

def get_db_collections():
    """Get MongoDB collections (creates new connection per task)"""
    if not MONGO_URI:
        raise RuntimeError('MONGO_URI not configured')
    
    client = MongoClient(MONGO_URI)
    db = client['invigilo_db']
    
    return {
        'users': db['users'],
        'exams': db['exams'],
        'proctor_events': db['proctor_events'],
        'exam_attempts': db['exam_attempts'],
    }

# ============================================================================
# ML SERVICE CLIENT (Reused from app.py)
# ============================================================================

def call_ml_service(endpoint: str, payload: dict, timeout: int = 30):
    """Call ML service with HMAC authentication"""
    if not ML_SERVICE_URL:
        print(f"[CELERY-ML] ERROR: ML_SERVICE_URL not configured")
        return False, {"error": "ML service not configured"}
    
    url = ML_SERVICE_URL.rstrip('/') + endpoint
    
    # Generate HMAC signature
    headers = {"Content-Type": "application/json"}
    if ML_SHARED_SECRET:
        try:
            payload_str = json.dumps(payload, sort_keys=True)
            signature = hmac.new(
                ML_SHARED_SECRET.encode('utf-8'),
                payload_str.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            headers["X-Signature"] = signature
        except Exception as e:
            print(f"[CELERY-ML] WARNING: Failed to generate signature: {e}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            print(f"[CELERY-ML] ERROR: {response.status_code} - {response.text}")
            return False, {"error": f"ML service error: {response.status_code}"}
    
    except requests.Timeout:
        print(f"[CELERY-ML] ERROR: Request timeout after {timeout}s")
        return False, {"error": "ML service timeout"}
    except Exception as e:
        print(f"[CELERY-ML] ERROR: {e}")
        return False, {"error": str(e)}

# ============================================================================
# CELERY TASKS
# ============================================================================

@celery_app.task(name='process_proctor_frame', bind=True, max_retries=2)
def process_proctor_frame(self, exam_id: str, user_id: str, frame_base64: str):
    """
    Asynchronous task: Process proctoring frame with ML analysis.
    
    Args:
        exam_id: Exam identifier
        user_id: Student user ID
        frame_base64: Base64-encoded JPEG frame
    
    Returns:
        dict: Analysis results (face_count, gaze, head_pose, etc.)
    """
    try:
        print(f"[CELERY-TASK] Processing frame for user {user_id} in exam {exam_id}")
        
        # Get database collections
        collections = get_db_collections()
        users_collection = collections['users']
        proctor_events_collection = collections['proctor_events']
        
        # Call ML service for frame analysis
        ml_payload = {
            'imageDataUrl': f"data:image/jpeg;base64,{frame_base64}"
        }
        
        ok_ml, ml_result = call_ml_service('/analyze-frame', ml_payload, timeout=30)
        
        if not ok_ml or not isinstance(ml_result, dict):
            print(f"[CELERY-TASK] ML service failed for user {user_id}")
            return {
                'success': False,
                'error': 'ML service unavailable',
                'user_id': user_id,
                'exam_id': exam_id
            }
        
        # Extract ML results
        face_count = ml_result.get('face_count', 0)
        blink_status = ml_result.get('blink_status', 'Unknown')
        gaze_direction = ml_result.get('gaze_direction', 'Unknown')
        mouth_status = ml_result.get('mouth_status', 'Unknown')
        head_pose = ml_result.get('head_pose', 'Unknown')
        
        # Identity verification (if needed)
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
                    verify_payload = {'imageDataUrl': f"data:image/jpeg;base64,{frame_base64}"}
                    ok_verify, verify_result = call_ml_service('/verify-face', verify_payload, timeout=15)
                    
                    if ok_verify and isinstance(verify_result, dict) and 'embedding' in verify_result:
                        # Compute cosine similarity locally
                        import numpy as np
                        
                        def cosine_similarity(a, b):
                            a = np.asarray(a, dtype=np.float32).reshape(-1)
                            b = np.asarray(b, dtype=np.float32).reshape(-1)
                            if a.size == 0 or b.size == 0 or a.shape != b.shape:
                                return None
                            denom = (np.linalg.norm(a) * np.linalg.norm(b))
                            if denom == 0:
                                return None
                            return float(np.dot(a, b) / denom)
                        
                        sims = []
                        for stored in stored_embeddings:
                            sim = cosine_similarity(verify_result['embedding'], stored)
                            if sim is not None:
                                sims.append(sim)
                        
                        if sims:
                            similarity_score = max(sims)
                            # Default threshold 0.58 (can be configurable)
                            identity_verified = similarity_score >= 0.58
        
        except Exception as e:
            print(f"[CELERY-TASK] Identity verification error: {e}")
        
        # Store results in database
        result = {
            'success': True,
            'exam_id': exam_id,
            'user_id': user_id,
            'face_count': face_count,
            'identity_verified': identity_verified,
            'similarity': similarity_score,
            'blink_status': blink_status,
            'gaze_direction': gaze_direction,
            'mouth_status': mouth_status,
            'head_pose': head_pose,
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
        }
        
        print(f"[CELERY-TASK] Completed processing for user {user_id}: face_count={face_count}, identity={identity_verified}")
        
        return result
        
    except Exception as e:
        print(f"[CELERY-TASK] ERROR processing frame: {e}")
        traceback.print_exc()
        
        # Retry on failure (max 2 retries)
        raise self.retry(exc=e, countdown=5)

@celery_app.task(name='process_violation_snapshot', bind=True, max_retries=1)
def process_violation_snapshot(self, exam_id: str, user_id: str, violation_type: str, frame_base64: str):
    """
    Asynchronous task: Process and store violation snapshot.
    
    Used in Phase 2 when client detects violations locally (MediaPipe).
    Backend validates and stores critical violations only.
    
    Args:
        exam_id: Exam identifier
        user_id: Student user ID
        violation_type: Type of violation (face_missing, multiple_faces, etc.)
        frame_base64: Base64-encoded JPEG frame evidence
    
    Returns:
        dict: Processing result
    """
    try:
        print(f"[CELERY-VIOLATION] Processing {violation_type} for user {user_id}")
        
        collections = get_db_collections()
        proctor_events_collection = collections['proctor_events']
        
        # Store violation event
        event = {
            'examId': exam_id,
            'userId': user_id,
            'eventType': violation_type,
            'severity': 'high' if violation_type in ['multiple_faces', 'identity_mismatch'] else 'medium',
            'details': {
                'message': f'Client-detected {violation_type}',
                'source': 'client_mediapipe'
            },
            'timestamp': datetime.datetime.utcnow(),
            'frameEvidence': f"data:image/jpeg;base64,{frame_base64}",
            'reviewStatus': 'pending'
        }
        
        result = proctor_events_collection.insert_one(event)
        
        print(f"[CELERY-VIOLATION] Stored violation {result.inserted_id} for user {user_id}")
        
        return {
            'success': True,
            'violation_id': str(result.inserted_id),
            'exam_id': exam_id,
            'user_id': user_id,
            'violation_type': violation_type
        }
        
    except Exception as e:
        print(f"[CELERY-VIOLATION] ERROR: {e}")
        traceback.print_exc()
        raise self.retry(exc=e, countdown=3)


# ============================================================================
# HEALTH CHECK TASK
# ============================================================================

@celery_app.task(name='health_check')
def health_check():
    """Simple health check task to verify Celery is running"""
    return {
        'status': 'ok',
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'worker': 'invigilo_celery'
    }


if __name__ == '__main__':
    print("Starting Celery worker...")
    print(f"Redis URL: {REDIS_URL}")
    print(f"ML Service URL: {ML_SERVICE_URL}")
    celery_app.start()

import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
import bcrypt
from bson import ObjectId
from dotenv import load_dotenv
import requests
import json
import datetime
import numpy as np
import base64
import cv2
import time
from PIL import Image
# DeepFace may not be available in all dev environments (native deps).
DEEPFACE_AVAILABLE = True
try:
    from deepface import DeepFace
except Exception as e:
    print(f"DeepFace not available: {e}")
    DeepFace = None
    DEEPFACE_AVAILABLE = False
import io

# --- Your existing imports for proctoring ---
try:
    from proctoring_module import (
        detectFace, isBlinking, mouthTrack, 
        gazeDetection, head_pose_detection, process_audio_chunk
    )
except Exception as e:
    print(f"proctoring_module import failed or not available: {e}")
    # Provide lightweight stubs so the server can run in dev without full native deps
    def detectFace(frame):
        # naive stub: assume 1 face detected with minimal face structure
        return 1, [{'bbox': [0,0,10,10]}]
    def isBlinking(faces, frame):
        return 'normal'
    def mouthTrack(faces, frame):
        return 'closed'
    def gazeDetection(faces, frame):
        return 'Center'
    def head_pose_detection(faces, frame):
        return 'Forward'
    def process_audio_chunk(audio_bytes):
        return 'clean'

# --- Configuration ---
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"

# --- Setup ---
load_dotenv()
app = Flask(__name__)
CORS(app)
APP_START = datetime.datetime.utcnow()

# --- Database ---
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MONGO_URI not found in .env file")

client = MongoClient(MONGO_URI)
db = client['invigilo_db']
users_collection = db['users']
exams_collection = db['exams']
proctor_events_collection = db['proctor_events']
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None

# --- ROUTES ---

# ✅ REGISTER USER + FACE ENROLLMENT
@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.get_json()
    required = ['fullName', 'email', 'phoneNumber', 'roleId', 'password', 'role', 'institution', 'department', 'imageDataUrl']
    if not all(field in data for field in required):
        return jsonify({"error": "Missing required fields"}), 400

    existing = {"$or": [{"email": data['email']}, {"phoneNumber": data['phoneNumber']}]}
    if data['role'] == 'student':
        existing["$or"].append({"studentId": data['roleId']})
    else:
        existing["$or"].append({"lecturerId": data['roleId']})
    if users_collection.find_one(existing):
        return jsonify({"error": "User already exists"}), 409

    # --- Decode and process face image ---
    image = decode_base64_image(data['imageDataUrl'])
    if image is None:
        return jsonify({"error": "Invalid image data"}), 400

    try:
        # Try to extract face embedding using DeepFace (ArcFace) when available.
        face_vector = None
        if DEEPFACE_AVAILABLE and DeepFace is not None:
            try:
                embeddings = DeepFace.represent(
                    img_path=image,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=True
                )
                if not embeddings or not isinstance(embeddings, list):
                    return jsonify({"error": "No face detected"}), 400
                face_vector = embeddings[0]['embedding']
            except Exception as e:
                print(f"DeepFace representation failed during register: {e}")
                # In dev environments we allow registration but mark faceVerified False

        hashed_pw = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())

        new_user = {
            "name": data['fullName'],
            "email": data['email'],
            "phoneNumber": data['phoneNumber'],
            "role": data['role'],
            "password": hashed_pw,
            "institution": data['institution'],
            "department": data['department'],
            "faceEmbedding": face_vector,
            "faceVerified": bool(face_vector),
            "isActive": True,
            "createdAt": datetime.datetime.utcnow()
        }
        if data['role'] == 'student':
            new_user['studentId'] = data['roleId']
            new_user['year'] = data.get('year')
        else:
            new_user['lecturerId'] = data['roleId']

        users_collection.insert_one(new_user)
        return jsonify({"message": "User registered successfully with face embedding!"}), 201

    except ValueError:
        return jsonify({"error": "No or multiple faces detected"}), 400
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ✅ LOGIN
@app.route('/api/login', methods=['POST'])
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
        user.pop('password', None)
        user.pop('faceEmbedding', None)
        return jsonify({"message": "Login successful", "user": user}), 200
    return jsonify({"error": "Invalid credentials"}), 401

# ✅ FACE VERIFICATION
@app.route('/api/verify-face', methods=['POST'])
def verify_face():
    data = request.get_json()
    identifier = data.get('identifier')
    role = data.get('role')
    image_data_url = data.get('imageDataUrl')

    if not identifier or not image_data_url or not role:
        return jsonify({"error": "Missing parameters"}), 400

    user = users_collection.find_one({
        "role": role,
        "$or": [{"email": identifier}, {"phoneNumber": identifier},
                {"studentId": identifier}, {"lecturerId": identifier}]
    })
    if not user or 'faceEmbedding' not in user:
        return jsonify({"error": "User not found or no face data"}), 404

    image = decode_base64_image(image_data_url)
    if image is None:
        return jsonify({"error": "Invalid image data"}), 400
    try:
        # If DeepFace is not available in this environment, short-circuit verification
        if not DEEPFACE_AVAILABLE or DeepFace is None:
            return jsonify({"message": "Face verification unavailable in this environment", "verified": False, "similarity": 0.0}), 200

        # helper: create simple variants to account for lighting/contrast differences
        def make_variants(img):
            variants = []
            try:
                base = cv2.resize(img, (160, 160))
            except Exception:
                base = img
            variants.append(base)
            try:
                # brightness up
                hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
                hsv = hsv.astype('float32')
                hsv[...,2] = np.clip(hsv[...,2] * 1.25, 0, 255)
                bright = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)
                variants.append(bright)
            except Exception:
                pass
            try:
                # CLAHE on L channel (LAB)
                lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl,a,b))
                clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                variants.append(clahe_img)
            except Exception:
                pass
            return variants

        variants = make_variants(image)

        # compute embeddings for variants, pick the best similarity
        stored_embedding = np.array(user['faceEmbedding'])
        similarities = []
        for v in variants:
            try:
                emb_obj = DeepFace.represent(img_path=v, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
                if not emb_obj or not isinstance(emb_obj, list):
                    continue
                new_emb = np.array(emb_obj[0].get('embedding'))
                denom = (np.linalg.norm(stored_embedding) * np.linalg.norm(new_emb))
                sim = float(np.dot(stored_embedding, new_emb) / denom) if denom > 0 else 0.0
                similarities.append(sim)
            except Exception as e:
                app.logger.debug('Variant embedding failed: %s', e)
                continue

        if not similarities:
            return jsonify({"message": "Face embedding extraction failed", "verified": False, "similarities": []}), 200

        max_sim = max(similarities)
        app.logger.info('Face verify for %s: similarities=%s, max=%s', identifier, similarities, max_sim)

        # Allow a slightly lower threshold when multiple variants are tried; still configurable
        THRESHOLD = float(os.getenv('FACE_SIMILARITY_THRESHOLD', '0.58'))
        if max_sim >= THRESHOLD:
            return jsonify({"message": "Face verified successfully", "verified": True, "similarity": float(max_sim), "similarities": similarities}), 200
        else:
            return jsonify({"message": "Face verification failed", "verified": False, "similarity": float(max_sim), "similarities": similarities}), 401

    except ValueError:
        return jsonify({"error": "No face detected"}), 400
    except Exception as e:
        app.logger.exception('Verification error')
        return jsonify({"error": "Internal verification error", "detail": str(e)}), 500

# --- (Keep all your other routes exactly as before) ---
# exams, proctoring, ai_generate_questions, etc. remain unchanged.



@app.route('/api/proctor', methods=['POST'])
def proctor_activity():
    data = request.get_json()
    image_data_url = data.get('imageDataUrl')
    user_id = data.get('userId')

    if not image_data_url or not user_id:
        return jsonify({"error": "Image data and User ID are required"}), 400
    
    frame = decode_base64_image(image_data_url)
    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400
        
    face_count, faces = detectFace(frame)
    
    if face_count == 0:
        return jsonify({"faceCount": 0, "error": "No face detected"}), 200
    if face_count > 1:
        return jsonify({"faceCount": face_count, "error": "Multiple faces detected"}), 200

    # --- Live Identity Verification (DeepFace ArcFace) ---
    identity_verified = False
    similarity_score = None
    try:
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if user and user.get('faceEmbedding') and DEEPFACE_AVAILABLE and DeepFace is not None:
            try:
                stored_embedding = np.array(user['faceEmbedding'])

                # Create small variants similar to verify_face to be tolerant to lighting/pose
                def _make_variants_local(img):
                    vs = []
                    try:
                        b = cv2.resize(img, (160, 160))
                    except Exception:
                        b = img
                    vs.append(b)
                    try:
                        hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV).astype('float32')
                        hsv[...,2] = np.clip(hsv[...,2] * 1.25, 0, 255)
                        bright = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)
                        vs.append(bright)
                    except Exception:
                        pass
                    try:
                        lab = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
                        l, a, ba = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        cl = clahe.apply(l)
                        limg = cv2.merge((cl,a,ba))
                        clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                        vs.append(clahe_img)
                    except Exception:
                        pass
                    return vs

                variants = _make_variants_local(frame)
                sims = []
                for v in variants:
                    try:
                        emb_o = DeepFace.represent(img_path=v, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
                        if not emb_o or not isinstance(emb_o, list):
                            continue
                        new_embedding = np.array(emb_o[0].get('embedding'))
                        denom = (np.linalg.norm(stored_embedding) * np.linalg.norm(new_embedding))
                        sim = float(np.dot(stored_embedding, new_embedding) / denom) if denom > 0 else 0.0
                        sims.append(sim)
                    except Exception as e:
                        app.logger.debug('Proctor variant embed failed: %s', e)
                        continue

                if sims:
                    similarity_score = float(max(sims))
                    app.logger.info('Proctor identity check for %s similarities=%s max=%s', user_id, sims, similarity_score)
                    THRESH = float(os.getenv('FACE_SIMILARITY_THRESHOLD', '0.58'))
                    if similarity_score >= THRESH:
                        identity_verified = True
            except Exception as e:
                print(f"DeepFace check failed during proctoring: {e}")
                identity_verified = False
        else:
            # No face embedding available or DeepFace not present; skip identity verification
            identity_verified = False
    except Exception as e:
        print(f"Error during live identity verification: {e}")
        
    # --- Proctoring Behavioral Checks (unchanged) ---
    results = {
        "faceCount": face_count,
        "identityVerified": identity_verified,
        "similarity": similarity_score,
        "blinkStatus": isBlinking(faces, frame),
        "gazeDirection": gazeDetection(faces, frame),
        "mouthStatus": mouthTrack(faces, frame),
        "headPose": head_pose_detection(faces, frame)
    }

    return jsonify(results), 200


@app.route('/api/proctor/audio', methods=['POST'])
def proctor_audio():
    data = request.get_json()
    audio_b64 = data.get('audioData')
    if not audio_b64:
        return jsonify({"error": "Audio data is required"}), 400
    
    audio_bytes = base64.b64decode(audio_b64)
    result = process_audio_chunk(audio_bytes)

    return jsonify({"audioStatus": result}), 200

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

@app.route('/api/exams/<exam_id>/submit', methods=['POST'])
def submit_exam(exam_id):
    data = request.get_json()
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
                            # fallback to string compare
                            if str(user_answer) == str(correct_answer):
                                correct = True
                    else:
                        # correct_answer might be option text; attempt to match
                        if isinstance(user_answer, (int, float)):
                            # if numeric, map to option text (1-based index)
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
                    # Coerce to boolean
                    ua_bool = None
                    if isinstance(user_answer, bool):
                        ua_bool = user_answer
                    else:
                        # Accept 'true'/'false' strings or '1'/'0'
                        s = str(user_answer).lower()
                        if s in ('true', '1', 'yes'): ua_bool = True
                        elif s in ('false', '0', 'no'): ua_bool = False
                    if ua_bool is not None and bool(correct_answer) == ua_bool:
                        correct = True

                else:
                    # Short answer / essay: perform trimmed case-insensitive match for small keywords
                    if correct_answer is not None and str(correct_answer).strip() != '':
                        if str(user_answer).strip().lower() == str(correct_answer).strip().lower():
                            correct = True
                    else:
                        # No ground truth: treat as not auto-gradable
                        correct = False

        except Exception as e:
            print(f"Error comparing answers for q {q_id}: {e}")

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
    try:
        exams_collection.update_one(
            {'_id': ObjectId(exam_id)},
            {
                '$push': {'attempts': attempt_record},
                '$addToSet': {'completedBy': user_id}
            }
        )
    except Exception as e:
        print(f"Failed to persist attempt for exam {exam_id}: {e}")

    return jsonify({
        "message": "Exam submitted successfully!",
        "score": percentage,
        "totalMarks": total_marks,
        "perQuestion": per_question
    }), 200

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
    """Return small set of stats for the lecturer dashboard (mock-friendly)."""
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
        'blink': 'low'
    }
    severity = severity_map.get(event_type, 'low')

    event = {
        'examId': str(data['examId']),
        'userId': str(data['userId']),
        'eventType': event_type,
        'details': data.get('details', {}),
        'severity': severity,
        'timestamp': datetime.datetime.utcnow()
    }
    try:
        proctor_events_collection.insert_one(event)
        return jsonify({'message': 'Event recorded'}), 201
    except Exception as e:
        print(f"Error recording proctor event: {e}")
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
        docs = list(proctor_events_collection.find({'examId': str(exam_id), 'userId': str(user_id)}).sort('timestamp', 1))
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
        print(f"Error getting proctoring details: {e}")
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
        docs = list(proctor_events_collection.find(q).sort('timestamp', 1).limit(limit))
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
        user.pop('password', None)
        return jsonify({'message': 'User updated', 'user': user}), 200
    except Exception as e:
        print(f"Error updating user {user_id}: {e}")
        return jsonify({'error': str(e)}), 500


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

    print(f"Starting Invigilo server on 0.0.0.0:{port} (DEV_MODE={DEV_MODE}, DEEPFACE_AVAILABLE={DEEPFACE_AVAILABLE})")

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

    # Run the app inside a restart loop so transient errors don't silently close the terminal.
    max_restarts = 3
    restarts = 0
    while True:
        try:
            # Avoid Flask reloader spawning extra processes which can close parent terminals on Windows.
            app.run(host='0.0.0.0', port=port, debug=DEV_MODE, use_reloader=False)
            break
        except Exception as e:
            app.logger.exception('Flask server crashed with exception: %s', e)
            restarts += 1
            if restarts >= max_restarts:
                print(f"Server crashed {restarts} times; giving up. See {log_file} for details.")
                break
            print(f"Server crashed, restarting ({restarts}/{max_restarts}) in 1s...")
            time.sleep(1)


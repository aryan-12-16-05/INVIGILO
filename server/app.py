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
from PIL import Image
from deepface import DeepFace
import io

# --- Your existing imports for proctoring ---
from proctoring_module import (
    detectFace, isBlinking, mouthTrack, 
    gazeDetection, head_pose_detection, process_audio_chunk
)

# --- Configuration ---
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"

# --- Setup ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- Database ---
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MONGO_URI not found in .env file")

client = MongoClient(MONGO_URI)
db = client['invigilo_db']
users_collection = db['users']
exams_collection = db['exams']
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
        # Extract face embedding using DeepFace (ArcFace)
        embeddings = DeepFace.represent(
            img_path=image,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
        if not embeddings or not isinstance(embeddings, list):
            return jsonify({"error": "No face detected"}), 400

        face_vector = embeddings[0]['embedding']
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
            "faceVerified": True,
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
        # Extract embedding for the live frame
        new_embedding = DeepFace.represent(
            img_path=image,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )[0]['embedding']

        stored_embedding = np.array(user['faceEmbedding'])
        new_embedding = np.array(new_embedding)

        # Compare cosine similarity
        similarity = np.dot(stored_embedding, new_embedding) / (
            np.linalg.norm(stored_embedding) * np.linalg.norm(new_embedding)
        )

        if similarity > 0.65:
            return jsonify({"message": "Face verified successfully", "verified": True, "similarity": float(similarity)}), 200
        else:
            return jsonify({"message": "Face verification failed", "verified": False, "similarity": float(similarity)}), 401

    except ValueError:
        return jsonify({"error": "No face detected"}), 400
    except Exception as e:
        print(f"Verification error: {e}")
        return jsonify({"error": "Internal verification error"}), 500

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
    try:
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if user and user.get('faceEmbedding'):
            stored_embedding = np.array(user['faceEmbedding'])

            # Generate new embedding for live frame
            new_embedding_obj = DeepFace.represent(
                img_path=frame,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )

            if new_embedding_obj and isinstance(new_embedding_obj, list):
                new_embedding = np.array(new_embedding_obj[0]['embedding'])

                # Cosine similarity check
                similarity = np.dot(stored_embedding, new_embedding) / (
                    np.linalg.norm(stored_embedding) * np.linalg.norm(new_embedding)
                )
                if similarity > 0.65:
                    identity_verified = True

    except Exception as e:
        print(f"Error during live identity verification: {e}")
        
    # --- Proctoring Behavioral Checks (unchanged) ---
    results = {
        "faceCount": face_count,
        "identityVerified": identity_verified,
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
    
    for question in exam.get('questions', []):
        q_id = str(question['_id'])
        total_marks += question.get('marks', 0)
        
        user_answer = answers.get(q_id)
        if user_answer is not None:
            if str(user_answer) == str(question['correctAnswer']):
                score += question.get('marks', 0)

    percentage = round((score / total_marks) * 100) if total_marks > 0 else 0

    attempt_record = {
        'userId': user_id,
        'score': percentage,
        'completedAt': datetime.datetime.utcnow().isoformat()
    }
    
    exams_collection.update_one(
        {'_id': ObjectId(exam_id)},
        {
            '$push': {'attempts': attempt_record}
        }
    )

    return jsonify({
        "message": "Exam submitted successfully!",
        "score": percentage,
        "totalMarks": total_marks
    }), 200

@app.route('/api/exams', methods=['GET'])
def get_exams():
    all_exams_cursor = exams_collection.find()
    all_exams = [serialize_doc(exam) for exam in all_exams_cursor]
    return jsonify({"exams": all_exams}), 200

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

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(url, json=payload)
        response.raise_for_status() 
        
        response_data = response.json()
        text_content = response_data['candidates'][0]['content']['parts'][0]['text']
        questions_json = json.loads(text_content)

        return jsonify(questions_json), 200

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"API request failed: {e}"}), 500
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return jsonify({"error": f"Failed to parse AI response: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


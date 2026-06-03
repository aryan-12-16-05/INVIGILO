# Invigilo-proctoring


# INVIGILO – AI-Powered Online Examination & Intelligent Proctoring System

## Overview

INVIGILO is a full-stack AI-powered online examination platform designed to conduct secure remote assessments while minimizing cheating through intelligent proctoring mechanisms.

The system combines Computer Vision, Face Recognition, Real-Time Monitoring, WebSockets, and AI-powered analytics to ensure examination integrity.

Students can securely take examinations online, while lecturers can monitor candidates in real time through an advanced proctoring dashboard.

---

# Key Features

## Examination Management

* Create and schedule examinations
* Define exam duration and time limits
* Support multiple question types
* Automatic evaluation and scoring
* Real-time submission tracking
* Exam analytics and reporting

## Secure Authentication

* Student and Lecturer role-based login
* Password encryption using bcrypt
* Face registration during account creation
* Face verification before examination access

## AI-Based Face Recognition

* Face enrollment during registration
* Face embedding generation using InsightFace
* Face verification before exam entry
* Identity mismatch detection
* Multiple face embedding support for higher accuracy

## Real-Time Proctoring

The system continuously monitors students throughout the examination.

### Face Monitoring

* No Face Detection
* Multiple Face Detection
* Identity Verification
* Face Mismatch Detection

### Eye Monitoring

* Eye Blink Detection
* Gaze Direction Tracking
* Looking Away Detection

### Head Pose Estimation

* Looking Left
* Looking Right
* Looking Up
* Looking Down
* Forward Position Detection

### Mouth Tracking

* Mouth Open Detection
* Talking Detection

### Browser Security Monitoring

* Tab Switching Detection
* Fullscreen Exit Detection
* Window Focus Loss Detection
* Developer Tools Detection
* Copy/Paste Attempt Detection
* Right Click Monitoring

### Environment Monitoring

* Camera Blocking Detection
* Background Change Detection
* External Assistance Detection

---

# System Architecture

# Detailed System Architecture

```text
                                    ┌──────────────────────┐
                                    │      Lecturer        │
                                    │ Monitoring Dashboard │
                                    └──────────┬───────────┘
                                               │
                                               │ WebSocket
                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Flask Backend                              │
│                                                                     │
│  Authentication Module                                              │
│  Exam Management Module                                             │
│  Face Verification Module                                           │
│  Proctoring Event Manager                                           │
│  Violation Detection Engine                                         │
│  Real-Time Notification System                                      │
└───────────────┬──────────────────────────────┬──────────────────────┘
                │                              │
                │                              │
                ▼                              ▼

      ┌─────────────────┐          ┌─────────────────────────┐
      │    MongoDB      │          │ Hugging Face ML Service │
      │                 │          │                         │
      │ Users           │          │ Face Recognition        │
      │ Exams           │          │ Face Embeddings         │
      │ Exam Attempts   │          │ Eye Tracking            │
      │ Violations      │          │ Head Pose Detection     │
      │ Logs            │          │ Mouth Detection         │
      └─────────────────┘          │ Gaze Estimation         │
                                   │ Multi-Face Detection    │
                                   └───────────┬─────────────┘
                                               │
                                               ▼
                                   ┌─────────────────────────┐
                                   │ InsightFace + OpenCV   │
                                   │ Dlib + ONNX Runtime    │
                                   └─────────────────────────┘


                ▲
                │ REST API
                │
                ▼

┌─────────────────────────────────────────────────────────────────────┐
│                    React + TypeScript Frontend                     │
│                                                                     │
│ Student Dashboard                                                   │
│ Lecturer Dashboard                                                  │
│ Registration Module                                                 │
│ Login Module                                                        │
│ Exam Interface                                                      │
│ Live Proctoring Client                                              │
│ Browser Activity Monitoring                                         │
│ Real-Time Alerts                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

# Registration Workflow

Student Registration

1. Student enters details.
2. Webcam captures face images.
3. Frontend sends images to Flask Backend.
4. Backend forwards images to ML Service.
5. ML Service generates facial embeddings using InsightFace.
6. Embeddings are stored in MongoDB.
7. User account is created.

```text
Student
   │
   ▼
Frontend
   │
   ▼
Flask Backend
   │
   ▼
ML Service
   │
   ▼
Face Embedding Generation
   │
   ▼
MongoDB Storage
```

---

# Login & Face Verification Workflow

```text
Student Login
      │
      ▼
Credential Validation
      │
      ▼
Face Capture
      │
      ▼
Generate New Embedding
      │
      ▼
Cosine Similarity Matching
      │
      ▼
Identity Verified
      │
      ▼
Allow Exam Access
```

---

# Examination Workflow

```text
Student Starts Exam
        │
        ▼
Exam Questions Loaded
        │
        ▼
Timer Activated
        │
        ▼
Continuous Camera Capture
        │
        ▼
Frame Sent To Backend
        │
        ▼
Frame Forwarded To ML Service
        │
        ▼
Violation Detection
        │
        ▼
Store Events In MongoDB
        │
        ▼
Send Alerts Through WebSocket
        │
        ▼
Lecturer Dashboard Updated
```

---

# Proctoring Pipeline

Every frame undergoes the following checks:

```text
Video Frame
     │
     ▼
Face Detection
     │
     ├── No Face
     ├── Multiple Faces
     └── Single Face
               │
               ▼
      Identity Verification
               │
               ▼
      Eye Tracking Module
               │
               ▼
      Gaze Direction Analysis
               │
               ▼
      Mouth Detection
               │
               ▼
      Head Pose Estimation
               │
               ▼
      Violation Classification
               │
               ▼
      Risk Scoring
               │
               ▼
      Database Logging
               │
               ▼
      Real-Time Alert
```

---

# Browser Security Architecture

```text
Student Browser
       │
       ├── Tab Switch Detection
       ├── Fullscreen Exit Detection
       ├── DevTools Detection
       ├── Copy/Paste Detection
       ├── Right Click Detection
       └── Window Blur Detection
                     │
                     ▼
             Flask Backend
                     │
                     ▼
             Violation Logs
                     │
                     ▼
          Lecturer Monitoring Panel
```

---

# Deployment Architecture

```text
Frontend
(Vercel)

      │
      ▼

Flask Backend
(Render)

      │
      ├──────────────► MongoDB Atlas
      │
      ▼

ML Service
(Hugging Face Spaces)

      │
      ▼

InsightFace + OpenCV + Dlib
```

This architecture ensures:

• Scalability through separation of ML and Backend services
• Real-time monitoring using WebSockets
• Secure authentication and face verification
• Continuous AI-powered proctoring during examinations
• Centralized event logging and analytics
• Cloud-native deployment architecture

```
```


# Working Flow

## Step 1: User Registration

1. User enters personal details.
2. Face images are captured.
3. Images are sent to the ML Service.
4. Face embeddings are generated.
5. Embeddings are stored securely in MongoDB.

## Step 2: Login

1. User enters credentials.
2. Password is validated.
3. User role is verified.
4. Dashboard is loaded.

## Step 3: Face Verification

Before entering an examination:

1. Camera image is captured.
2. Face embedding is generated.
3. Cosine similarity is calculated.
4. User identity is verified.

## Step 4: Examination Session

During examination:

* Webcam frames are captured periodically.
* Frames are sent to the backend.
* Backend forwards frames to ML Service.
* Violations are detected.
* Events are stored in MongoDB.
* Real-time alerts are sent to lecturers.

## Step 5: Live Monitoring

Lecturers can view:

* Active students
* Face status
* Violation count
* Risk score
* Exam progress
* Suspicious activity logs

---

# AI & Machine Learning Components

## Face Recognition

Technology:

* InsightFace
* ONNX Runtime

Purpose:

* Face Embedding Generation
* Identity Verification

## Computer Vision

Technology:

* OpenCV
* Dlib

Purpose:

* Face Detection
* Eye Tracking
* Gaze Estimation
* Head Pose Detection
* Mouth Tracking

## Similarity Matching

Method:

* Cosine Similarity

Purpose:

* Match live face with registered face.

---

# Security Features

## Authentication Security

* Password Hashing (bcrypt)
* Input Validation
* User Sanitization
* Role-Based Access Control

## API Security

* Rate Limiting
* Request Validation
* Error Handling
* CORS Protection

## Examination Security

* Face Verification
* Browser Lock Monitoring
* Fullscreen Enforcement
* Tab Change Detection
* Real-Time Violation Logging

---

# Technology Stack

## Frontend

* React
* TypeScript
* Vite
* CSS

## Backend

* Flask
* Flask SocketIO
* Flask Limiter
* Flask CORS

## Database

* MongoDB

## AI / ML

* InsightFace
* OpenCV
* Dlib
* NumPy
* ONNX Runtime

## Deployment

* Render (Backend)
* Hugging Face Spaces (ML Service)
* MongoDB Atlas (Database)

---

# Project Structure

frontend/
├── App.tsx
├── Sidebar.tsx
├── main.tsx
├── App.css
└── index.css

backend/
├── app.py
├── face_engine.py
├── proctoring_module.py

ml-service/
├── app.py
├── face_engine.py
└── proctoring_module.py

---

# Future Enhancements

* Object Detection (Mobile Phone Detection)
* Audio-Based Cheating Detection
* Speech Recognition Monitoring
* Liveness Detection
* AI-Based Risk Scoring
* Examination Analytics Dashboard
* Automated Cheating Report Generation

---



# Conclusion

INVIGILO provides a scalable, secure, and AI-driven online examination ecosystem that combines face recognition, computer vision, real-time monitoring, and intelligent analytics to maintain examination integrity in remote environments.

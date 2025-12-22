# Live Proctoring & Reporting Implementation Guide

## Overview
This document outlines the complete implementation of:
1. Live Proctoring Monitoring Screen
2. Professional Exam Report Screen

## Phase 1: Frontend Structure (App.tsx)

### 1. Add New App States
```typescript
type AppState = '... | 'lecturer-proctor' | 'lecturer-report';
```

### 2. Add State Variables in App()
```typescript
const [selectedExamId, setSelectedExamId] = useState<string | null>(null);
```

### 3. Update LecturerDashboard Buttons
In the exam card buttons section, update:
```typescript
<Button onClick={() => { setSelectedExamId(exam._id); navigateTo('lecturer-proctor'); }}>
  Proctor
</Button>
<Button onClick={() => { setSelectedExamId(exam._id); navigateTo('lecturer-report'); }}>
  View Report
</Button>
```

### 4. Add Routes in renderContent()
```typescript
case 'lecturer-proctor': 
  return currentUser && selectedExamId && <LecturerProcto

Monitor />;
case 'lecturer-report': 
  return currentUser && selectedExamId && <LecturerExamReport />;
```

## Phase 2: Backend API Endpoints (server/app.py)

### API 1: Get Exam Roster
```python
@app.route('/api/lecturer/exams/<exam_id>/roster', methods=['GET'])
def get_exam_roster(exam_id):
    # Verify lecturer owns exam
    # Return students matching institution/department/year
    pass
```

### API 2: Monitor Students
```python
@app.route('/api/lecturer/exams/<exam_id>/monitor', methods=['POST'])
def monitor_exam(exam_id):
    # Return real-time student status + incidents
    # Data from proctoring_logs collection
    pass
```

### API 3: Resolve Incident
```python
@app.route('/api/lecturer/exams/<exam_id>/incident/resolve', methods=['POST'])
def resolve_incident(exam_id):
    # Update incident with lecturer decision
    pass
```

### API 4: Disqualify Student
```python
@app.route('/api/lecturer/exams/<exam_id>/student/<student_id>/disqualify', methods=['POST'])
def disqualify_student(exam_id, student_id):
    # Mark student attempt as disqualified
    pass
```

### API 5: Get Exam Report
```python
@app.route('/api/lecturer/exams/<exam_id>/report', methods=['GET'])
def get_exam_report(exam_id):
    # Compute comprehensive report with analytics
    # Return: roster, attempts, scores, incidents, risk scores
    pass
```

## Phase 3: MongoDB Schema

### Collections
```
proctoring_logs: {
  _id, examId, userId, timestamp, eventType, severity,
  details: { faceCount, gazeDirection, headPose, audioStatus, ... },
  frameEvidence: dataUrl,
  resolved: boolean,
  lecturerNotes: string,
  actionTaken: 'confirmed' | 'dismissed' | 'pending'
}

exam_attempts: {
  _id, examId, userId, status, score, startedAt, completedAt,
  disqualified: boolean, disqualificationReason: string,
  riskScore: number
}
```

## Phase 4: React Components

### LecturerPro ctorMonitor Component
Key features:
- Header with exam info
- Polling mechanism (every 3 seconds)
- Student grid with status indicators
- Real-time alerts panel
- Incident modal
- Student detail modal

### LecturerExamReport Component
Key features:
- Professional header
- KPI summary cards
- Performance table
- Violation analytics
- Export functionality
- Print-friendly layout

## Risk Score Calculation
```
Base = 0
+ Identity mismatch: +30
+ Multiple faces: +25
+ Phone detected: +20
+ Tab switch: +15
+ Gaze away (repeated): +10
+ Audio suspicious: +10
Max: 100

Levels:
- 0-19: Low (green)
- 20-49: Medium (yellow)
- 50-74: High (orange)
- 75+: Critical (red)
```

## Implementation Order
1. ✅ Add app states
2. ⏳ Create basic component shells
3. ⏳ Implement backend APIs
4. ⏳ Add polling & real-time updates
5. ⏳ Style and polish UI
6. ⏳ Test end-to-end

## Next Steps
1. Update App.tsx with state and routing
2. Create backend endpoints in server/app.py
3. Implement components with mock data
4. Connect to real APIs
5. Add polling and WebSocket support

import dlib
import cv2
import numpy as np
from math import hypot, degrees, atan
from imutils import face_utils

# ================================================================================
# DETECTION THRESHOLDS CONFIGURATION
# ================================================================================
# Centralized threshold values for all proctoring detection modules.
# Adjust these values to calibrate detection sensitivity based on testing.

# === EYE BLINK DETECTION THRESHOLDS ===
# Eye Aspect Ratio (EAR) - ratio of horizontal to vertical eye distance
# Higher value = more closed eye required to register as blink
# Typical range: 3.0 - 5.0
# Setting to 4.2 for reduced false positives - natural blinking won't trigger
BLINK_THRESHOLD = 4.2

# === GAZE DETECTION THRESHOLDS ===
# Ratio of white pixels on one side vs other side of eye
# Higher value = more extreme eye movement required (stricter)
# Lower value = detects even small eye movements (more sensitive)
# Typical range: 1.0 - 2.5
# Setting to 2.0 for reduced false positives - requires significant gaze deviation
GAZE_RATIO_THRESHOLD = 2.0

# Threshold value for binary eye segmentation
# Lower value = more sensitive to darker pixels (pupil/iris)
# Higher value = requires stronger contrast
# Typical range: 40 - 70
# Setting to 42 for better pupil detection with reduced false positives
GAZE_THRESHOLD_VALUE = 42

# === MOUTH DETECTION THRESHOLDS ===
# Distance in pixels between outer top and bottom lip
# Higher value = mouth must open wider to register
# Typical range: 15 - 40 pixels (depends on camera distance)
# Setting to 35 to avoid false positives from breathing, yawning (requires clear talking)
MOUTH_OPEN_THRESHOLD = 35

# === HEAD POSE DETECTION THRESHOLDS ===
# Angle in degrees for vertical head movement (up/down)
# Higher value = more head tilt required (less strict)
# Lower value = detects smaller head movements (more strict)
# Typical range: 20 - 50 degrees
# Setting to 45 for natural reading angles and posture shifts - more tolerance
HEAD_VERTICAL_ANGLE_THRESHOLD = 45

# Horizontal offset in pixels for lateral head movement (left/right)
# Distance nose must be from eye center line
# Higher value = more head turn required (less strict)
# Lower value = detects smaller head turns (more strict)
# Typical range: 5 - 20 pixels (depends on camera distance)
# Setting to 18 for natural movement tolerance - reduced false positives
HEAD_HORIZONTAL_OFFSET_THRESHOLD = 18

# === AUDIO DETECTION THRESHOLDS ===
# Amplitude range for human voice detection
# Minimum amplitude to distinguish from silence
# Typical range: 500 - 2000 (16-bit audio)
AUDIO_AMPLITUDE_MIN = 1000

# Maximum amplitude to distinguish from loud noise/music
# Typical range: 15000 - 25000 (16-bit audio)
AUDIO_AMPLITUDE_MAX = 20000

# Zero-Crossing Rate (ZCR) range for voice detection
# Human voice typically has ZCR between 0.1 - 0.3
# Background noise/music has different patterns

# Minimum ZCR for voice
# Typical range: 0.03 - 0.1
AUDIO_ZCR_MIN = 0.05

# Maximum ZCR for voice
# Typical range: 0.25 - 0.4
AUDIO_ZCR_MAX = 0.35

# Minimum standard deviation of amplitude to detect variation
# Helps distinguish speech from constant background noise
# Typical range: 300 - 800
AUDIO_AMPLITUDE_VARIATION_MIN = 500

# ================================================================================
# END CONFIGURATION
# ================================================================================

# --- INITIALIZE MODELS AND PREDICTORS ---
print("Loading Dlib Shape Predictor...")
shapePredictorModel = 'shape_predictor_model/shape_predictor_68_face_landmarks.dat'
shapePredictor = dlib.shape_predictor(shapePredictorModel)
faceDetector = dlib.get_frontal_face_detector()
print("Dlib Shape Predictor loaded.")

# --- FACIAL DETECTION ---
def detectFace(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector(gray,0)
    return len(faces), faces

# --- BLINK DETECTION ---
def midPoint(pointA, pointB):
    X = int((pointA.x + pointB.x) / 2)
    Y = int((pointA.y + pointB.y) / 2)
    return (X,Y)

def findDist(pointA, pointB):
    dist = hypot((pointA[0]-pointB[0]), (pointA[1]-pointB[1]))
    return dist

def isBlinking(faces, frame):
    left = [36,37,38,39,40,41]
    right = [42,43,44,45,46,47]
    for face in faces:
        facialLandmarks = shapePredictor(frame, face)
        lLeftPoint = (facialLandmarks.part(36).x, facialLandmarks.part(36).y)
        lRightPoint = (facialLandmarks.part(39).x, facialLandmarks.part(39).y)
        lTopPoint = midPoint(facialLandmarks.part(37), facialLandmarks.part(38))
        lBottomPoint = midPoint(facialLandmarks.part(40), facialLandmarks.part(41))
        leftHorLen = findDist(lLeftPoint, lRightPoint)
        leftVerLen = findDist(lTopPoint, lBottomPoint)
        lRatio = leftHorLen/leftVerLen if leftVerLen > 0 else 0
        
        rLeftPoint = (facialLandmarks.part(42).x, facialLandmarks.part(42).y)
        rRightPoint = (facialLandmarks.part(45).x, facialLandmarks.part(45).y)
        rTopPoint = midPoint(facialLandmarks.part(43), facialLandmarks.part(44))
        rBottomPoint = midPoint(facialLandmarks.part(46), facialLandmarks.part(47))
        rightHorLen = findDist(rLeftPoint, rRightPoint)
        rightVerLen = findDist(rTopPoint, rBottomPoint)
        rRatio = rightHorLen/rightVerLen if rightVerLen > 0 else 0

        if (lRatio >= BLINK_THRESHOLD or rRatio >= BLINK_THRESHOLD):
            return "Blink"
        else:
            return "No Blink"
    return "N/A"

# --- EYE GAZE DETECTION ---
def createMask(frame):
    height, width, _ = frame.shape
    return np.zeros((height, width), np.uint8)

def extractEye(mask, region, frame):
    cv2.polylines(mask, [region], True, 255, 2)
    cv2.fillPoly(mask, [region], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)

def eyeSegmentationAndReturnWhite(img, side):
    height, width = img.shape
    if (side == 'left'):
        return cv2.countNonZero(img[0:height, 0:int(width/2)])
    else:
        return cv2.countNonZero(img[0:height, int(width/2):width])

def gazeDetection(faces, frame):
    leftEye = [36,37,38,39,40,41]
    rightEye = [42,43,44,45,46,47]
    for face in faces:
        facialLandmarks = shapePredictor(frame, face)
        leftEyeRegion = np.array([(facialLandmarks.part(i).x, facialLandmarks.part(i).y) for i in leftEye], np.int32)
        rightEyeRegion = np.array([(facialLandmarks.part(i).x, facialLandmarks.part(i).y) for i in rightEye], np.int32)
        
        mask = createMask(frame)
        left_eye_mask = mask.copy()
        right_eye_mask = mask.copy()

        left_eye_extracted = extractEye(left_eye_mask, leftEyeRegion, frame)
        right_eye_extracted = extractEye(right_eye_mask, rightEyeRegion, frame)
        
        lmin_x, lmax_x = np.min(leftEyeRegion[:,0]), np.max(leftEyeRegion[:,0])
        lmin_y, lmax_y = np.min(leftEyeRegion[:,1]), np.max(leftEyeRegion[:,1])
        rmin_x, rmax_x = np.min(rightEyeRegion[:,0]), np.max(rightEyeRegion[:,0])
        rmin_y, rmax_y = np.min(rightEyeRegion[:,1]), np.max(rightEyeRegion[:,1])

        left_eye_frame = left_eye_extracted[lmin_y:lmax_y, lmin_x:lmax_x]
        right_eye_frame = right_eye_extracted[rmin_y:rmax_y, rmin_x:rmax_x]

        leftGrayEye = cv2.cvtColor(left_eye_frame, cv2.COLOR_BGR2GRAY)
        rightGrayEye = cv2.cvtColor(right_eye_frame, cv2.COLOR_BGR2GRAY)

        _, leftTh = cv2.threshold(leftGrayEye, GAZE_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        _, rightTh = cv2.threshold(rightGrayEye, GAZE_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        
        leftSideOfLeftEye, rightSideOfLeftEye = eyeSegmentationAndReturnWhite(leftTh, 'right'), eyeSegmentationAndReturnWhite(leftTh, 'left')
        leftSideOfRightEye, rightSideOfRightEye = eyeSegmentationAndReturnWhite(rightTh, 'right'), eyeSegmentationAndReturnWhite(rightTh, 'left')
        
        if (rightSideOfRightEye >= GAZE_RATIO_THRESHOLD * leftSideOfRightEye): return 'Left'
        elif (leftSideOfLeftEye >= GAZE_RATIO_THRESHOLD * rightSideOfLeftEye): return 'Right'
        else: return 'Center'
    return "N/A"

# --- MOUTH TRACKING ---
def mouthTrack(faces, frame):
    for face in faces:
        facialLandmarks = shapePredictor(frame, face)
        outerTop = (facialLandmarks.part(51).x, facialLandmarks.part(51).y)
        outerBottom = (facialLandmarks.part(57).x, facialLandmarks.part(57).y)
        dist = hypot(outerTop[0] - outerBottom[0], outerTop[1] - outerBottom[1])
        if (dist > MOUTH_OPEN_THRESHOLD):
            return "Mouth Open"
        else:
            return "Mouth Closed"
    return "N/A"

# --- HEAD POSE ESTIMATION ---
model_points = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
])

def head_pose_detection(faces, img):
    size = img.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    
    for face in faces:
        marks = shapePredictor(img, face)
        image_points = np.array([
            (marks.part(30).x, marks.part(30).y), (marks.part(8).x, marks.part(8).y),
            (marks.part(36).x, marks.part(36).y), (marks.part(45).x, marks.part(45).y),
            (marks.part(48).x, marks.part(48).y), (marks.part(54).x, marks.part(54).y)
        ], dtype="double")
        
        dist_coeffs = np.zeros((4,1))
        (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
        
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        # Safe angle calculation with division by zero protection
        try:
            dx = p2[0] - p1[0]  # Horizontal difference
            dy = p2[1] - p1[1]  # Vertical difference
            
            # Handle edge cases
            if dx == 0 and dy == 0:
                # No movement detected - assume forward position
                ang1 = 0
            elif dx == 0:
                # Vertical line - head is straight up or down
                ang1 = 90 if dy > 0 else -90
            else:
                # Safe division - calculate angle
                ang1 = int(degrees(atan(dy / dx)))
                # Clamp angle between -90 and 90 degrees
                ang1 = max(-90, min(90, ang1))
        except Exception as e:
            # Fallback to forward position if calculation fails
            import logging
            logging.debug(f"Head pose angle calculation failed: {e}")
            ang1 = 0
            
        if ang1 >= HEAD_VERTICAL_ANGLE_THRESHOLD: return "Head Down"
        elif ang1 <= -HEAD_VERTICAL_ANGLE_THRESHOLD: return "Head Up"

        nose_x, left_eye_x, right_eye_x = marks.part(30).x, marks.part(36).x, marks.part(45).x

        if nose_x < left_eye_x - HEAD_HORIZONTAL_OFFSET_THRESHOLD: return "Head Left"
        elif nose_x > right_eye_x + HEAD_HORIZONTAL_OFFSET_THRESHOLD: return "Head Right"
        
        return "Forward"
    return "N/A"

# --- AUDIO DETECTION ---
def process_audio_chunk(audio_bytes):
    """
    Detect human voice in audio chunk using amplitude and frequency analysis.
    Returns 'Voice detected' only when human speech patterns are present.
    """
    try:
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Basic amplitude check - skip silent audio
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude < AUDIO_AMPLITUDE_MIN:  # Too quiet to be speech
            return "Normal audio level"
        
        # Voice detection using zero-crossing rate (ZCR)
        # Human voice has ZCR typically between 0.1-0.3
        # Background noise/music has different ZCR patterns
        zero_crossings = np.where(np.diff(np.sign(audio_data)))[0]
        zcr = len(zero_crossings) / len(audio_data)
        
        # Voice characteristics:
        # - Moderate amplitude (typical for speech)
        # - ZCR in voice range
        # - Some variation in amplitude (not constant noise)
        
        amplitude_variation = np.std(audio_data)
        is_voice = (
            AUDIO_AMPLITUDE_MIN < max_amplitude < AUDIO_AMPLITUDE_MAX and  # Voice amplitude range
            AUDIO_ZCR_MIN < zcr < AUDIO_ZCR_MAX and              # Voice ZCR range
            amplitude_variation > AUDIO_AMPLITUDE_VARIATION_MIN  # Has variation (not static noise)
        )
        
        if is_voice:
            return "Voice detected"
        else:
            return "Normal audio level"
            
    except Exception as e:
        print(f"Error processing audio chunk: {e}")
        return "Audio error"

# --- Compatibility Aliases ---
# Provide names expected by callers (e.g., ML service app)
def headPoseEstimation(faces, img):
    return head_pose_detection(faces, img)

def gazeEstimation(faces, frame):
    return gazeDetection(faces, frame)



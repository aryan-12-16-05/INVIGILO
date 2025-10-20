import dlib
import cv2
import numpy as np
from math import hypot, degrees, atan
from imutils import face_utils

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

        if (lRatio >= 3.6 or rRatio >= 3.6):
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
    TrialRation = 1.2
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

        _, leftTh = cv2.threshold(leftGrayEye, 50, 255, cv2.THRESH_BINARY)
        _, rightTh = cv2.threshold(rightGrayEye, 50, 255, cv2.THRESH_BINARY)
        
        leftSideOfLeftEye, rightSideOfLeftEye = eyeSegmentationAndReturnWhite(leftTh, 'right'), eyeSegmentationAndReturnWhite(leftTh, 'left')
        leftSideOfRightEye, rightSideOfRightEye = eyeSegmentationAndReturnWhite(rightTh, 'right'), eyeSegmentationAndReturnWhite(rightTh, 'left')
        
        if (rightSideOfRightEye >= TrialRation * leftSideOfRightEye): return 'Left'
        elif (leftSideOfLeftEye >= TrialRation * rightSideOfLeftEye): return 'Right'
        else: return 'Center'
    return "N/A"

# --- MOUTH TRACKING ---
def mouthTrack(faces, frame):
    for face in faces:
        facialLandmarks = shapePredictor(frame, face)
        outerTop = (facialLandmarks.part(51).x, facialLandmarks.part(51).y)
        outerBottom = (facialLandmarks.part(57).x, facialLandmarks.part(57).y)
        dist = hypot(outerTop[0] - outerBottom[0], outerTop[1] - outerBottom[1])
        if (dist > 23):
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

        if p2[0] == p1[0]: ang1 = 90
        else: ang1 = int(degrees(atan((p2[1] - p1[1])/(p2[0] - p1[0]))))
            
        if ang1 >= 30: return "Head Down"
        elif ang1 <= -30: return "Head Up"

        nose_x, left_eye_x, right_eye_x = marks.part(30).x, marks.part(36).x, marks.part(45).x

        if nose_x < left_eye_x - 10: return "Head Left"
        elif nose_x > right_eye_x + 10: return "Head Right"
        
        return "Forward"
    return "N/A"

# --- AUDIO DETECTION ---
def process_audio_chunk(audio_bytes):
    try:
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        if np.max(np.abs(audio_data)) > 2000: # Threshold
            return "Suspicious audio detected"
        else:
            return "Normal audio level"
    except Exception as e:
        print(f"Error processing audio chunk: {e}")
        return "Audio error"


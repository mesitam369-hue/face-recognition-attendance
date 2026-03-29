import cv2
import face_recognition
import numpy as np
import base64

# -------------------------------
# 📥 Decode Image from Frontend
# -------------------------------
def decode_base64_image(image_data):
    try:
        image_data = image_data.split(',')[1]
        decoded = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("Decode error:", e)
        return None


# -------------------------------
# 🎯 Face Encoding
# -------------------------------
def get_face_encoding(image_bgr):
    if image_bgr is None:
        return None

    height, width = image_bgr.shape[:2]
    target_width = 640

    if width > target_width:
        scale = target_width / width
        image = cv2.resize(image_bgr, (0, 0), fx=scale, fy=scale)
    else:
        image = image_bgr

    rgb_image = image[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_image)

    if not face_locations:
        return None

    encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return encodings[0] if encodings else None


# -------------------------------
# 🔍 Face Matching
# -------------------------------
def match_face(unknown_encoding, known_encodings, tolerance=0.5):
    if not known_encodings or unknown_encoding is None:
        return -1

    proc_known = [np.array(e) for e in known_encodings]

    matches = face_recognition.compare_faces(proc_known, unknown_encoding, tolerance=tolerance)

    if True in matches:
        distances = face_recognition.face_distance(proc_known, unknown_encoding)
        best_match = np.argmin(distances)
        if matches[best_match]:
            return best_match

    return -1


# -------------------------------
# 👁️ Eye Aspect Ratio (Liveness)
# -------------------------------
def calculate_ear(eye_points):
    if len(eye_points) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = [np.array(pt) for pt in eye_points]

    d_v1 = np.linalg.norm(p2 - p6)
    d_v2 = np.linalg.norm(p3 - p5)
    d_h = np.linalg.norm(p1 - p4)

    return (d_v1 + d_v2) / (2.0 * d_h) if d_h != 0 else 0.0


# -------------------------------
# 🧭 Face Orientation
# -------------------------------
def get_face_orientation(landmarks):
    try:
        nose_tip = np.array(landmarks['nose_tip'][2])
        left_eye = np.array(landmarks['left_eye']).mean(axis=0)
        right_eye = np.array(landmarks['right_eye']).mean(axis=0)

        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist == 0:
            return 0, 0

        eyes_center = (left_eye + right_eye) / 2

        yaw = (nose_tip[0] - eyes_center[0]) / eye_dist
        pitch = (nose_tip[1] - eyes_center[1]) / eye_dist - 0.4

        return yaw, pitch
    except:
        return 0, 0


# -------------------------------
# 🧠 Liveness Detection
# -------------------------------
def get_face_liveness_metrics(image_bgr):
    if image_bgr is None:
        return None

    small = cv2.resize(image_bgr, (0, 0), fx=0.5, fy=0.5)
    rgb = small[:, :, ::-1]

    landmarks_list = face_recognition.face_landmarks(rgb)

    if not landmarks_list:
        return None

    landmarks = landmarks_list[0]

    if 'left_eye' not in landmarks:
        return None

    left_ear = calculate_ear(landmarks['left_eye'])
    right_ear = calculate_ear(landmarks['right_eye'])
    avg_ear = (left_ear + right_ear) / 2

    yaw, pitch = get_face_orientation(landmarks)

    return {
        "ear": avg_ear,
        "yaw": yaw,
        "pitch": pitch
    }


# -------------------------------
# 🟩 Draw Face Box
# -------------------------------
def draw_face_box(image, name=""):
    if image is None:
        return ""

    img = image.copy()

    small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    rgb = small[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb)

    for (top, right, bottom, left) in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        color = (0, 255, 0)

        if name in ["Unknown", "Scanning..."]:
            color = (0, 165, 255)
        elif "Step" in name:
            color = (255, 255, 0)

        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.rectangle(img, (left, bottom - 25), (right, bottom), color, cv2.FILLED)

        cv2.putText(img, name, (left + 6, bottom - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    _, jpeg = cv2.imencode('.jpg', img)
    return base64.b64encode(jpeg.tobytes()).decode('utf-8')

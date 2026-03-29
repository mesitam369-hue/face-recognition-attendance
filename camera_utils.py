import cv2
import face_recognition
import numpy as np
import threading
import time
from datetime import datetime
import base64

class VideoCamera:
    def __init__(self):
        try:
        self.video = cv2.VideoCapture(0)
        except:
        self.last_face_locations = []
        self.last_frame = None
        self.success = False
        self.stopped = False
        
        # Start background thread for frame reading
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
    
    def _update(self):
        """Internal thread to continuously read frames."""
        while not self.stopped:
            if not self.video.isOpened():
                time.sleep(0.1)
                continue
                
            success, frame = self.video.read()
            if success:
                self.success = True
                self.last_frame = frame
            else:
                self.success = False
            
            # Tiny sleep to prevent 100% CPU usage on this thread
            time.sleep(0.01)

    def __del__(self):
        self.release()
        
    def release(self):
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        if hasattr(self, 'video') and self.video is not None:
            if self.video.isOpened():
                self.video.release()
            self.video = None
        
        self.success = False
        self.last_frame = None
        
    def get_frame(self, draw_box=False, name="Unknown"):
        """Old method for backward compatibility, uses threaded frame."""
        if not self.success or self.last_frame is None:
            return None, None
        
        image = self.last_frame.copy()
        return self.get_frame_with_box(image, name if draw_box else "")

    def get_frame_with_box(self, image, name="", update_locations=True):
        """
        Encodes the image to JPEG, optionally drawing a box.
        If update_locations is True, it recalculates face positions.
        Otherwise, it uses the cached locations.
        """
        ret_image = image.copy()
        
        if update_locations:
            # Resize frame once for processing face locations
            small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
            # Convert BGR to RGB (performance enhancement for face_recognition)
            rgb_small_frame = small_frame[:, :, ::-1]
            self.last_face_locations = face_recognition.face_locations(rgb_small_frame)

        if name and name != "":
            for (top, right, bottom, left) in self.last_face_locations:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                box_color = (0, 255, 0) # Green for match
                if name == "Unknown" or name == "Scanning...":
                    box_color = (0, 165, 255) # Orange for unknown/scanning
                
                cv2.rectangle(ret_image, (left, top), (right, bottom), box_color, 2)
                # Smaller label box for smaller text
                cv2.rectangle(ret_image, (left, bottom - 25), (right, bottom), box_color, cv2.FILLED)
                cv2.putText(ret_image, name, (left + 6, bottom - 8), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        ret, jpeg = cv2.imencode('.jpg', ret_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes(), image

def get_face_encoding(image_bgr):
    """
    Returns the first face encoding found in the BGR image, else None.
    Optimized: Resizes for detection speed and ensures RGB conversion.
    """
    if image_bgr is None:
        return None
        
    # Standardize detection size to 480p width for speed without sacrificing too much accuracy
    height, width = image_bgr.shape[:2]
    target_width = 640
    if width > target_width:
        scale = target_width / width
        image_to_process = cv2.resize(image_bgr, (0, 0), fx=scale, fy=scale)
    else:
        scale = 1.0
        image_to_process = image_bgr
        
    # Convert BGR to RGB
    rgb_image = image_to_process[:, :, ::-1]
    
    # Locate faces in the resized image
    face_locations = face_recognition.face_locations(rgb_image)
    if not face_locations:
        return None
    
    # Get encoding for the first face found
    encodings = face_recognition.face_encodings(rgb_image, face_locations)
    if encodings:
        return encodings[0]
    return None

def match_face(unknown_encoding, known_encodings, tolerance=0.5):
    """
    Given a list of known_encodings, checks if unknown matches any.
    Returns the index of the first match, or -1 if no match.
    Tolerance 0.5 is stricter than default (0.6). Lower = stricter.
    """
    if not known_encodings or unknown_encoding is None:
        return -1
        
    # Ensure all known encodings are numpy arrays
    proc_known = [np.array(e) if not isinstance(e, np.ndarray) else e for e in known_encodings]
    
    # Compare faces returns a list of True/False
    # Note: face_recognition.compare_faces uses 0.6 as library default
    matches = face_recognition.compare_faces(proc_known, unknown_encoding, tolerance=tolerance)
    
    if True in matches:
        # If there are matches, find the one with the smallest distance (best match)
        face_distances = face_recognition.face_distance(proc_known, unknown_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return best_match_index
            
    return -1

def calculate_ear(eye_points):
    """
    Calculates the Eye Aspect Ratio (EAR) given 6 landmark points.
    Formula: EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    if len(eye_points) < 6:
        return 0.0
        
    # Vertical distances
    # Point indices for EAR: 0-left, 3-right, 1,2-top, 4,5-bottom
    # But face_recognition returns points in order: p1, p2, p3, p4, p5, p6
    # p1(0), p2(1), p3(2), p4(3), p5(4), p6(5)
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    p1, p2, p3, p4, p5, p6 = [np.array(pt) for pt in eye_points]
    
    d_v1 = np.linalg.norm(p2 - p6)
    d_v2 = np.linalg.norm(p3 - p5)
    d_h = np.linalg.norm(p1 - p4)
    
    # Avoid division by zero
    if d_h == 0:
        return 0.0
        
    ear = (d_v1 + d_v2) / (2.0 * d_h)
    return ear

def get_face_orientation(landmarks):
    """
    Estimates horizontal (Yaw) and vertical (Pitch) head orientation.
    Yaw: Negative = Left, Positive = Right
    Pitch: Positive = Down, Negative = Up
    """
    try:
        # Key landmarks
        nose_tip = np.array(landmarks['nose_tip'][2]) # Tip of the nose
        left_eye = np.array(landmarks['left_eye']).mean(axis=0) # Center of left eye
        right_eye = np.array(landmarks['right_eye']).mean(axis=0) # Center of right eye
        
        # Distance between eyes for normalization
        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist == 0: return 0, 0
        
        # Yaw: Relative horizontal position of nose tip between eyes
        eyes_center = (left_eye + right_eye) / 2
        yaw = (nose_tip[0] - eyes_center[0]) / eye_dist
        
        # Pitch: Relative vertical position of nose tip compared to eyes
        # Higher positive value = looking down, negative = looking up
        # We offset by a baseline (approx 0.3-0.5) because nose is naturally below eyes
        pitch = (nose_tip[1] - eyes_center[1]) / eye_dist - 0.4
        
        return yaw, pitch
    except (KeyError, IndexError):
        return 0, 0

def get_face_liveness_metrics(image_bgr):
    """
    Returns liveness metrics (EAR and Orientation) for the first face found.
    Returns: { 'ear': float, 'yaw': float, 'pitch': float } or None
    Optimized: Resizes image for faster landmark detection.
    """
    if image_bgr is None:
        return None
        
    # Resize for faster landmark detection (0.5x scale = 4x faster)
    small_frame = cv2.resize(image_bgr, (0, 0), fx=0.5, fy=0.5)
    rgb_small = small_frame[:, :, ::-1]
    
    # Landmark detection
    face_landmarks_list = face_recognition.face_landmarks(rgb_small)
    
    if not face_landmarks_list:
        return None
        
    landmarks = face_landmarks_list[0]
    if 'left_eye' not in landmarks or 'right_eye' not in landmarks or 'nose_tip' not in landmarks:
        return None
        
    # EAR and Orientation calculations are scale-invariant (they use ratios/norms)
    left_ear = calculate_ear(landmarks['left_eye'])
    right_ear = calculate_ear(landmarks['right_eye'])
    avg_ear = (left_ear + right_ear) / 2.0
    
    yaw, pitch = get_face_orientation(landmarks)
    
    return {
        'ear': avg_ear,
        'yaw': yaw,
        'pitch': pitch
    }

def draw_face_box(image, name=""):
    """
    Draws a box around detected faces and labels them with the given name.
    Returns the base64 encoded JPEG image.
    """
    if image is None:
        return ""
    ret_image = image.copy()
    
    # Resize frame once for processing face locations
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    # Convert BGR to RGB (performance enhancement for face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)

    for (top, right, bottom, left) in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        box_color = (0, 255, 0) # Green for match
        if name == "Unknown" or name == "Scanning...":
            box_color = (0, 165, 255) # Orange for unknown/scanning
        elif "Step" in name:
             box_color = (255, 255, 0) # Yellow for liveness prompt
        
        cv2.rectangle(ret_image, (left, top), (right, bottom), box_color, 2)
        cv2.rectangle(ret_image, (left, bottom - 25), (right, bottom), box_color, cv2.FILLED)
        cv2.putText(ret_image, name, (left + 6, bottom - 8), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    ret, jpeg = cv2.imencode('.jpg', ret_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(jpeg.tobytes()).decode('utf-8')

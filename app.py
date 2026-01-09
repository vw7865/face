"""
LooksMax AI Backend API
Face analysis using MediaPipe Face Mesh
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
import os

# DeepFace for gender detection (97%+ accuracy)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("DeepFace imported successfully for gender detection")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("WARNING: DeepFace not available. Gender detection will fall back to user input.")

# Try to import MediaPipe with better error handling
mp = None
try:
    import mediapipe as mp
    mp_version = getattr(mp, '__version__', 'unknown')
    print(f"MediaPipe imported successfully, version: {mp_version}")
    # Verify MediaPipe is properly installed (0.10.21 has solutions, 0.10.31+ doesn't)
    if not hasattr(mp, 'solutions'):
        print(f"WARNING: MediaPipe {mp_version} imported but 'solutions' attribute missing")
        print("This means you're using MediaPipe 0.10.30+ which removed solutions API")
        print("Downgrade to 0.10.21 or update to use Tasks API")
        mp = None
    else:
        print("MediaPipe solutions module available")
except ImportError as e:
    print(f"ERROR: MediaPipe import failed: {e}")
    print(f"Python path: {os.sys.path}")
    mp = None
except Exception as e:
    print(f"ERROR: Unexpected error importing MediaPipe: {e}")
    mp = None

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Face Mesh (lazy initialization)
face_mesh = None

def get_face_mesh():
    """Lazy initialization of MediaPipe Face Mesh"""
    global face_mesh
    if face_mesh is None:
        if mp is None:
            raise RuntimeError("MediaPipe is not installed. Please install it with: pip install mediapipe")
        try:
            if not hasattr(mp, 'solutions'):
                raise RuntimeError("MediaPipe solutions module not available. MediaPipe may not be installed correctly.")
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        except AttributeError as e:
            raise RuntimeError(f"MediaPipe not properly installed: {e}. Try: pip install --upgrade mediapipe")
    return face_mesh

# MediaPipe landmark indices (468 points)
# Key landmarks for calculations
LANDMARKS = {
    # Eyes
    'left_eye_inner': 133,
    'left_eye_outer': 33,
    'right_eye_inner': 362,
    'right_eye_outer': 263,
    'left_eye_top': 159,
    'left_eye_bottom': 145,
    'right_eye_top': 386,
    'right_eye_bottom': 374,
    'left_eye_center': 468,  # Calculated
    'right_eye_center': 468,  # Calculated
    
    # Nose
    'nose_tip': 4,
    'nose_bridge': 6,
    'nose_left': 131,
    'nose_right': 360,
    'nose_bottom': 2,
    
    # Mouth
    'mouth_left': 61,
    'mouth_right': 291,
    'mouth_top': 13,
    'mouth_bottom': 14,
    'upper_lip_top': 13,
    'upper_lip_bottom': 14,
    'lower_lip_top': 14,
    'lower_lip_bottom': 18,
    
    # Face outline
    'chin': 18,
    'jaw_left': 172,
    'jaw_right': 397,
    'forehead_center': 10,
    'left_cheek': 116,
    'right_cheek': 345,
    
    # Brows
    'left_brow_inner': 107,
    'left_brow_outer': 70,
    'right_brow_inner': 336,
    'right_brow_outer': 300,
    
    # Additional points
    'glabella': 10,  # Between brows
    'nasion': 6,  # Between eyes
    'subnasale': 2,  # Below nose
    'pronasale': 4,  # Nose tip
    'menton': 18,  # Chin
    'zygion_left': 116,
    'zygion_right': 345,
    'gonion_left': 172,
    'gonion_right': 397,
}

def image_from_bytes(image_bytes):
    """Convert image bytes to numpy array"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_landmarks(image_rgb):
    """Extract face mesh landmarks"""
    mesh = get_face_mesh()
    results = mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    
    landmarks = results.multi_face_landmarks[0].landmark
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Calculate eye centers
    left_eye_center = (points[LANDMARKS['left_eye_inner']] + points[LANDMARKS['left_eye_outer']]) / 2
    right_eye_center = (points[LANDMARKS['right_eye_inner']] + points[LANDMARKS['right_eye_outer']]) / 2
    
    # Add calculated points
    points = np.vstack([points, left_eye_center, right_eye_center])
    
    return points

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(p1 - p2)

def angle_between_vectors(v1, v2):
    """Calculate angle between two vectors in degrees"""
    try:
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    except:
        return 0.0

def score_metric(value, ideal_min, ideal_max, method='linear'):
    """Convert raw metric value to 0-100 score
    
    More realistic scoring that provides better separation:
    - 100 at center of ideal range
    - 50 at the edges of ideal range
    - Goes down toward 0 as you move further away
    """
    try:
        if np.isnan(value) or np.isinf(value):
            return 50.0
        
        center = (ideal_min + ideal_max) / 2.0
        half_range = (ideal_max - ideal_min) / 2.0
        
        if half_range <= 0:
            return 50.0
        
        # Distance from center in "half-range units"
        d = abs(value - center) / half_range
        
        if d <= 1.0:
            # Inside the ideal window → 100 down to 50 (linear)
            return float(100.0 - 50.0 * d)
        else:
            # Outside the ideal window → 50 down towards 0
            return float(max(0.0, 50.0 - 20.0 * (d - 1.0)))
    except:
        return 50.0

def calculate_ipd(landmarks):
    """Calculate interpupillary distance"""
    try:
        left_eye = landmarks[LANDMARKS['left_eye_inner']]
        right_eye = landmarks[LANDMARKS['right_eye_inner']]
        ipd = euclidean_distance(left_eye, right_eye)
        if ipd <= 0 or np.isnan(ipd) or np.isinf(ipd):
            return 0.06  # Default fallback
        return ipd
    except:
        return 0.06

def normalize_by_ipd(distance, ipd):
    """Normalize distance by IPD"""
    try:
        if ipd is None or ipd <= 0 or np.isnan(ipd) or np.isinf(ipd):
            return distance
        result = distance / ipd
        return result if not (np.isnan(result) or np.isinf(result)) else distance
    except:
        return distance

# ========== EYE METRICS ==========

def calculate_canthal_tilt(landmarks, gender='Male'):
    """Calculate canthal tilt (angle of eye corners)"""
    try:
        left_outer = landmarks[LANDMARKS['left_eye_outer']]
        left_inner = landmarks[LANDMARKS['left_eye_inner']]
        right_outer = landmarks[LANDMARKS['right_eye_outer']]
        right_inner = landmarks[LANDMARKS['right_eye_inner']]
        
        v_left = left_inner - left_outer
        v_right = right_inner - right_outer
        
        tilt_left = np.degrees(np.arctan2(v_left[1], v_left[0]))
        tilt_right = np.degrees(np.arctan2(v_right[1], v_right[0]))
        tilt = (tilt_left + tilt_right) / 2
        
        # Normalize angle into [-90, 90] range to handle coordinate system issues
        # This prevents angles like 100° or -120° from causing problems
        while tilt > 90:
            tilt -= 180
        while tilt < -90:
            tilt += 180
        
        # Debug logging
        print(f"[TILT DEBUG] left={tilt_left:.2f}°, right={tilt_right:.2f}°, avg={tilt:.2f}° (normalized)")
        
        if np.isnan(tilt) or np.isinf(tilt):
            print(f"[TILT DEBUG] Invalid tilt (NaN/Inf), returning 50.0")
            return 50.0
        
        # Use realistic ideal range - positive tilt (upturned) is preferred
        ideal_min, ideal_max = (-5, 10) if gender == 'Male' else (-3, 12)
        score = score_metric(tilt, ideal_min, ideal_max)
        
        # No extra custom min clamp - let scores reflect actual geometry
        final_score = float(np.clip(score, 0.0, 100.0))
        print(f"[TILT DEBUG] FINAL tilt={tilt:.2f}°, score={final_score:.1f}")
        return final_score
    except Exception as e:
        print(f"ERROR calculating canthal tilt: {e}")
        import traceback
        traceback.print_exc()
        return 50.0

def calculate_eyelid_exposure(landmarks, ipd):
    """Calculate eyelid exposure (eye aperture ratio)"""
    try:
        left_eye_height = euclidean_distance(
            landmarks[LANDMARKS['left_eye_top']],
            landmarks[LANDMARKS['left_eye_bottom']]
        )
        left_eye_width = euclidean_distance(
            landmarks[LANDMARKS['left_eye_inner']],
            landmarks[LANDMARKS['left_eye_outer']]
        )
        
        right_eye_height = euclidean_distance(
            landmarks[LANDMARKS['right_eye_top']],
            landmarks[LANDMARKS['right_eye_bottom']]
        )
        right_eye_width = euclidean_distance(
            landmarks[LANDMARKS['right_eye_inner']],
            landmarks[LANDMARKS['right_eye_outer']]
        )
        
        if left_eye_width <= 0 or right_eye_width <= 0:
            return 50.0
        
        left_aperture = left_eye_height / left_eye_width
        right_aperture = right_eye_height / right_eye_width
        aperture = (left_aperture + right_aperture) / 2
        
        if np.isnan(aperture) or np.isinf(aperture):
            return 50.0
        
        # Ideal range: 0.25-0.35
        return score_metric(aperture, 0.25, 0.35)
    except:
        return 50.0

def calculate_orbital_depth(landmarks, ipd):
    """Calculate orbital depth using 3D z-values"""
    try:
        left_eye_center = landmarks[468]  # Calculated center
        left_brow = landmarks[LANDMARKS['left_brow_inner']]
        left_cheek = landmarks[LANDMARKS['left_cheek']]
        
        depth = left_eye_center[2] - (left_brow[2] + left_cheek[2]) / 2
        depth_norm = normalize_by_ipd(depth, ipd)
        
        if np.isnan(depth_norm) or np.isinf(depth_norm):
            return 50.0
        
        # More negative = deeper set (generally more attractive)
        # Ideal range: -0.05 to -0.15
        return score_metric(abs(depth_norm), 0.05, 0.15)
    except:
        return 50.0

def calculate_eyebrow_density(landmarks):
    """Proxy for eyebrow density (placeholder - would need CNN in production)"""
    # Return neutral 50 instead of inflated 75
    # TODO: Implement actual eyebrow density calculation
    return 50.0

def calculate_eyelash_density(landmarks):
    """Proxy for eyelash density (placeholder)"""
    # Return neutral 50 instead of inflated 78
    # TODO: Implement actual eyelash density calculation
    return 50.0

def calculate_under_eye_health(landmarks):
    """Proxy for under-eye health (placeholder - would need CNN)"""
    # Return neutral 50 instead of inflated 80
    # TODO: Implement actual under-eye health calculation
    return 50.0

# ========== MIDFACE METRICS ==========

def calculate_cheekbones(landmarks, ipd):
    """Calculate cheekbone prominence"""
    try:
        bizygomatic_width = euclidean_distance(
            landmarks[LANDMARKS['zygion_left']],
            landmarks[LANDMARKS['zygion_right']]
        )
        bizygomatic_norm = normalize_by_ipd(bizygomatic_width, ipd)
        
        if np.isnan(bizygomatic_norm) or np.isinf(bizygomatic_norm):
            return 50.0
        
        # Calibrated range based on actual MediaPipe normalized values (typically 3-7)
        # Use wider range to avoid zero scores
        return score_metric(bizygomatic_norm, 3.5, 6.0)
    except:
        return 50.0

def calculate_maxilla_projection(landmarks, ipd):
    """Calculate maxilla (midface) forward projection"""
    try:
        subnasale = landmarks[LANDMARKS['subnasale']]
        # Use negative z as forward projection
        projection = -subnasale[2]
        projection_norm = normalize_by_ipd(projection, ipd)
        
        if np.isnan(projection_norm) or np.isinf(projection_norm):
            return 50.0
        
        # Calibrated range - z values normalized by IPD are typically larger
        # Use wider range to avoid zero scores
        return score_metric(projection_norm, 0.5, 3.0)
    except:
        return 50.0

def calculate_nose_metrics(landmarks, ipd):
    """Calculate nose length and projection"""
    try:
        nasion = landmarks[LANDMARKS['nasion']]
        pronasale = landmarks[LANDMARKS['pronasale']]
        subnasale = landmarks[LANDMARKS['subnasale']]
        chin = landmarks[LANDMARKS['menton']]
        
        nose_length = euclidean_distance(nasion, subnasale)
        face_length = euclidean_distance(nasion, chin)
        
        if face_length <= 0:
            return 50.0
        
        nose_ratio = nose_length / face_length
        
        nose_projection = -pronasale[2]
        nose_proj_norm = normalize_by_ipd(nose_projection, ipd)
        
        if np.isnan(nose_ratio) or np.isnan(nose_proj_norm):
            return 50.0
        
        # Calibrated ranges - nose ratio is reasonable, but projection needs wider range
        length_score = score_metric(nose_ratio, 0.25, 0.40)
        proj_score = score_metric(nose_proj_norm, 0.3, 2.5)
        
        result = (length_score + proj_score) / 2
        return result if not (np.isnan(result) or np.isinf(result)) else 50.0
    except:
        return 50.0

def calculate_ipd_score(landmarks):
    """Calculate IPD score (interpupillary distance)"""
    try:
        ipd = calculate_ipd(landmarks)
        face_width = euclidean_distance(
            landmarks[LANDMARKS['zygion_left']],
            landmarks[LANDMARKS['zygion_right']]
        )
        
        if face_width <= 0 or np.isnan(face_width):
            return 50.0
        
        ipd_ratio = ipd / face_width
        
        if np.isnan(ipd_ratio) or np.isinf(ipd_ratio):
            return 50.0
        
        # Ideal range: 0.45-0.50
        return score_metric(ipd_ratio, 0.45, 0.50)
    except:
        return 50.0

def calculate_fwhr(landmarks):
    """Calculate Facial Width-to-Height Ratio"""
    try:
        bizygomatic_width = euclidean_distance(
            landmarks[LANDMARKS['zygion_left']],
            landmarks[LANDMARKS['zygion_right']]
        )
        face_height = euclidean_distance(
            landmarks[LANDMARKS['forehead_center']],
            landmarks[LANDMARKS['menton']]
        )
        
        if face_height <= 0 or np.isnan(face_height) or np.isnan(bizygomatic_width):
            return 50.0
        
        fwhr = bizygomatic_width / face_height
        
        if np.isnan(fwhr) or np.isinf(fwhr):
            return 50.0
        
        # Wider range - fWHR typically 1.5-2.5 for normal faces
        ideal_min, ideal_max = (1.6, 2.4) if True else (1.5, 2.3)  # Default to male
        return score_metric(fwhr, ideal_min, ideal_max)
    except:
        return 50.0

def calculate_compactness(landmarks):
    """Calculate face compactness"""
    try:
        face_width = euclidean_distance(
            landmarks[LANDMARKS['zygion_left']],
            landmarks[LANDMARKS['zygion_right']]
        )
        face_height = euclidean_distance(
            landmarks[LANDMARKS['forehead_center']],
            landmarks[LANDMARKS['menton']]
        )
        
        if face_width <= 0 or np.isnan(face_width) or np.isnan(face_height):
            return 50.0
        
        compactness = face_height / face_width
        
        if np.isnan(compactness) or np.isinf(compactness):
            return 50.0
        
        # Wider range - compactness typically 1.0-1.6 for normal faces
        return score_metric(compactness, 1.1, 1.5)
    except:
        return 50.0

# ========== LOWER THIRD METRICS ==========

def calculate_lips(landmarks, ipd):
    """Calculate lip fullness"""
    try:
        upper_lip_thickness = euclidean_distance(
            landmarks[LANDMARKS['upper_lip_top']],
            landmarks[LANDMARKS['upper_lip_bottom']]
        )
        lower_lip_thickness = euclidean_distance(
            landmarks[LANDMARKS['lower_lip_top']],
            landmarks[LANDMARKS['lower_lip_bottom']]
        )
        mouth_width = euclidean_distance(
            landmarks[LANDMARKS['mouth_left']],
            landmarks[LANDMARKS['mouth_right']]
        )
        
        if mouth_width <= 0:
            return 50.0
        
        upper_ratio = upper_lip_thickness / mouth_width
        lower_ratio = lower_lip_thickness / mouth_width
        fullness = (upper_ratio + lower_ratio) / 2
        
        if np.isnan(fullness) or np.isinf(fullness):
            return 50.0
        
        # Ideal range: 0.08-0.12
        return score_metric(fullness, 0.08, 0.12)
    except:
        return 50.0

def calculate_mandible(landmarks, ipd):
    """Calculate mandible length"""
    try:
        gonion_left = landmarks[LANDMARKS['gonion_left']]
        chin = landmarks[LANDMARKS['menton']]
        face_height = euclidean_distance(
            landmarks[LANDMARKS['glabella']],
            landmarks[LANDMARKS['menton']]
        )
        
        if face_height <= 0:
            return 50.0
        
        mandible_length = euclidean_distance(gonion_left, chin)
        mandible_ratio = mandible_length / face_height
        
        if np.isnan(mandible_ratio) or np.isinf(mandible_ratio):
            return 50.0
        
        # Wider range - mandible ratio typically 0.25-0.50 for normal faces
        return score_metric(mandible_ratio, 0.30, 0.50)
    except:
        return 50.0

def calculate_gonial_angle(landmarks):
    """Calculate gonial angle (jaw angle)"""
    try:
        gonion = landmarks[LANDMARKS['gonion_left']]
        chin = landmarks[LANDMARKS['menton']]
        ramus_top = landmarks[LANDMARKS['jaw_left']]
        
        v1 = chin - gonion
        v2 = ramus_top - gonion
        
        angle = angle_between_vectors(v1, v2)
        
        if np.isnan(angle) or np.isinf(angle):
            return None  # Return None for null in JSON
        
        # Ideal range: 115-125° (more acute = more masculine)
        return score_metric(angle, 115, 125)
    except:
        return None

def calculate_ramus(landmarks, ipd):
    """Calculate ramus length"""
    try:
        gonion = landmarks[LANDMARKS['gonion_left']]
        ramus_top = landmarks[LANDMARKS['jaw_left']]
        
        ramus_length = euclidean_distance(gonion, ramus_top)
        ramus_norm = normalize_by_ipd(ramus_length, ipd)
        
        if np.isnan(ramus_norm) or np.isinf(ramus_norm):
            return 50.0
        
        # Calibrated range - normalized values are typically larger
        return score_metric(ramus_norm, 1.5, 4.5)
    except:
        return 50.0

def calculate_hyoid_skin_tightness(landmarks, ipd):
    """Calculate hyoid skin tightness (neck sag)"""
    try:
        chin = landmarks[LANDMARKS['menton']]
        # Approximate neck points
        neck_base = landmarks[LANDMARKS['jaw_left']]
        
        straight = euclidean_distance(chin, neck_base)
        # Placeholder for curve calculation
        curve = straight * 1.1  # Would need actual hyoid point
        
        if straight <= 0:
            return 50.0
        
        sag_ratio = curve / straight
        
        if np.isnan(sag_ratio) or np.isinf(sag_ratio):
            return 50.0
        
        # Lower ratio = tighter (better)
        return score_metric(1.0 / sag_ratio, 0.85, 1.0)
    except:
        return 50.0

def calculate_jaw_width(landmarks, ipd):
    """Calculate jaw width"""
    try:
        jaw_width = euclidean_distance(
            landmarks[LANDMARKS['gonion_left']],
            landmarks[LANDMARKS['gonion_right']]
        )
        face_width = euclidean_distance(
            landmarks[LANDMARKS['zygion_left']],
            landmarks[LANDMARKS['zygion_right']]
        )
        
        if face_width <= 0:
            return 50.0
        
        jaw_ratio = jaw_width / face_width
        
        if np.isnan(jaw_ratio) or np.isinf(jaw_ratio):
            return 50.0
        
        # Calibrated range - wider to avoid zeros
        return score_metric(jaw_ratio, 0.55, 0.85)
    except:
        return 50.0

# ========== UPPER THIRD METRICS ==========

def calculate_forehead_slope(landmarks):
    """Calculate forehead slope"""
    try:
        glabella = landmarks[LANDMARKS['glabella']]
        forehead_top = landmarks[LANDMARKS['forehead_center']]
        
        slope = np.degrees(np.arctan2(
            forehead_top[1] - glabella[1],
            abs(forehead_top[0] - glabella[0])
        ))
        
        if np.isnan(slope) or np.isinf(slope):
            return None
        
        # Ideal range: 5-15° (slight backward slope)
        return score_metric(abs(slope), 5, 15)
    except:
        return None

def calculate_norwood_stage(landmarks):
    """Calculate Norwood stage (hairline recession) - placeholder"""
    # Return neutral 50 instead of inflated 85
    # TODO: Implement actual Norwood stage calculation
    return 50.0

def calculate_forehead_projection(landmarks, ipd):
    """Calculate forehead projection"""
    try:
        forehead = landmarks[LANDMARKS['forehead_center']]
        projection = -forehead[2]
        projection_norm = normalize_by_ipd(projection, ipd)
        
        if np.isnan(projection_norm) or np.isinf(projection_norm):
            return 50.0
        
        # Calibrated range - z values normalized by IPD are typically larger
        return score_metric(projection_norm, 0.3, 2.5)
    except:
        return 50.0

def calculate_hairline_recession(landmarks):
    """Calculate hairline recession - placeholder"""
    # Return neutral 50 instead of inflated 82
    # TODO: Implement actual hairline recession calculation
    return 50.0

def calculate_hair_thinning(landmarks):
    """Calculate hair thinning - placeholder"""
    # Return neutral 50 instead of inflated 80
    # TODO: Implement actual hair thinning calculation
    return 50.0

def calculate_hairline_density(landmarks):
    """Calculate hairline density - placeholder"""
    # Return neutral 50 instead of inflated 83
    # TODO: Implement actual hairline density calculation
    return 50.0

# ========== MISCELLANEOUS METRICS ==========

def calculate_symmetry(landmarks):
    """Calculate facial symmetry"""
    try:
        # Compare left and right side landmarks
        symmetric_pairs = [
            (LANDMARKS['left_eye_outer'], LANDMARKS['right_eye_outer']),
            (LANDMARKS['left_eye_inner'], LANDMARKS['right_eye_inner']),
            (LANDMARKS['left_cheek'], LANDMARKS['right_cheek']),
            (LANDMARKS['zygion_left'], LANDMARKS['zygion_right']),
        ]
        
        total_diff = 0
        count = 0
        
        for left_idx, right_idx in symmetric_pairs:
            left_point = landmarks[left_idx]
            right_point = landmarks[right_idx]
            
            # Calculate horizontal distance from center
            left_dist = abs(left_point[0] - 0.5)
            right_dist = abs(right_point[0] - 0.5)
            
            diff = abs(left_dist - right_dist)
            total_diff += diff
            count += 1
        
        if count == 0:
            return 50.0
        
        avg_diff = total_diff / count
        
        if np.isnan(avg_diff) or np.isinf(avg_diff):
            return 50.0
        
        # Lower diff = more symmetric (better)
        # Convert to 0-100 score (inverse)
        symmetry_score = 100 * (1 - avg_diff * 2)
        return float(np.clip(symmetry_score, 0, 100))
    except:
        return 50.0

def calculate_neck_width(landmarks, ipd):
    """Calculate neck width"""
    try:
        # Approximate neck points
        neck_left = landmarks[LANDMARKS['jaw_left']]
        neck_right = landmarks[LANDMARKS['jaw_right']]
        
        neck_width = euclidean_distance(neck_left, neck_right)
        neck_norm = normalize_by_ipd(neck_width, ipd)
        
        if np.isnan(neck_norm) or np.isinf(neck_norm):
            return 50.0
        
        # Calibrated range - normalized values are typically larger
        return score_metric(neck_norm, 2.0, 5.0)
    except:
        return 50.0

def calculate_bloat(landmarks):
    """Calculate facial bloat (soft tissue thickness) - placeholder"""
    # Return neutral 50 instead of inflated 78
    # TODO: Implement actual facial bloat calculation
    return 50.0

def calculate_bone_mass(landmarks, ipd):
    """Calculate bone mass proxy (facial structure)"""
    try:
        # Use combination of jaw width, cheekbone width, etc.
        jaw_width = euclidean_distance(
            landmarks[LANDMARKS['gonion_left']],
            landmarks[LANDMARKS['gonion_right']]
        )
        cheekbone_width = euclidean_distance(
            landmarks[LANDMARKS['zygion_left']],
            landmarks[LANDMARKS['zygion_right']]
        )
        
        bone_mass = (jaw_width + cheekbone_width) / 2
        bone_mass_norm = normalize_by_ipd(bone_mass, ipd)
        
        if np.isnan(bone_mass_norm) or np.isinf(bone_mass_norm):
            return 50.0
        
        # Calibrated range - normalized values are typically 3-6
        return score_metric(bone_mass_norm, 3.0, 6.5)
    except:
        return 50.0

def calculate_skin_quality(landmarks):
    """Calculate skin quality - placeholder (would need CNN)"""
    # Return neutral 50 instead of inflated 84
    # TODO: Implement actual skin quality calculation using CNN
    return 50.0

def calculate_harmony(landmarks):
    """Calculate facial harmony (meta-score)"""
    try:
        # Average of key proportions
        ipd = calculate_ipd(landmarks)
        
        # Get a few key metrics
        fwhr = calculate_fwhr(landmarks)
        compactness = calculate_compactness(landmarks)
        symmetry = calculate_symmetry(landmarks)
        
        # Average them
        harmony = (fwhr + compactness + symmetry) / 3
        
        if np.isnan(harmony) or np.isinf(harmony):
            return 50.0
        
        return float(np.clip(harmony, 0, 100))
    except:
        return 50.0

def calculate_ascension_date():
    """Calculate projected ascension date based on potential"""
    # Add 30-180 days from today
    days = np.random.randint(30, 180)
    return (datetime.now() + timedelta(days=days)).isoformat() + 'Z'

def sanitize_for_json(obj):
    """Recursively replace NaN and Inf with None for JSON serialization"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(elem) for elem in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj

def calculate_all_metrics(front_landmarks, side_landmarks, gender='Male'):
    """Calculate all facial metrics
    
    Note: side_landmarks is currently accepted but not used in calculations.
    Future versions can use side profile for more accurate 3D metrics like
    nose projection, mandible angle, and chin-neck angle.
    """
    try:
        ipd = calculate_ipd(front_landmarks)
        
        # Ensure IPD is valid
        if ipd <= 0 or np.isnan(ipd):
            print("WARNING: Invalid IPD, using fallback")
            ipd = 0.06
        
        # Overall
        eyes_avg = (
            calculate_canthal_tilt(front_landmarks, gender) +
            calculate_eyelid_exposure(front_landmarks, ipd) +
            calculate_orbital_depth(front_landmarks, ipd) +
            calculate_eyebrow_density(front_landmarks) +
            calculate_eyelash_density(front_landmarks) +
            calculate_under_eye_health(front_landmarks)
        ) / 6
        
        midface_avg = (
            calculate_cheekbones(front_landmarks, ipd) +
            calculate_maxilla_projection(front_landmarks, ipd) +
            calculate_nose_metrics(front_landmarks, ipd) +
            calculate_ipd_score(front_landmarks) +
            calculate_fwhr(front_landmarks) +
            calculate_compactness(front_landmarks)
        ) / 6
        
        lower_third_avg = (
            calculate_lips(front_landmarks, ipd) +
            calculate_mandible(front_landmarks, ipd) +
            (calculate_gonial_angle(front_landmarks) or 50.0) +
            calculate_ramus(front_landmarks, ipd) +
            calculate_hyoid_skin_tightness(front_landmarks, ipd) +
            calculate_jaw_width(front_landmarks, ipd)
        ) / 6
        
        upper_third_avg = (
            (calculate_forehead_slope(front_landmarks) or 50.0) +
            calculate_norwood_stage(front_landmarks) +
            calculate_forehead_projection(front_landmarks, ipd) +
            calculate_hairline_recession(front_landmarks) +
            calculate_hair_thinning(front_landmarks) +
            calculate_hairline_density(front_landmarks)
        ) / 6
        
        misc_avg = (
            calculate_skin_quality(front_landmarks) +
            calculate_harmony(front_landmarks) +
            calculate_symmetry(front_landmarks) +
            calculate_neck_width(front_landmarks, ipd) +
            calculate_bloat(front_landmarks) +
            calculate_bone_mass(front_landmarks, ipd)
        ) / 6
        
        # Calculate raw PSL without any rescaling or inflation
        psl = (eyes_avg + midface_avg + lower_third_avg + upper_third_avg + misc_avg) / 5.0
        
        # Potential is same as PSL (no artificial boost)
        # In the future, you could add a small fixed offset like psl + 5 if desired
        potential = psl
        
        # Ensure no NaN values
        if np.isnan(psl) or np.isinf(psl):
            psl = None
        if np.isnan(potential) or np.isinf(potential):
            potential = None
        
        return {
            'overall': {
                'psl': round(psl, 1) if psl is not None else None,
                'potential': round(potential, 1) if potential is not None else None
            },
            'eyes': {
                'orbitalDepth': round(calculate_orbital_depth(front_landmarks, ipd), 1),
                'canthalTilt': round(calculate_canthal_tilt(front_landmarks, gender), 1),
                'eyebrowDensity': round(calculate_eyebrow_density(front_landmarks), 1),
                'eyelashDensity': round(calculate_eyelash_density(front_landmarks), 1),
                'eyelidExposure': round(calculate_eyelid_exposure(front_landmarks, ipd), 1),
                'underEyeHealth': round(calculate_under_eye_health(front_landmarks), 1)
            },
            'midface': {
                'cheekbones': round(calculate_cheekbones(front_landmarks, ipd), 1),
                'maxilla': round(calculate_maxilla_projection(front_landmarks, ipd), 1),
                'nose': round(calculate_nose_metrics(front_landmarks, ipd), 1),
                'ipd': round(calculate_ipd_score(front_landmarks), 1),
                'fwhr': round(calculate_fwhr(front_landmarks), 1),
                'compactness': round(calculate_compactness(front_landmarks), 1)
            },
            'lowerThird': {
                'lips': round(calculate_lips(front_landmarks, ipd), 1),
                'mandible': round(calculate_mandible(front_landmarks, ipd), 1),
                'gonialAngle': calculate_gonial_angle(front_landmarks),
                'ramus': round(calculate_ramus(front_landmarks, ipd), 1),
                'hyoidSkinTightness': round(calculate_hyoid_skin_tightness(front_landmarks, ipd), 1),
                'jawWidth': round(calculate_jaw_width(front_landmarks, ipd), 1)
            },
            'upperThird': {
                'norwoodStage': round(calculate_norwood_stage(front_landmarks), 1),
                'foreheadProjection': round(calculate_forehead_projection(front_landmarks, ipd), 1),
                'hairlineRecession': round(calculate_hairline_recession(front_landmarks), 1),
                'hairThinning': round(calculate_hair_thinning(front_landmarks), 1),
                'hairlineDensity': round(calculate_hairline_density(front_landmarks), 1),
                'foreheadSlope': calculate_forehead_slope(front_landmarks)
            },
            'miscellaneous': {
                'skin': round(calculate_skin_quality(front_landmarks), 1),
                'harmony': round(calculate_harmony(front_landmarks), 1),
                'symmetry': round(calculate_symmetry(front_landmarks), 1),
                'neckWidth': round(calculate_neck_width(front_landmarks, ipd), 1),
                'bloat': round(calculate_bloat(front_landmarks), 1),
                'boneMass': round(calculate_bone_mass(front_landmarks, ipd), 1)
            },
            'ascensionDate': calculate_ascension_date()
        }
    except Exception as e:
        print(f"ERROR in calculate_all_metrics: {e}")
        import traceback
        traceback.print_exc()
        # Return error structure instead of mock results
        # This will be caught by the error handler in analyze_face()
        raise Exception(f"Failed to calculate metrics: {e}")

def detect_gender_from_image(image_array):
    """Detect gender from image using DeepFace (97%+ accuracy)"""
    try:
        if not DEEPFACE_AVAILABLE:
            return None
        
        # DeepFace expects BGR format (OpenCV default)
        # Analyze gender with high accuracy backend
        result = DeepFace.analyze(
            img_path=image_array,
            actions=['gender'],
            enforce_detection=False,  # Don't fail if face detection is uncertain
            detector_backend='retinaface',  # Most accurate detector
            silent=True  # Suppress verbose output
        )
        
        # Handle both single dict and list of dicts
        if isinstance(result, list):
            result = result[0]
        
        # Extract gender prediction
        gender_pred = result.get('gender', '')
        # DeepFace returns "Man" or "Woman"
        if 'Man' in gender_pred or 'Male' in gender_pred:
            return 'Male'
        elif 'Woman' in gender_pred or 'Female' in gender_pred:
            return 'Female'
        else:
            return None
    except Exception as e:
        print(f"Gender detection error: {e}")
        return None

@app.route('/api/analyze-face', methods=['POST'])
def analyze_face():
    try:
        if 'front_image' not in request.files or 'side_image' not in request.files:
            return jsonify({'error': 'Missing images'}), 400
        
        front_file = request.files['front_image']
        side_file = request.files['side_image']
        
        # Get gender from form, or detect automatically
        gender = request.form.get('gender', '').strip()
        
        # If gender not provided, detect it from the front image
        if not gender:
            try:
                # Read front image for gender detection
                front_image_bytes = front_file.read()
                front_file.seek(0)  # Reset file pointer for later use
                
                # Convert to numpy array for DeepFace
                nparr = np.frombuffer(front_image_bytes, np.uint8)
                front_image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if front_image_cv is None:
                    return jsonify({'error': 'Invalid front image format'}), 400
                
                # Detect gender
                detected_gender = detect_gender_from_image(front_image_cv)
                
                if detected_gender:
                    gender = detected_gender
                    print(f"✅ Gender detected automatically: {gender}")
                else:
                    # Fallback to Male if detection fails
                    gender = 'Male'
                    print(f"⚠️ Gender detection failed, defaulting to: {gender}")
            except Exception as e:
                print(f"⚠️ Gender detection error: {e}, defaulting to Male")
                gender = 'Male'
        
        # Ensure gender is valid
        if gender not in ['Male', 'Female']:
            gender = 'Male'  # Default fallback
        
        # Check if MediaPipe is available
        if mp is None or not hasattr(mp, 'solutions'):
            # In production, return error instead of mock results
            print("ERROR: MediaPipe not available")
            return jsonify({
                'error': 'Face analysis service is currently unavailable. Please try again later.'
            }), 503
        
        # Read image bytes
        front_bytes = front_file.read()
        side_bytes = side_file.read()
        
        # Convert to images
        front_image = image_from_bytes(front_bytes)
        side_image = image_from_bytes(side_bytes)
        
        if front_image is None or side_image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Get landmarks
        front_landmarks = get_landmarks(front_image)
        side_landmarks = get_landmarks(side_image)
        
        if front_landmarks is None:
            return jsonify({'error': 'Face not detected in front image'}), 400
        
        if side_landmarks is None:
            return jsonify({'error': 'Face not detected in side image'}), 400
        
        # Calculate all metrics
        try:
            results = calculate_all_metrics(front_landmarks, side_landmarks, gender)
        except Exception as e:
            print(f"ERROR calculating metrics: {e}")
            import traceback
            traceback.print_exc()
            # Return error instead of mock results
            return jsonify({
                'error': f'Failed to calculate metrics: {str(e)}. Please ensure your face is clearly visible and well-lit in both photos.'
            }), 500
        
        # Sanitize NaN/Inf values to None (null in JSON)
        results = sanitize_for_json(results)
        
        return jsonify(results), 200
        
    except Exception as e:
        print(f"ERROR in analyze_face: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint with MediaPipe status"""
    status = {
        'status': 'healthy',
        'mediapipe_installed': mp is not None,
        'mediapipe_has_solutions': mp is not None and hasattr(mp, 'solutions'),
        'python_version': str(os.sys.version),
    }
    if mp is not None:
        status['mediapipe_version'] = getattr(mp, '__version__', 'unknown')
    
    return jsonify(status), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

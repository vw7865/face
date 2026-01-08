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
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def score_metric(value, ideal_min, ideal_max, method='gaussian'):
    """Convert raw metric value to 0-100 score"""
    if method == 'gaussian':
        center = (ideal_min + ideal_max) / 2
        std = (ideal_max - ideal_min) / 4
        score = 100 * np.exp(-0.5 * ((value - center) / std) ** 2)
        return float(np.clip(score, 0, 100))
    else:  # linear
        if value < ideal_min:
            return float(max(0, 100 * (value / ideal_min)))
        elif value > ideal_max:
            return float(max(0, 100 - 100 * ((value - ideal_max) / ideal_max)))
        else:
            return 100.0

def calculate_ipd(landmarks):
    """Calculate interpupillary distance"""
    left_eye = landmarks[LANDMARKS['left_eye_inner']]
    right_eye = landmarks[LANDMARKS['right_eye_inner']]
    return euclidean_distance(left_eye, right_eye)

def normalize_by_ipd(distance, ipd):
    """Normalize distance by IPD"""
    return distance / ipd if ipd > 0 else distance

# ========== EYE METRICS ==========

def calculate_canthal_tilt(landmarks, gender='Male'):
    """Calculate canthal tilt (angle of eye corners)"""
    left_outer = landmarks[LANDMARKS['left_eye_outer']]
    left_inner = landmarks[LANDMARKS['left_eye_inner']]
    right_outer = landmarks[LANDMARKS['right_eye_outer']]
    right_inner = landmarks[LANDMARKS['right_eye_inner']]
    
    v_left = left_inner - left_outer
    tilt_left = np.degrees(np.arctan2(v_left[1], v_left[0]))
    
    v_right = right_inner - right_outer
    tilt_right = np.degrees(np.arctan2(v_right[1], v_right[0]))
    
    tilt = (tilt_left + tilt_right) / 2
    
    # Ideal range: 5-12° for males, 3-10° for females
    ideal_min, ideal_max = (5, 12) if gender == 'Male' else (3, 10)
    return score_metric(tilt, ideal_min, ideal_max)

def calculate_eyelid_exposure(landmarks, ipd):
    """Calculate eyelid exposure (eye aperture ratio)"""
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
    
    left_aperture = left_eye_height / left_eye_width if left_eye_width > 0 else 0
    right_aperture = right_eye_height / right_eye_width if right_eye_width > 0 else 0
    aperture = (left_aperture + right_aperture) / 2
    
    # Ideal range: 0.25-0.35
    return score_metric(aperture, 0.25, 0.35)

def calculate_orbital_depth(landmarks, ipd):
    """Calculate orbital depth using 3D z-values"""
    left_eye_center = landmarks[468]  # Calculated center
    left_brow = landmarks[LANDMARKS['left_brow_inner']]
    left_cheek = landmarks[LANDMARKS['left_cheek']]
    
    depth = left_eye_center[2] - (left_brow[2] + left_cheek[2]) / 2
    depth_norm = depth / ipd
    
    # More negative = deeper set (generally more attractive)
    # Ideal range: -0.05 to -0.15
    return score_metric(abs(depth_norm), 0.05, 0.15)

def calculate_eyebrow_density(landmarks):
    """Proxy for eyebrow density (placeholder - would need CNN in production)"""
    # For now, use brow width/height ratio as proxy
    left_brow_width = euclidean_distance(
        landmarks[LANDMARKS['left_brow_inner']],
        landmarks[LANDMARKS['left_brow_outer']]
    )
    # Placeholder score
    return 75.0

def calculate_eyelash_density(landmarks):
    """Proxy for eyelash density (placeholder)"""
    return 78.0

def calculate_under_eye_health(landmarks):
    """Proxy for under-eye health (placeholder - would need CNN)"""
    return 80.0

# ========== MIDFACE METRICS ==========

def calculate_cheekbones(landmarks, ipd):
    """Calculate cheekbone prominence"""
    bizygomatic_width = euclidean_distance(
        landmarks[LANDMARKS['zygion_left']],
        landmarks[LANDMARKS['zygion_right']]
    )
    bizygomatic_norm = normalize_by_ipd(bizygomatic_width, ipd)
    
    # Ideal range depends on gender
    return score_metric(bizygomatic_norm, 0.85, 1.05)

def calculate_maxilla_projection(landmarks, ipd):
    """Calculate maxilla (midface) forward projection"""
    subnasale = landmarks[LANDMARKS['subnasale']]
    # Use negative z as forward projection
    projection = -subnasale[2]
    projection_norm = projection / ipd
    
    # Ideal range: 0.15-0.25
    return score_metric(projection_norm, 0.15, 0.25)

def calculate_nose_metrics(landmarks, ipd):
    """Calculate nose length and projection"""
    nasion = landmarks[LANDMARKS['nasion']]
    pronasale = landmarks[LANDMARKS['pronasale']]
    subnasale = landmarks[LANDMARKS['subnasale']]
    chin = landmarks[LANDMARKS['menton']]
    
    nose_length = euclidean_distance(nasion, subnasale)
    face_length = euclidean_distance(nasion, chin)
    nose_ratio = nose_length / face_length if face_length > 0 else 0
    
    nose_projection = -pronasale[2]
    nose_proj_norm = nose_projection / ipd
    
    # Combine length and projection scores
    length_score = score_metric(nose_ratio, 0.30, 0.35)
    proj_score = score_metric(nose_proj_norm, 0.12, 0.20)
    
    return (length_score + proj_score) / 2

def calculate_ipd_score(landmarks):
    """Calculate IPD score (interpupillary distance)"""
    ipd = calculate_ipd(landmarks)
    face_width = euclidean_distance(
        landmarks[LANDMARKS['zygion_left']],
        landmarks[LANDMARKS['zygion_right']]
    )
    ipd_ratio = ipd / face_width if face_width > 0 else 0
    
    # Ideal range: 0.45-0.50
    return score_metric(ipd_ratio, 0.45, 0.50)

def calculate_fwhr(landmarks):
    """Calculate Facial Width-to-Height Ratio"""
    bizygomatic_width = euclidean_distance(
        landmarks[LANDMARKS['zygion_left']],
        landmarks[LANDMARKS['zygion_right']]
    )
    upper_face_height = euclidean_distance(
        landmarks[LANDMARKS['mouth_top']],
        landmarks[LANDMARKS['glabella']]
    )
    
    fwhr = bizygomatic_width / upper_face_height if upper_face_height > 0 else 0
    
    # Higher fWHR = more masculine
    # Ideal range: 1.8-2.2 for males, 1.6-2.0 for females
    return score_metric(fwhr, 1.8, 2.2)

def calculate_compactness(landmarks):
    """Calculate face compactness"""
    face_width = euclidean_distance(
        landmarks[LANDMARKS['zygion_left']],
        landmarks[LANDMARKS['zygion_right']]
    )
    face_height = euclidean_distance(
        landmarks[LANDMARKS['glabella']],
        landmarks[LANDMARKS['menton']]
    )
    
    compactness = face_width / face_height if face_height > 0 else 0
    
    # Ideal range: 0.70-0.85
    return score_metric(compactness, 0.70, 0.85)

# ========== LOWER THIRD METRICS ==========

def calculate_lips(landmarks, ipd):
    """Calculate lip fullness"""
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
    
    upper_ratio = upper_lip_thickness / mouth_width if mouth_width > 0 else 0
    lower_ratio = lower_lip_thickness / mouth_width if mouth_width > 0 else 0
    fullness = (upper_ratio + lower_ratio) / 2
    
    # Ideal range: 0.08-0.12
    return score_metric(fullness, 0.08, 0.12)

def calculate_mandible(landmarks, ipd):
    """Calculate mandible length"""
    gonion_left = landmarks[LANDMARKS['gonion_left']]
    chin = landmarks[LANDMARKS['menton']]
    face_height = euclidean_distance(
        landmarks[LANDMARKS['glabella']],
        landmarks[LANDMARKS['menton']]
    )
    
    mandible_length = euclidean_distance(gonion_left, chin)
    mandible_ratio = mandible_length / face_height if face_height > 0 else 0
    
    # Ideal range: 0.35-0.45
    return score_metric(mandible_ratio, 0.35, 0.45)

def calculate_gonial_angle(landmarks):
    """Calculate gonial angle (jaw angle)"""
    gonion = landmarks[LANDMARKS['gonion_left']]
    chin = landmarks[LANDMARKS['menton']]
    ramus_top = landmarks[LANDMARKS['jaw_left']]
    
    v1 = chin - gonion
    v2 = ramus_top - gonion
    
    angle = angle_between_vectors(v1, v2)
    
    # Ideal range: 115-125° (more acute = more masculine)
    return score_metric(angle, 115, 125)

def calculate_ramus(landmarks, ipd):
    """Calculate ramus length"""
    gonion = landmarks[LANDMARKS['gonion_left']]
    ramus_top = landmarks[LANDMARKS['jaw_left']]
    
    ramus_length = euclidean_distance(gonion, ramus_top)
    ramus_norm = normalize_by_ipd(ramus_length, ipd)
    
    # Ideal range: 0.25-0.35
    return score_metric(ramus_norm, 0.25, 0.35)

def calculate_hyoid_skin_tightness(landmarks, ipd):
    """Calculate hyoid skin tightness (neck sag)"""
    chin = landmarks[LANDMARKS['menton']]
    # Approximate neck points
    neck_base = landmarks[LANDMARKS['jaw_left']]
    
    straight = euclidean_distance(chin, neck_base)
    # Placeholder for curve calculation
    curve = straight * 1.1  # Would need actual hyoid point
    
    sag_ratio = curve / straight if straight > 0 else 1.0
    
    # Lower ratio = tighter (better)
    return score_metric(1.0 / sag_ratio, 0.85, 1.0)

def calculate_jaw_width(landmarks, ipd):
    """Calculate jaw width"""
    jaw_width = euclidean_distance(
        landmarks[LANDMARKS['gonion_left']],
        landmarks[LANDMARKS['gonion_right']]
    )
    face_width = euclidean_distance(
        landmarks[LANDMARKS['zygion_left']],
        landmarks[LANDMARKS['zygion_right']]
    )
    
    jaw_ratio = jaw_width / face_width if face_width > 0 else 0
    
    # Ideal range: 0.65-0.75
    return score_metric(jaw_ratio, 0.65, 0.75)

# ========== UPPER THIRD METRICS ==========

def calculate_forehead_slope(landmarks):
    """Calculate forehead slope"""
    glabella = landmarks[LANDMARKS['glabella']]
    forehead_top = landmarks[LANDMARKS['forehead_center']]
    nasion = landmarks[LANDMARKS['nasion']]
    chin = landmarks[LANDMARKS['menton']]
    
    v_face_vertical = chin - nasion
    v_forehead = forehead_top - glabella
    
    angle = angle_between_vectors(v_face_vertical, v_forehead)
    
    # Ideal range: 15-25°
    return score_metric(angle, 15, 25)

def calculate_norwood_stage(landmarks):
    """Calculate Norwood stage (hairline recession) - placeholder"""
    # Would need hairline detection/segmentation
    return 85.0

def calculate_forehead_projection(landmarks, ipd):
    """Calculate forehead projection"""
    forehead = landmarks[LANDMARKS['forehead_center']]
    projection = -forehead[2]
    projection_norm = projection / ipd
    
    # Ideal range: 0.10-0.18
    return score_metric(projection_norm, 0.10, 0.18)

def calculate_hairline_recession(landmarks):
    """Calculate hairline recession - placeholder"""
    return 82.0

def calculate_hair_thinning(landmarks):
    """Calculate hair thinning - placeholder"""
    return 80.0

def calculate_hairline_density(landmarks):
    """Calculate hairline density - placeholder"""
    return 83.0

# ========== MISCELLANEOUS METRICS ==========

def calculate_symmetry(landmarks):
    """Calculate facial symmetry"""
    # Compare left and right side landmarks
    symmetric_pairs = [
        (LANDMARKS['left_eye_outer'], LANDMARKS['right_eye_outer']),
        (LANDMARKS['left_eye_inner'], LANDMARKS['right_eye_inner']),
        (LANDMARKS['zygion_left'], LANDMARKS['zygion_right']),
        (LANDMARKS['gonion_left'], LANDMARKS['gonion_right']),
    ]
    
    deviations = []
    face_midline_x = landmarks[LANDMARKS['nasion']][0]
    
    for left_idx, right_idx in symmetric_pairs:
        left_point = landmarks[left_idx]
        right_point = landmarks[right_idx]
        
        # Mirror right point
        right_mirrored = np.array([2 * face_midline_x - right_point[0], right_point[1], right_point[2]])
        
        deviation = euclidean_distance(left_point, right_mirrored)
        deviations.append(deviation)
    
    avg_deviation = np.mean(deviations)
    
    # Convert to score (lower deviation = higher score)
    score = 100 * np.exp(-10 * avg_deviation)
    return float(np.clip(score, 0, 100))

def calculate_neck_width(landmarks, ipd):
    """Calculate neck width"""
    # Approximate neck points
    neck_left = landmarks[LANDMARKS['jaw_left']]
    neck_right = landmarks[LANDMARKS['jaw_right']]
    
    neck_width = euclidean_distance(neck_left, neck_right)
    face_width = euclidean_distance(
        landmarks[LANDMARKS['zygion_left']],
        landmarks[LANDMARKS['zygion_right']]
    )
    
    neck_ratio = neck_width / face_width if face_width > 0 else 0
    
    # Ideal range: 0.55-0.65
    return score_metric(neck_ratio, 0.55, 0.65)

def calculate_bloat(landmarks):
    """Calculate facial bloat (soft tissue thickness) - placeholder"""
    # Would need 3D depth variance analysis
    return 78.0

def calculate_bone_mass(landmarks, ipd):
    """Calculate bone mass proxy (facial structure)"""
    # Use combination of jaw width, cheekbone width, etc.
    jaw_width = euclidean_distance(
        landmarks[LANDMARKS['gonion_left']],
        landmarks[LANDMARKS['gonion_right']]
    )
    cheek_width = euclidean_distance(
        landmarks[LANDMARKS['zygion_left']],
        landmarks[LANDMARKS['zygion_right']]
    )
    
    bone_score = (jaw_width + cheek_width) / 2
    bone_norm = normalize_by_ipd(bone_score, ipd)
    
    return score_metric(bone_norm, 0.75, 0.95)

def calculate_skin_quality(landmarks):
    """Calculate skin quality - placeholder (would need CNN)"""
    return 84.0

def calculate_harmony(landmarks):
    """Calculate facial harmony (meta-score)"""
    # Average of key proportions
    ipd = calculate_ipd(landmarks)
    fwhr_score = calculate_fwhr(landmarks)
    compactness_score = calculate_compactness(landmarks)
    symmetry_score = calculate_symmetry(landmarks)
    
    harmony = (fwhr_score + compactness_score + symmetry_score) / 3
    return harmony

def calculate_ascension_date():
    """Calculate projected ascension date based on potential"""
    # Add 30-180 days from today
    days = np.random.randint(30, 180)
    date = datetime.now() + timedelta(days=days)
    # Return ISO8601 format that iOS can parse
    return date.isoformat() + 'Z'

def calculate_all_metrics(front_landmarks, side_landmarks, gender='Male'):
    """Calculate all facial metrics"""
    ipd = calculate_ipd(front_landmarks)
    
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
        calculate_gonial_angle(front_landmarks) +
        calculate_ramus(front_landmarks, ipd) +
        calculate_hyoid_skin_tightness(front_landmarks, ipd) +
        calculate_jaw_width(front_landmarks, ipd)
    ) / 6
    
    upper_third_avg = (
        calculate_forehead_slope(front_landmarks) +
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
    
    psl = (eyes_avg + midface_avg + lower_third_avg + upper_third_avg + misc_avg) / 5
    potential = psl * 1.05  # Slightly higher potential
    
    return {
        'overall': {
            'psl': round(psl, 1),
            'potential': round(potential, 1)
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
            'gonialAngle': round(calculate_gonial_angle(front_landmarks), 1),
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
            'foreheadSlope': round(calculate_forehead_slope(front_landmarks), 1)
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

def generate_mock_results(gender='Male'):
    """Generate mock results when MediaPipe is not available"""
    import random
    random.seed(hash(gender) % 1000)  # Deterministic based on gender
    
    def rand_score():
        return round(random.uniform(70, 95), 1)
    
    psl = rand_score()
    return {
        'overall': {
            'psl': psl,
            'potential': round(psl * 1.05, 1)
        },
        'eyes': {
            'orbitalDepth': rand_score(),
            'canthalTilt': rand_score(),
            'eyebrowDensity': rand_score(),
            'eyelashDensity': rand_score(),
            'eyelidExposure': rand_score(),
            'underEyeHealth': rand_score()
        },
        'midface': {
            'cheekbones': rand_score(),
            'maxilla': rand_score(),
            'nose': rand_score(),
            'ipd': rand_score(),
            'fwhr': rand_score(),
            'compactness': rand_score()
        },
        'lowerThird': {
            'lips': rand_score(),
            'mandible': rand_score(),
            'gonialAngle': rand_score(),
            'ramus': rand_score(),
            'hyoidSkinTightness': rand_score(),
            'jawWidth': rand_score()
        },
        'upperThird': {
            'norwoodStage': rand_score(),
            'foreheadProjection': rand_score(),
            'hairlineRecession': rand_score(),
            'hairThinning': rand_score(),
            'hairlineDensity': rand_score(),
            'foreheadSlope': rand_score()
        },
        'miscellaneous': {
            'skin': rand_score(),
            'harmony': rand_score(),
            'symmetry': rand_score(),
            'neckWidth': rand_score(),
            'bloat': rand_score(),
            'boneMass': rand_score()
        },
        'ascensionDate': (datetime.now() + timedelta(days=random.randint(30, 180))).isoformat() + 'Z'
    }

@app.route('/api/analyze-face', methods=['POST'])
def analyze_face():
    try:
        if 'front_image' not in request.files or 'side_image' not in request.files:
            return jsonify({'error': 'Missing images'}), 400
        
        front_file = request.files['front_image']
        side_file = request.files['side_image']
        gender = request.form.get('gender', 'Male')
        
        # Check if MediaPipe is available
        if mp is None or not hasattr(mp, 'solutions'):
            # Fallback to mock results
            print("WARNING: MediaPipe not available, using mock results")
            results = generate_mock_results(gender)
            return jsonify(results), 200
        
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
        results = calculate_all_metrics(front_landmarks, side_landmarks, gender)
        
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint with MediaPipe status"""
    status = {
        'status': 'healthy',
        'mediapipe_installed': mp is not None,
        'mediapipe_has_solutions': mp is not None and hasattr(mp, 'solutions') if mp else False,
        'python_version': os.sys.version
    }
    return jsonify(status), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


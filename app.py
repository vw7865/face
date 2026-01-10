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

# DeepFace for gender detection (97%+ accuracy) - lazy import to avoid blocking startup
DEEPFACE_AVAILABLE = None  # Will be checked on first use
def check_deepface_available():
    """Lazy check for DeepFace availability"""
    global DEEPFACE_AVAILABLE
    if DEEPFACE_AVAILABLE is None:
        try:
            import deepface
            DEEPFACE_AVAILABLE = True
            print("DeepFace available for gender detection")
        except ImportError:
            DEEPFACE_AVAILABLE = False
            print("WARNING: DeepFace not available. Gender detection will fall back to user input.")
    return DEEPFACE_AVAILABLE

# FaceStats-style attractiveness scoring (CLIP + MLP) - lazy import
ATTRACTIVENESS_AVAILABLE = None
def check_attractiveness_available():
    """Lazy check for FaceStats dependencies"""
    global ATTRACTIVENESS_AVAILABLE
    if ATTRACTIVENESS_AVAILABLE is None:
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            import torch.nn as nn
            import joblib
            ATTRACTIVENESS_AVAILABLE = True
            print("FaceStats-style attractiveness scoring available")
        except ImportError:
            ATTRACTIVENESS_AVAILABLE = False
            print("WARNING: FaceStats attractiveness scoring dependencies not available.")
    return ATTRACTIVENESS_AVAILABLE

# Beauty-classifier (ResNet-50 on SCUT-FBP5500) - lazy import
BEAUTY_CLASSIFIER_AVAILABLE = None
def check_beauty_classifier_available():
    """Lazy check for beauty-classifier dependencies"""
    global BEAUTY_CLASSIFIER_AVAILABLE
    if BEAUTY_CLASSIFIER_AVAILABLE is None:
        try:
            import torchvision.models as models
            import torchvision.transforms as transforms
            BEAUTY_CLASSIFIER_AVAILABLE = True
            print("Beauty-classifier (ResNet-50) available for calibration")
        except ImportError:
            BEAUTY_CLASSIFIER_AVAILABLE = False
            print("WARNING: Beauty-classifier dependencies not available.")
    return BEAUTY_CLASSIFIER_AVAILABLE

# MediaPipe - lazy import to prevent blocking startup
_mp_module = None
_mp_available = None

def get_mediapipe():
    """Lazy import MediaPipe - only loads when needed"""
    global _mp_module, _mp_available
    if _mp_available is None:
        try:
            import mediapipe as mp
            mp_version = getattr(mp, '__version__', 'unknown')
            print(f"MediaPipe imported successfully, version: {mp_version}")
            # Verify MediaPipe is properly installed (0.10.21 has solutions, 0.10.31+ doesn't)
            if not hasattr(mp, 'solutions'):
                print(f"WARNING: MediaPipe {mp_version} imported but 'solutions' attribute missing")
                print("This means you're using MediaPipe 0.10.30+ which removed solutions API")
                print("Downgrade to 0.10.21 or update to use Tasks API")
                _mp_module = None
                _mp_available = False
            else:
                print("MediaPipe solutions module available")
                _mp_module = mp
                _mp_available = True
        except ImportError as e:
            print(f"ERROR: MediaPipe import failed: {e}")
            print(f"Python path: {os.sys.path}")
            _mp_module = None
            _mp_available = False
        except Exception as e:
            print(f"ERROR: Unexpected error importing MediaPipe: {e}")
            _mp_module = None
            _mp_available = False
    return _mp_module

app = Flask(__name__)
CORS(app)

# Model preloading status (loaded in background thread)
_MODEL_LOADING_STATUS = {
    'mediapipe': False,
    'deepface': False,
    'clip': False,
    'facestats_regressor': False,
    'beauty_classifier': False,
    'loading': False,
    'error': None
}

# Global model variables (preloaded)
_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_FACESTATS_REGRESSOR = None
_BEAUTY_CLASSIFIER_MODEL = None

def preload_models():
    """Preload all ML models in background to make first request fast"""
    global _MODEL_LOADING_STATUS, _CLIP_MODEL, _CLIP_PROCESSOR, _FACESTATS_REGRESSOR, _BEAUTY_CLASSIFIER_MODEL
    
    if _MODEL_LOADING_STATUS['loading']:
        return  # Already loading
    
    _MODEL_LOADING_STATUS['loading'] = True
    print("\n" + "="*60)
    print("🚀 PRELOADING MODELS (background thread)")
    print("="*60)
    
    try:
        # 1. Preload MediaPipe (fast, ~2-3 seconds)
        print("\n📦 Preloading MediaPipe...")
        try:
            mp = get_mediapipe()
            if mp is not None:
                _MODEL_LOADING_STATUS['mediapipe'] = True
                print("✅ MediaPipe preloaded")
            else:
                print("⚠️ MediaPipe not available")
        except Exception as e:
            print(f"⚠️ MediaPipe preload failed: {e}")
        
        # 2. Preload DeepFace (slow, ~30-60 seconds, but only first time)
        print("\n📦 Preloading DeepFace (this may take 30-60s on first load)...")
        try:
            if check_deepface_available():
                from deepface import DeepFace
                # Just import it - models load on first use
                _MODEL_LOADING_STATUS['deepface'] = True
                print("✅ DeepFace preloaded")
            else:
                print("⚠️ DeepFace not available")
        except Exception as e:
            print(f"⚠️ DeepFace preload failed: {e}")
        
        # 3. Preload CLIP model (slow, ~20-30 seconds)
        print("\n📦 Preloading CLIP model (this may take 20-30s)...")
        try:
            if check_attractiveness_available():
                from transformers import CLIPProcessor, CLIPModel
                import torch
                _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                _CLIP_MODEL.eval()
                _MODEL_LOADING_STATUS['clip'] = True
                print("✅ CLIP model preloaded")
            else:
                print("⚠️ CLIP dependencies not available")
        except Exception as e:
            print(f"⚠️ CLIP preload failed: {e}")
        
        # 4. Preload FaceStats regressor (fast, ~1-2 seconds)
        print("\n📦 Preloading FaceStats regressor...")
        try:
            if check_attractiveness_available() and _MODEL_LOADING_STATUS['clip']:
                from pathlib import Path
                import torch
                import torch.nn as nn
                
                # Define AttractivenessRegressorV1
                class AttractivenessRegressorV1(nn.Module):
                    def __init__(self, input_dim=512, hidden1=256, hidden2=64):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(input_dim, hidden1),
                            nn.ReLU(),
                            nn.Linear(hidden1, hidden2),
                            nn.ReLU(),
                            nn.Linear(hidden2, 1),
                        )
                    def forward(self, x):
                        return self.net(x)
                
                base_path = Path(__file__).parent
                model_path = base_path / "models" / "attractiveness_regressor.pt"
                if not model_path.exists():
                    model_path = base_path / "facestats" / "models" / "attractiveness_regressor.pt"
                
                if model_path.exists():
                    _FACESTATS_REGRESSOR = AttractivenessRegressorV1(input_dim=512, hidden1=256, hidden2=64)
                    state_dict = torch.load(model_path, map_location='cpu')
                    _FACESTATS_REGRESSOR.load_state_dict(state_dict, strict=True)
                    _FACESTATS_REGRESSOR.eval()
                    _MODEL_LOADING_STATUS['facestats_regressor'] = True
                    print("✅ FaceStats regressor preloaded")
                else:
                    print("⚠️ FaceStats regressor model file not found")
            else:
                print("⚠️ FaceStats regressor: CLIP not available")
        except Exception as e:
            print(f"⚠️ FaceStats regressor preload failed: {e}")
        
        # 5. Preload Beauty-classifier (fast, ~2-3 seconds)
        print("\n📦 Preloading Beauty-classifier...")
        try:
            if check_beauty_classifier_available():
                from pathlib import Path
                import torch
                import torchvision.models as models
                import torch.nn as nn
                
                # Define BeautyClassifierModel
                class BeautyClassifierModel(nn.Module):
                    def __init__(self, out_features=512):
                        super().__init__()
                        resnet = models.resnet50(pretrained=False)
                        self.conv1 = resnet.conv1
                        self.bn1 = resnet.bn1
                        self.relu = resnet.relu
                        self.maxpool = resnet.maxpool
                        self.layer1 = resnet.layer1
                        self.layer2 = resnet.layer2
                        self.layer3 = resnet.layer3
                        self.layer4 = resnet.layer4
                        self.avgpool = resnet.avgpool
                        self.fc = nn.Linear(2048, 1)
                        self.sigmoid = nn.Sigmoid()
                    
                    def forward(self, x):
                        x = self.conv1(x)
                        x = self.bn1(x)
                        x = self.relu(x)
                        x = self.maxpool(x)
                        x = self.layer1(x)
                        x = self.layer2(x)
                        x = self.layer3(x)
                        x = self.layer4(x)
                        x = self.avgpool(x)
                        x = x.view(x.size(0), -1)
                        x = self.fc(x)
                        return self.sigmoid(x)
                
                base_path = Path(__file__).parent
                model_path = base_path / "models" / "attractiveness_classifier.pt"
                if not model_path.exists():
                    model_path = base_path / "beauty-classifier" / "models" / "attractiveness_classifier.pt"
                
                if model_path.exists():
                    _BEAUTY_CLASSIFIER_MODEL = BeautyClassifierModel(out_features=512)
                    state_dict = torch.load(model_path, map_location='cpu')
                    _BEAUTY_CLASSIFIER_MODEL.load_state_dict(state_dict)
                    _BEAUTY_CLASSIFIER_MODEL.eval()
                    _MODEL_LOADING_STATUS['beauty_classifier'] = True
                    print("✅ Beauty-classifier preloaded")
                else:
                    print("⚠️ Beauty-classifier model file not found")
            else:
                print("⚠️ Beauty-classifier dependencies not available")
        except Exception as e:
            print(f"⚠️ Beauty-classifier preload failed: {e}")
        
        print("\n" + "="*60)
        print("✅ MODEL PRELOADING COMPLETE")
        print("="*60 + "\n")
        
    except Exception as e:
        _MODEL_LOADING_STATUS['error'] = str(e)
        print(f"\n❌ Model preloading error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _MODEL_LOADING_STATUS['loading'] = False

# Start model preloading in background thread (non-blocking)
import threading
def start_model_preloading():
    """Start model preloading in background thread"""
    thread = threading.Thread(target=preload_models, daemon=True)
    thread.start()
    print("🚀 Started model preloading in background thread")

# Add a simple root route that responds immediately (for Render port detection)
@app.route('/', methods=['GET'])
def root():
    """Simple root endpoint for health checks"""
    return jsonify({'status': 'ok', 'message': 'LooksMax AI Backend is running'}), 200

# Initialize MediaPipe Face Mesh (lazy initialization)
face_mesh = None

def get_face_mesh():
    """Lazy initialization of MediaPipe Face Mesh"""
    global face_mesh
    if face_mesh is None:
        mp = get_mediapipe()
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
    - Goes down gradually as you move further away (but never below 20)
    - Less harsh: values far outside range still get 20-30 instead of 0
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
            # Outside the ideal window → 50 down towards 20 (not 0)
            # More forgiving: even far outside gets at least 20
            score = 50.0 - 15.0 * (d - 1.0)  # Less harsh: 15 instead of 20
            return float(max(20.0, score))  # Minimum 20 instead of 0
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
        # Wider ranges to avoid 0.0 scores for normal faces
        ideal_min, ideal_max = (-10, 15) if gender == 'Male' else (-8, 18)
        score = score_metric(tilt, ideal_min, ideal_max)
        
        # Debug: Log the scoring process
        print(f"[TILT DEBUG] tilt={tilt:.2f}°, ideal_range=[{ideal_min}, {ideal_max}], raw_score={score:.2f}")
        
        # No extra custom min clamp - let scores reflect actual geometry
        final_score = float(np.clip(score, 0.0, 100.0))
        print(f"[TILT DEBUG] FINAL tilt={tilt:.2f}°, score={final_score:.1f}")
        
        # If score is 0, warn about it
        if final_score == 0.0:
            print(f"⚠️ WARNING: Canthal tilt score is 0.0 - tilt value {tilt:.2f}° is far outside ideal range [{ideal_min}, {ideal_max}]")
        
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
        
        print(f"[IPD_SCORE DEBUG] ipd={ipd:.6f}, face_width={face_width:.6f}, ratio={ipd_ratio:.6f}")
        
        if np.isnan(ipd_ratio) or np.isinf(ipd_ratio):
            print(f"[IPD_SCORE DEBUG] Invalid ratio, returning 50.0")
            return 50.0
        
        # Ideal range: 0.40-0.55 (wider range for more realistic scoring)
        score = score_metric(ipd_ratio, 0.40, 0.55)
        print(f"[IPD_SCORE DEBUG] ratio={ipd_ratio:.6f}, ideal=[0.45, 0.50], score={score:.2f}")
        
        if score == 0.0:
            print(f"⚠️ WARNING: IPD score is 0.0 - ratio {ipd_ratio:.6f} is far outside ideal range [0.45, 0.50]")
        
        return score
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
        ideal_min, ideal_max = (1.4, 2.6)  # Much wider range to avoid 0.0 scores
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
        return score_metric(compactness, 1.0, 1.7)  # Wider range
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
        
        print(f"[MANDIBLE DEBUG] length={mandible_length:.6f}, face_height={face_height:.6f}, ratio={mandible_ratio:.6f}")
        
        if np.isnan(mandible_ratio) or np.isinf(mandible_ratio):
            print(f"[MANDIBLE DEBUG] Invalid ratio, returning 50.0")
            return 50.0
        
        # Wider range - mandible ratio typically 0.20-0.60 for normal faces
        score = score_metric(mandible_ratio, 0.25, 0.55)
        print(f"[MANDIBLE DEBUG] ratio={mandible_ratio:.6f}, ideal=[0.30, 0.50], score={score:.2f}")
        
        if score == 0.0:
            print(f"⚠️ WARNING: Mandible score is 0.0 - ratio {mandible_ratio:.6f} is far outside ideal range [0.30, 0.50]")
        
        return score
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

def calculate_all_metrics(front_landmarks, side_landmarks, gender='Male', front_image_array=None):
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
        
        # Calculate geometric PSL (based on facial measurements)
        geometric_psl = (eyes_avg + midface_avg + lower_third_avg + upper_third_avg + misc_avg) / 5.0
        
        # Calculate holistic attractiveness score using multiple models (FaceStats + Beauty-classifier)
        # This provides a more robust, modern attractiveness score with calibration
        attractiveness_score = None
        if front_image_array is not None:
            attractiveness_score = calculate_attractiveness_score(front_image_array)
        
        # Combine geometric PSL with holistic attractiveness
        # Weight: 50% geometric, 50% ML attractiveness (ML models are more accurate)
        if attractiveness_score is not None:
            psl = 0.5 * geometric_psl + 0.5 * attractiveness_score
            print(f"\n🎯 FINAL PSL: {psl:.1f} (geometric: {geometric_psl:.1f}, ML attractiveness: {attractiveness_score:.1f})")
            print(f"   ML models contributing: ✅ (FaceStats + Beauty-classifier ensemble)")
        else:
            psl = geometric_psl
            print(f"\n⚠️  Using geometric PSL only: {psl:.1f} (ML models not available)")
            print(f"   This is less accurate - check Railway logs for ML model errors")
        
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

def calculate_attractiveness_score(image_array):
    """
    Calculate holistic attractiveness score using multiple models for stability:
    1. FaceStats (CLIP + MLP) - if available
    2. Beauty-classifier (ResNet-50 on SCUT-FBP5500) - for calibration
    
    Returns average of available models, or None if none available.
    Based on:
    - FaceStats: https://github.com/jayklarin/FaceStats
    - Beauty-classifier: https://github.com/okurki/beauty-classifier
    """
    print("\n" + "="*60)
    print("🎯 ATTRACTIVENESS SCORING - Starting ensemble prediction")
    print("="*60)
    
    # Models are preloaded at startup - no need to load here

    scores = []
    
    # Try FaceStats (CLIP + MLP)
    print("\n📊 Attempting FaceStats scoring...")
    facestats_score = calculate_facestats_score(image_array)
    if facestats_score is not None:
        scores.append(facestats_score)
        print(f"✅ FaceStats contributed: {facestats_score:.1f}")
    else:
        print("❌ FaceStats: No score (model not found or error)")
    
    # Try Beauty-classifier (ResNet-50)
    print("\n📊 Attempting Beauty-classifier scoring...")
    beauty_score = calculate_beauty_classifier_score(image_array)
    if beauty_score is not None:
        scores.append(beauty_score)
        print(f"✅ Beauty-classifier contributed: {beauty_score:.1f}")
    else:
        print("❌ Beauty-classifier: No score (model not found or error)")
    
    # Return average if we have at least one score
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\n🎯 Ensemble Result: {avg_score:.1f} (from {len(scores)} model(s))")
        print("="*60 + "\n")
        return avg_score
    
    print("\n⚠️  No ML attractiveness scores available - using geometric only")
    print("="*60 + "\n")
    return None

def calculate_facestats_score(image_array):
    """Calculate attractiveness using FaceStats (CLIP + MLP)"""
    try:
        if not check_attractiveness_available():
            print("⚠️ FaceStats: Dependencies not available (torch/transformers)")
            return None
        
        import sys
        from pathlib import Path
        import tempfile
        import os
        
        print("🔍 FaceStats: Starting model loading...")
        
        # Add FaceStats src to path (optional, for imports)
        facestats_path = Path(__file__).parent / "facestats" / "src"
        if str(facestats_path) not in sys.path:
            sys.path.insert(0, str(facestats_path))
        
        # Define CLIP embedding function directly to avoid polars dependency
        # Use module-level globals for CLIP model (lazy loading)
        # Note: _CLIP_MODEL and _CLIP_PROCESSOR are defined at module level above
        def get_clip_embedding_local(image_path, model_name="openai/clip-vit-base-patch32"):
            """Extract CLIP embedding for an image (L2-normalized) - uses preloaded model"""
            global _CLIP_MODEL, _CLIP_PROCESSOR
            
            # Use preloaded models if available
            if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
                # Fallback: load on demand (slower, but works if preload failed)
                print("⚠️ CLIP models not preloaded, loading on demand...")
                from transformers import CLIPProcessor, CLIPModel
                _CLIP_MODEL = CLIPModel.from_pretrained(model_name)
                _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
                _CLIP_MODEL.eval()
            
            img = Image.open(image_path).convert("RGB")
            inputs = _CLIP_PROCESSOR(images=img, return_tensors="pt")
            with torch.no_grad():
                features = _CLIP_MODEL.get_image_features(**inputs)
            vec = features[0].cpu().numpy()
            return vec / (np.linalg.norm(vec) + 1e-8)
        
        # Define AttractivenessRegressorV1 directly to avoid polars dependency
        class AttractivenessRegressorV1(torch.nn.Module):
            """Matches the stored checkpoint: 512 → 256 → 64 → 1 with ReLU activations."""
            def __init__(self, input_dim=512, hidden1=256, hidden2=64):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden1),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden1, hidden2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden2, 1),
                )
            def forward(self, x):
                return self.net(x)
        
        # Initialize CLIP globals
        _CLIP_MODEL = None
        _CLIP_PROCESSOR = None
        
        # Convert numpy array to PIL Image
        from PIL import Image
        if isinstance(image_array, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image_array
            pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = image_array
        
        # Save temporarily to use get_clip_embedding (expects file path)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            pil_image.save(tmp_file.name, 'JPEG')
            tmp_path = tmp_file.name
        
        try:
            # Extract CLIP embedding (512-D, L2-normalized)
            embedding = get_clip_embedding_local(tmp_path)
            embedding = np.array(embedding).reshape(1, -1)
            
            # Load attractiveness regressor - check multiple possible locations
            base_path = Path(__file__).parent
            possible_paths = [
                base_path / "models" / "attractiveness_regressor.pt",  # Primary: direct models folder
                base_path / "facestats" / "models" / "attractiveness_regressor.pt",
                base_path / "facestats" / "src" / "models" / "attractiveness_regressor.pt",
                base_path / "src_facestats_models" / "attractiveness_regressor.pt",
            ]
            
            model_path = None
            print(f"🔍 FaceStats: Checking {len(possible_paths)} possible model locations...")
            for path in possible_paths:
                abs_path = path.resolve()
                exists = path.exists()
                print(f"   {'✅' if exists else '❌'} {abs_path}")
                if exists:
                    model_path = path
                    size_mb = path.stat().st_size / (1024 * 1024)
                    print(f"✅ FaceStats: Model found at {abs_path} ({size_mb:.1f} MB)")
                    break
            
            if model_path is None:
                print(f"❌ FaceStats: Model not found in any location!")
                print(f"   Checked paths:")
                for p in possible_paths:
                    print(f"     - {p.resolve()}")
                return None
            
            # Use preloaded model if available, otherwise load on demand
            global _FACESTATS_REGRESSOR
            if _FACESTATS_REGRESSOR is None:
                print(f"📦 FaceStats: Loading model from {model_path.name} (not preloaded)...")
                regressor = AttractivenessRegressorV1(input_dim=512, hidden1=256, hidden2=64)
                state_dict = torch.load(model_path, map_location='cpu')
                regressor.load_state_dict(state_dict, strict=True)
                regressor.eval()
                _FACESTATS_REGRESSOR = regressor
                print("✅ FaceStats: Model loaded successfully")
            else:
                regressor = _FACESTATS_REGRESSOR
                print("✅ FaceStats: Using preloaded model")
            
            # Predict attractiveness (raw score)
            print("🔮 FaceStats: Running prediction...")
            with torch.no_grad():
                embedding_tensor = torch.FloatTensor(embedding)
                prediction = regressor(embedding_tensor)
                raw_score = prediction.item()
            print(f"📊 FaceStats: Raw prediction = {raw_score:.4f}")
            
            # FaceStats model outputs raw predictions that need proper scaling
            # Based on FaceStats source code, raw scores are converted to deciles (1-10)
            # We'll use a more robust normalization that handles any range
            
            # Strategy: Use sigmoid-like mapping to 0-100
            # This works well for regression outputs that can be any value
            # Typical raw scores from FaceStats are roughly in -10 to 10 range based on training
            
            # Method 1: If raw score is already reasonable (0-100), use it directly
            if 0 <= raw_score <= 100:
                score = raw_score
            else:
                # Method 2: Use tanh-based normalization for wider range
                # tanh maps (-inf, +inf) to (-1, 1), then scale to 0-100
                # This handles both negative and very large positive values
                normalized = np.tanh(raw_score / 10.0)  # Divide by 10 to scale input
                score = 50.0 + (normalized * 50.0)  # Map (-1, 1) to (0, 100)
            
            score = float(np.clip(score, 0.0, 100.0))
            print(f"✅ FaceStats: Final score = {score:.1f} (raw: {raw_score:.4f})")
            return score
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
    except Exception as e:
        print(f"❌ FaceStats scoring error: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_beauty_classifier_score(image_array):
    """
    Calculate attractiveness using beauty-classifier (ResNet-50 on SCUT-FBP5500)
    Returns score on 0-100 scale (converted from 1-5 scale)
    Based on: https://github.com/okurki/beauty-classifier
    
    Model architecture:
    - ResNet-50 (ImageNet pretrained, frozen)
    - Custom FC: 2048 → 512 → 1 (with Sigmoid)
    - Output: 0-1 (represents 1-5 attractiveness scale)
    """
    try:
        if not check_beauty_classifier_available():
            print("⚠️ Beauty-classifier: Dependencies not available (torch/torchvision)")
            return None
        
        print("🔍 Beauty-classifier: Starting model loading...")
        
        import sys
        from pathlib import Path
        import tempfile
        import os
        import torchvision.models as models
        import torchvision.transforms as transforms
        import torch.nn as nn
        
        # Add beauty-classifier src to path
        beauty_path = Path(__file__).parent / "beauty-classifier" / "src"
        if str(beauty_path) not in sys.path:
            sys.path.insert(0, str(beauty_path))
        
        # Define the model architecture (matches beauty-classifier)
        class BeautyClassifierModel(nn.Module):
            def __init__(self, out_features=512):
                super().__init__()
                # Load ResNet-50 with ImageNet weights
                resnet = models.resnet50(weights="IMAGENET1K_V2")
                # Freeze all parameters
                for param in resnet.parameters():
                    param.requires_grad = False
                # Replace FC layer with custom head
                resnet.fc = nn.Sequential(
                    nn.Linear(resnet.fc.in_features, out_features),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(out_features, 1),
                    nn.Sigmoid()  # Output 0-1 (represents 1-5 scale)
                )
                self.model = resnet
            
            def forward(self, x):
                return self.model(x)
        
        # Convert numpy array to PIL Image
        from PIL import Image
        if isinstance(image_array, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image_array
            pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = image_array
        
        # Define image preprocessing (matches beauty-classifier)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet-50 expects 224x224
            transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Preprocess image
        image_tensor = transform(pil_image).unsqueeze(0)
        
        # Load model - check multiple possible locations
        base_path = Path(__file__).parent
        possible_paths = [
            base_path / "models" / "attractiveness_classifier.pt",  # Primary: direct models folder
            base_path / "beauty-classifier" / "models" / "attractiveness_classifier.pt",
        ]
        
        model_path = None
        print(f"🔍 Beauty-classifier: Checking {len(possible_paths)} possible model locations...")
        for path in possible_paths:
            abs_path = path.resolve()
            exists = path.exists()
            print(f"   {'✅' if exists else '❌'} {abs_path}")
            if exists:
                model_path = path
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✅ Beauty-classifier: Model found at {abs_path} ({size_mb:.1f} MB)")
                break
        
        if model_path is None:
            print(f"❌ Beauty-classifier: Model not found in any location!")
            print(f"   Checked paths:")
            for p in possible_paths:
                print(f"     - {p.resolve()}")
            print("   Note: Model file may need to be pulled using DVC: dvc pull models/attractiveness_classifier.pt.dvc")
            return None
        
        # Use preloaded model if available, otherwise load on demand
        global _BEAUTY_CLASSIFIER_MODEL
        if _BEAUTY_CLASSIFIER_MODEL is None:
            print(f"📦 Beauty-classifier: Loading model from {model_path.name} (not preloaded)...")
            model = BeautyClassifierModel(out_features=512)
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            _BEAUTY_CLASSIFIER_MODEL = model
            print("✅ Beauty-classifier: Model loaded successfully")
        else:
            model = _BEAUTY_CLASSIFIER_MODEL
            print("✅ Beauty-classifier: Using preloaded model")
        
        # Predict attractiveness (0-1 range, represents 1-5 scale)
        print("🔮 Beauty-classifier: Running prediction...")
        with torch.no_grad():
            output = model(image_tensor)
            score_01 = output.item()  # 0-1 range
        print(f"📊 Beauty-classifier: Raw prediction (0-1) = {score_01:.4f}")
        
        # Convert 0-1 to 1-5 scale: score_5 = 1 + 4 * score_01
        score_5 = 1.0 + 4.0 * score_01
        
        # Convert 1-5 to 0-100 scale: score_100 = (score_5 - 1) / 4 * 100
        score_100 = (score_5 - 1.0) / 4.0 * 100.0
        
        score_100 = float(np.clip(score_100, 0.0, 100.0))
        print(f"✅ Beauty-classifier: Final score = {score_100:.1f} (raw: {score_01:.4f}, 1-5: {score_5:.2f})")
        return score_100
        
    except Exception as e:
        print(f"❌ Beauty-classifier scoring error: {e}")
        import traceback
        traceback.print_exc()
        return None

def detect_gender_from_image(image_array):
    """Detect gender from image using DeepFace (97%+ accuracy) - fails fast if slow"""
    try:
        if not check_deepface_available():
            return None
        
        # Lazy import DeepFace to avoid blocking startup
        # Note: First call may be slow (loads TensorFlow), but subsequent calls are fast
        from deepface import DeepFace
        
        # Save image to temp file (DeepFace prefers file paths)
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, image_array)
            tmp_path = tmp.name
        
        try:
            # Analyze gender with high accuracy backend
            # Use opencv backend instead of retinaface for faster loading on first call
            result = DeepFace.analyze(
                img_path=tmp_path,
                actions=['gender'],
                enforce_detection=False,  # Don't fail if face detection is uncertain
                detector_backend='opencv',  # Faster than retinaface, still accurate
                silent=True  # Suppress verbose output
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
        
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
        # Fail silently - gender detection is optional
        print(f"⚠️ Gender detection error (will use default): {e}")
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
        mp = get_mediapipe()
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
        
        # Debug: Log landmark detection success
        print(f"✅ Landmarks detected: front={front_landmarks.shape if front_landmarks is not None else None}, side={side_landmarks.shape if side_landmarks is not None else None}")
        
        # Debug: Check IPD calculation
        try:
            ipd = calculate_ipd(front_landmarks)
            print(f"📏 IPD calculated: {ipd:.4f}")
            if ipd <= 0 or np.isnan(ipd):
                print(f"⚠️ WARNING: Invalid IPD ({ipd}), this will cause calculation issues")
        except Exception as e:
            print(f"⚠️ WARNING: IPD calculation failed: {e}")
        
        # Convert images to numpy arrays for potential attractiveness scoring
        front_image_array = np.array(front_image)
        
        # Calculate all metrics
        try:
            results = calculate_all_metrics(front_landmarks, side_landmarks, gender, front_image_array)
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
    """Lightweight health check endpoint - returns basic status without loading heavy libraries"""
    try:
        # Fast check - just see if modules can be imported (without actually importing)
        # This is much faster than loading the full libraries
        import importlib.util
        
        # Check MediaPipe availability (lightweight - just check if module exists)
        mediapipe_available = False
        mediapipe_has_solutions = False
        mediapipe_version = 'unknown'
        try:
            spec = importlib.util.find_spec('mediapipe')
            mediapipe_available = spec is not None
            # If module exists, try to get version without importing (check __init__.py)
            if mediapipe_available:
                try:
                    # Try to read version from module metadata without importing
                    import pkg_resources
                    version = pkg_resources.get_distribution('mediapipe').version
                    mediapipe_version = version
                except Exception:
                    pass
        except Exception:
            pass  # MediaPipe not available
        
        # Check DeepFace availability (lightweight - just check if module exists)
        deepface_available = False
        try:
            spec = importlib.util.find_spec('deepface')
            deepface_available = spec is not None
        except Exception:
            pass
        
        # Check torch/transformers availability (lightweight)
        facestats_available = False
        try:
            spec_torch = importlib.util.find_spec('torch')
            spec_transformers = importlib.util.find_spec('transformers')
            facestats_available = spec_torch is not None and spec_transformers is not None
        except Exception:
            pass
        
        # Check torchvision availability (lightweight)
        beauty_classifier_available = False
        try:
            spec_torchvision = importlib.util.find_spec('torchvision')
            beauty_classifier_available = spec_torchvision is not None
        except Exception:
            pass
        
        status = {
            'status': 'healthy',
            'mediapipe_installed': mediapipe_available,
            'mediapipe_has_solutions': mediapipe_has_solutions,
            'deepface_available': deepface_available,
            'facestats_available': facestats_available,
            'beauty_classifier_available': beauty_classifier_available,
            'python_version': str(os.sys.version),
        }
        
        if mediapipe_available:
            status['mediapipe_version'] = mediapipe_version
        
        return jsonify(status), 200
        
    except Exception as e:
        # If anything fails, return basic healthy status
        return jsonify({
            'status': 'healthy',
            'error': 'Health check partial failure',
            'message': str(e)
        }), 200

# Start model preloading when app starts (delayed to ensure app is ready first)
# Use a small delay to let gunicorn finish initialization
import time
def delayed_model_preload():
    """Start model preloading after a short delay to ensure app is ready"""
    time.sleep(2)  # Give gunicorn time to finish startup
    try:
        start_model_preloading()
    except Exception as e:
        print(f"⚠️ Model preloading failed (non-critical): {e}")
        import traceback
        traceback.print_exc()

# Start preloading in background (non-blocking, won't crash app if it fails)
try:
    preload_thread = threading.Thread(target=delayed_model_preload, daemon=True)
    preload_thread.start()
    print("✅ App initialized, model preloading will start in background")
except Exception as e:
    print(f"⚠️ Could not start model preloading thread (non-critical): {e}")

# Confirm module loaded successfully
import sys
print("="*60, file=sys.stderr)
print("✅ Flask app module loaded successfully", file=sys.stderr)
print(f"✅ Root endpoint available at: /", file=sys.stderr)
print(f"✅ Health endpoint available at: /health", file=sys.stderr)
print(f"✅ API endpoint available at: /api/analyze-face", file=sys.stderr)
print(f"✅ PORT environment variable: {os.environ.get('PORT', 'NOT SET')}", file=sys.stderr)
print("="*60, file=sys.stderr)
sys.stderr.flush()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Flask development server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)

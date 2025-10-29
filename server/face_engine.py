import os
import numpy as np
import cv2
from typing import Optional, List

# Try InsightFace first
INSIGHT_AVAILABLE = True
_insight_model = None
try:
    print('[FACE_ENGINE] Attempting to import InsightFace...')
    from insightface.app import FaceAnalysis
    print('[FACE_ENGINE] InsightFace.app imported successfully')
    import insightface
    print('[FACE_ENGINE] insightface module imported')
    # onnxruntime is used under the hood; ensure import doesn't fail
    import onnxruntime  # noqa: F401
    print('[FACE_ENGINE] onnxruntime imported successfully')
    print('[FACE_ENGINE] All InsightFace dependencies available')
except Exception as e:
    print(f'[FACE_ENGINE] InsightFace import failed: {e}')
    import traceback
    traceback.print_exc()
    INSIGHT_AVAILABLE = False
    FaceAnalysis = None  # type: ignore

print(f'[FACE_ENGINE] INSIGHT_AVAILABLE = {INSIGHT_AVAILABLE}')

# Fallbacks will be handled by the caller (e.g., DeepFace usage in app.py)

class FaceEngine:
    """
    Singleton-like face engine that lazily initializes InsightFace and caches embeddings.
    """
    def __init__(self):
        self.initialized = False
        self.det_name = os.getenv('INSIGHTFACE_DETECTOR', 'retinaface')
        self.rec_name = os.getenv('INSIGHTFACE_RECOGNIZER', 'buffalo_l')
        self.providers = os.getenv('INSIGHTFACE_PROVIDERS', 'CPUExecutionProvider').split(',')
        self.app = None  # Will be FaceAnalysis instance after init()
        # simple in-memory cache of user embeddings: user_id -> np.ndarray
        self.user_cache = {}

    def init(self):
        if self.initialized:
            return
        if not INSIGHT_AVAILABLE:
            raise RuntimeError('InsightFace not available')
        # Initialize FaceAnalysis with detection + recognition models
        self.app = FaceAnalysis(name=self.rec_name, providers=self.providers)
        # Set a reasonable context; det-size controls speed/accuracy tradeoff
        det_size = os.getenv('INSIGHTFACE_DET_SIZE', '640x640')
        try:
            w, h = det_size.lower().split('x')
            det_size_tuple = (int(w), int(h))
        except Exception:
            det_size_tuple = (640, 640)
        self.app.prepare(ctx_id=0, det_size=det_size_tuple)
        self.initialized = True

    def embed(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a face embedding from a BGR image. Returns None if no face found.
        Always returns L2-normalized embedding for consistent cosine similarity.
        """
        if not self.initialized:
            self.init()
        assert self.app is not None
        # InsightFace expects BGR np.ndarray
        faces = self.app.get(image_bgr)
        if not faces:
            return None
        # Select the largest face by bbox area
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) if hasattr(f, 'bbox') else 0, reverse=True)
        face = faces[0]
        # Prefer normed_embedding if available; fallback to embedding and normalize manually
        emb = getattr(face, 'normed_embedding', None)
        if emb is None:
            emb = getattr(face, 'embedding', None)
            if emb is not None:
                # Manually L2-normalize to ensure consistency
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
        if emb is None:
            return None
        # Return a copy as float32 to avoid mutable references
        return np.array(emb, dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def verify(self, stored_emb: np.ndarray, image_bgr: np.ndarray, variants: Optional[List[np.ndarray]] = None) -> tuple[bool, float, List[float]]:
        """
        Compute similarity against stored embedding across optional variants and return (verified, max_sim, sims).
        Threshold is read from FACE_SIMILARITY_THRESHOLD or defaults to 0.56 for InsightFace.
        """
        if variants is None:
            variants = [image_bgr]
        sims: List[float] = []
        for img in variants:
            emb = self.embed(img)
            if emb is None:
                continue
            sim = self.cosine_similarity(stored_emb, emb)
            sims.append(sim)
        if not sims:
            return False, 0.0, []
        max_sim = max(sims)
        thr = float(os.getenv('FACE_SIMILARITY_THRESHOLD', '0.56'))
        return max_sim >= thr, max_sim, sims

    def verify_any(self, stored_list: List[np.ndarray], image_bgr: np.ndarray, variants: Optional[List[np.ndarray]] = None) -> tuple[bool, float, List[float]]:
        """Check against multiple stored embeddings and return best-of across all.
        Returns (verified, best_similarity, all_similarities)
        """
        all_sims: List[float] = []
        best = 0.0
        verified_any = False
        thr = float(os.getenv('FACE_SIMILARITY_THRESHOLD', '0.60'))
        if not stored_list:
            return False, 0.0, []
        for s in stored_list:
            ok, max_sim, sims = self.verify(s, image_bgr=image_bgr, variants=variants)
            all_sims.extend(sims)
            if max_sim > best:
                best = max_sim
            if ok:
                verified_any = True
        return verified_any, best, all_sims

# Global accessor
_engine: Optional[FaceEngine] = None

def get_engine() -> Optional[FaceEngine]:
    global _engine
    print('[FACE_ENGINE] get_engine() called')
    if _engine is not None:
        print('[FACE_ENGINE] Returning cached engine')
        return _engine
    if not INSIGHT_AVAILABLE:
        print('[FACE_ENGINE] InsightFace not available, returning None')
        return None
    print('[FACE_ENGINE] Creating new FaceEngine instance')
    _engine = FaceEngine()
    try:
        print('[FACE_ENGINE] Initializing engine...')
        _engine.init()
        print('[FACE_ENGINE] Engine initialized successfully')
    except Exception as e:
        # InsightFace failed to initialize; fallback will be used by callers
        print(f'[FACE_ENGINE] Engine initialization failed: {e}')
        import traceback
        traceback.print_exc()
        _engine = None
    return _engine

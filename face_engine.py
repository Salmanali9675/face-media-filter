"""
face_engine.py
Deep learning face detection and recognition engine using OpenCV YuNet and SFace.
Compatible with OpenCV 4.x/5.x and Python 3.10+.
"""

import os
import sys
import time
import urllib.request
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image, ImageOps

# Model URLs (OpenCV Zoo Official Git LFS links via GitHub Media CDN)
YUNET_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_YUNET_PATH = os.path.join(DEFAULT_MODELS_DIR, "face_detection_yunet.onnx")
DEFAULT_SFACE_PATH = os.path.join(DEFAULT_MODELS_DIR, "face_recognition_sface.onnx")

# Standard thresholds for SFace:
# Cosine similarity range: [-1.0, 1.0]. OpenCV recommends >= 0.363 for identity match.
# Typical balanced threshold: 0.38 - 0.42.
DEFAULT_COSINE_THRESHOLD = 0.40


def ensure_models_exist(models_dir: str = DEFAULT_MODELS_DIR, progress_callback=None) -> Tuple[str, str]:
    """Ensures YuNet and SFace ONNX model files are present, downloading them if necessary."""
    os.makedirs(models_dir, exist_ok=True)
    yunet_path = os.path.join(models_dir, "face_detection_yunet.onnx")
    sface_path = os.path.join(models_dir, "face_recognition_sface.onnx")

    def _download(url: str, dest: str, name: str):
        if os.path.exists(dest) and os.path.getsize(dest) > 10000:
            return
        if progress_callback:
            progress_callback(f"Downloading {name} model...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total > 0:
                    pct = int((downloaded / total) * 100)
                    progress_callback(f"Downloading {name}: {pct}%")

    _download(YUNET_URL, yunet_path, "YuNet Face Detector")
    _download(SFACE_URL, sface_path, "SFace Face Recognizer")
    return yunet_path, sface_path


def load_image_robust(file_path: str) -> Optional[np.ndarray]:
    """
    Safely loads an image handling Unicode paths and EXIF rotation tags.
    Returns a BGR numpy array suitable for OpenCV, or None if reading failed.
    """
    if not os.path.exists(file_path):
        return None
    try:
        # Use PIL to read and automatically correct orientation via EXIF
        with Image.open(file_path) as pil_img:
            transposed = ImageOps.exif_transpose(pil_img)
            # Ensure RGB
            if transposed.mode != "RGB":
                transposed = transposed.convert("RGB")
            rgb_arr = np.array(transposed)
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            return bgr_arr
    except Exception:
        # Fallback to numpy fromfile + imdecode (safe for Windows Unicode paths)
        try:
            raw_bytes = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None


class FaceEngine:
    """
    High-performance face detection and feature recognition engine
    built on OpenCV DNN (YuNet detector + SFace recognizer).
    """

    def __init__(self,
                 yunet_path: Optional[str] = None,
                 sface_path: Optional[str] = None,
                 score_threshold: float = 0.6,
                 nms_threshold: float = 0.3):
        if not yunet_path or not sface_path or not os.path.exists(yunet_path) or not os.path.exists(sface_path):
            yunet_path, sface_path = ensure_models_exist()

        self.yunet_path = yunet_path
        self.sface_path = sface_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        # Initialize SFace recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.sface_path,
            config=""
        )

        # Initialize YuNet detector with a standard default input size
        self.current_input_size = (320, 320)
        self.detector = cv2.FaceDetectorYN.create(
            model=self.yunet_path,
            config="",
            input_size=self.current_input_size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000
        )

    def _ensure_detector_size(self, width: int, height: int):
        """Sets detector input size to match current image dimensions."""
        if self.current_input_size != (width, height):
            self.detector.setInputSize((int(width), int(height)))
            self.current_input_size = (int(width), int(height))

    def detect_faces(self, image_bgr: np.ndarray, max_dimension: int = 1280) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """
        Detects faces in an image. Automatically scales down huge images for speed
        while preserving coordinate accuracy.
        Returns: (processed_image_bgr, faces_array_or_None, scale_factor)
        """
        h, w = image_bgr.shape[:2]
        scale = 1.0
        if max(h, w) > max_dimension:
            scale = max_dimension / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = image_bgr

        rh, rw = resized.shape[:2]
        self._ensure_detector_size(rw, rh)
        _, faces = self.detector.detect(resized)
        return resized, faces, scale

    def extract_face_embeddings(self, image_bgr: np.ndarray, faces: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Extracts 128-dimensional embeddings for all detected faces in the image.
        Returns a list of tuples: (face_box_landmarks, embedding_vector)
        """
        if faces is None:
            resized, faces, _ = self.detect_faces(image_bgr)
            target_img = resized
        else:
            target_img = image_bgr

        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            try:
                aligned = self.recognizer.alignCrop(target_img, face)
                feature = self.recognizer.feature(aligned)
                results.append((face, feature))
            except Exception:
                continue
        return results

    def extract_target_profile(self, target_image_path: str) -> Dict[str, Any]:
        """
        Extracts the target face from the user-provided reference image.
        Returns a dictionary containing:
          - 'embedding': np.ndarray
          - 'face_count': int
          - 'face_crop': np.ndarray (RGB for GUI preview)
          - 'image_shape': (w, h)
          - 'error': Optional[str]
        """
        img = load_image_robust(target_image_path)
        if img is None:
            return {"error": f"Could not load image: {os.path.basename(target_image_path)}"}

        resized, faces, _ = self.detect_faces(img, max_dimension=1600)
        if faces is None or len(faces) == 0:
            return {"error": "No human face was detected in the reference picture. Please choose a clearer photo."}

        # Select the primary face (if multiple, pick the largest bounding box)
        best_face = None
        max_area = -1
        for face in faces:
            area = face[2] * face[3]
            if area > max_area:
                max_area = area
                best_face = face

        # Align and extract embedding
        aligned = self.recognizer.alignCrop(resized, best_face)
        feature = self.recognizer.feature(aligned)

        # Crop a preview thumbnail of the face
        x, y, w, h = [int(v) for v in best_face[:4]]
        # Add a 15% margin for visual comfort
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        img_h, img_w = resized.shape[:2]
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        crop_bgr = resized[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB) if crop_bgr.size > 0 else None

        return {
            "embedding": feature,
            "face_count": len(faces),
            "face_crop": crop_rgb,
            "box": (x, y, w, h),
            "image_shape": (img_w, img_h),
            "error": None
        }

    def compute_similarity(self, target_embedding: np.ndarray, candidate_embedding: np.ndarray) -> float:
        """
        Calculates cosine similarity between two face feature vectors.
        Returns a float between -1.0 and 1.0 (typically 0.0 to 1.0).
        """
        return float(self.recognizer.match(target_embedding, candidate_embedding, cv2.FaceRecognizerSF_FR_COSINE))

    def scan_image(self,
                   image_path: str,
                   target_embedding: np.ndarray,
                   threshold: float = DEFAULT_COSINE_THRESHOLD) -> Tuple[bool, float, str]:
        """
        Scans a single photo for the target face.
        Returns: (has_match, max_similarity, details_str)
        """
        img = load_image_robust(image_path)
        if img is None:
            return False, 0.0, "Unable to read image file"

        resized, faces, _ = self.detect_faces(img, max_dimension=1280)
        if faces is None or len(faces) == 0:
            return False, 0.0, "No faces detected in image"

        max_score = -1.0
        match_found = False

        for face in faces:
            try:
                aligned = self.recognizer.alignCrop(resized, face)
                feature = self.recognizer.feature(aligned)
                score = self.compute_similarity(target_embedding, feature)
                if score > max_score:
                    max_score = score
                if score >= threshold:
                    match_found = True
            except Exception:
                continue

        if match_found:
            details = f"Target person found (Confidence: {max_score:.2f}, Faces in photo: {len(faces)})"
        else:
            details = f"No match (Best similarity: {max(0.0, max_score):.2f}, Faces in photo: {len(faces)})"

        return match_found, max(0.0, max_score), details

    def scan_video(self,
                   video_path: str,
                   target_embedding: np.ndarray,
                   threshold: float = DEFAULT_COSINE_THRESHOLD,
                   sample_interval_sec: float = 1.0,
                   stop_on_first_match: bool = True,
                   progress_callback=None) -> Tuple[bool, float, str]:
        """
        Scans a video file by sampling frames at regular time intervals.
        Stops immediately on the first match if stop_on_first_match is True.
        Returns: (has_match, max_similarity, details_str)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0.0, "Could not open video file"

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0:
                fps = 30.0

            frame_step = max(1, int(round(fps * sample_interval_sec)))
            duration_sec = total_frames / fps if total_frames > 0 else 0

            max_score = -1.0
            match_found = False
            best_timestamp = 0.0
            checked_frames = 0
            current_frame_idx = 0

            while current_frame_idx < total_frames or total_frames <= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                checked_frames += 1
                current_time = current_frame_idx / fps

                if progress_callback:
                    # Provide frame-level progress
                    progress_callback(current_time, duration_sec)

                resized, faces, _ = self.detect_faces(frame, max_dimension=720)
                if faces is not None and len(faces) > 0:
                    for face in faces:
                        try:
                            aligned = self.recognizer.alignCrop(resized, face)
                            feature = self.recognizer.feature(aligned)
                            score = self.compute_similarity(target_embedding, feature)
                            if score > max_score:
                                max_score = score
                                best_timestamp = current_time
                            if score >= threshold:
                                match_found = True
                                if stop_on_first_match:
                                    break
                        except Exception:
                            continue

                if match_found and stop_on_first_match:
                    break

                current_frame_idx += frame_step

            mins, secs = divmod(int(best_timestamp), 60)
            time_str = f"{mins:02d}:{secs:02d}"

            if match_found:
                details = f"Target person found at {time_str} (Confidence: {max_score:.2f}, Checked {checked_frames} frames)"
            else:
                details = f"No match in video (Best similarity: {max(0.0, max_score):.2f}, Checked {checked_frames} frames)"

            return match_found, max(0.0, max_score), details

        finally:
            cap.release()

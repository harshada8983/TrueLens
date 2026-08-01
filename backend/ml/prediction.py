import os
import tempfile
import cv2
import numpy as np
import tensorflow as tf
import logging
from .utils import FaceExtractor

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepfakePredictor:
    def __init__(self):
        """
        Initializes the predictor, loads the trained model into memory, 
        and readies the MTCNN face extractor pipeline.
        """
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(self.base_dir, 'models', 'truelens_efficientnet.keras')
        
        self.logs = [] 
        self._log_event("Initializing TrueLens Predictor Engine...")
        
        self.extractor = FaceExtractor()
        self.model = self._load_model()

    def _log_event(self, message):
        logger.info(message)
        self.logs.append(message)

    def _load_model(self):
        if not os.path.exists(self.model_path):
            self._log_event(f"CRITICAL ERROR: Model weights not found at {self.model_path}")
            return None
        
        self._log_event("Loading EfficientNetB0 Deep Learning Model into memory...")
        return tf.keras.models.load_model(self.model_path)

    def _score_frame(self, image_path):
        """
        Runs a single image through the extraction + calibration + inference
        pipeline and returns the raw sigmoid score (float) or None if no
        face could be isolated. Shared by both image and video prediction.
        """
        # --- 1. HIGH-RES SPEED OPTIMIZATION ---
        try:
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                max_dim = 1000
                if max(h, w) > max_dim:
                    self._log_event("Optimizing high-resolution tensor matrix for rapid scanning...")
                    scale = max_dim / max(h, w)
                    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
                    cv2.imwrite(image_path, img_resized)
        except Exception as e:
            self._log_event(f"Warning: Resolution optimization skipped: {e}")

        # --- 2. CONTEXT-AWARE EXTRACTION ---
        face_bgr = self.extractor.extract_and_crop(image_path)
        if face_bgr is None:
            return None

        # --- 3. MATHEMATICAL CALIBRATION ---
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype('float32') / 255.0
        input_tensor = np.expand_dims(face_normalized, axis=0)

        # --- 4. NEURAL INFERENCE ---
        prediction = self.model.predict(input_tensor, verbose=0)
        return float(prediction[0][0])

    def _verdict_from_score(self, raw_score):
        if raw_score > 0.5:
            classification = "MANIPULATED"
            confidence = raw_score * 100
        else:
            classification = "REAL"
            confidence = (1.0 - raw_score) * 100

        # Professional Forensic Cap
        if confidence >= 99.9:
            confidence = 99.85

        return classification, round(confidence, 2)

    def predict(self, image_path):
        """
        The master inference pipeline: High-speed, context-aware, and mathematically calibrated.
        """
        self.logs = [] 
        self._log_event(f"Received target image: {os.path.basename(image_path)}")

        if self.model is None:
            return {"status": "error", "message": "Model not loaded.", "logs": self.logs}

        self._log_event("Passing image to MTCNN Extractor Pipeline...")
        raw_score = self._score_frame(image_path)

        if raw_score is None:
            self._log_event("FAIL: No human face detected in the image.")
            return {
                "status": "error", 
                "message": "No face detected. Please upload a clear photo of a person.",
                "logs": self.logs
            }

        self._log_event("Face successfully isolated and aligned. Preparing tensor matrix.")
        self._log_event("Injecting calibrated tensor into neural network...")
        self._log_event(f"Raw Neural Network Math Score: {raw_score:.4f}")

        classification, confidence = self._verdict_from_score(raw_score)
        self._log_event(f"Analysis complete. Verdict: {classification} ({confidence:.2f}% confidence)")

        return {
            "status": "success",
            "prediction": classification,
            "confidence_percentage": confidence,
            "raw_score": raw_score,
            "logs": self.logs
        }

    def predict_video(self, video_path, max_frames=12):
        """
        Samples up to `max_frames` frames evenly across the video, runs each
        through the same forensic pipeline used for images, and aggregates
        the per-frame verdicts into a single overall verdict.
        """
        self.logs = []
        self._log_event(f"Received target video: {os.path.basename(video_path)}")

        if self.model is None:
            return {"status": "error", "message": "Model not loaded.", "logs": self.logs}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log_event("FAIL: Video file could not be opened or is corrupted.")
            return {"status": "error", "message": "Could not read the video file.", "logs": self.logs}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0

        if total_frames <= 0:
            cap.release()
            self._log_event("FAIL: Video contains no readable frames.")
            return {"status": "error", "message": "Video contains no readable frames.", "logs": self.logs}

        self._log_event(f"Video contains {total_frames} frames at ~{fps:.1f} fps. Sampling up to {max_frames} frames.")

        # Evenly spaced frame indices across the whole clip (skip first/last 5% to avoid black frames)
        sample_count = min(max_frames, total_frames)
        start = int(total_frames * 0.02)
        end = int(total_frames * 0.98)
        if end <= start:
            start, end = 0, total_frames - 1
        frame_indices = sorted(set(
            int(start + i * (end - start) / max(1, sample_count - 1)) for i in range(sample_count)
        ))

        temp_dir = tempfile.gettempdir()
        frame_scores = []
        frame_results = []
        frames_skipped = 0

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success or frame is None:
                frames_skipped += 1
                continue

            frame_path = os.path.join(temp_dir, f"truelens_frame_{idx}_{os.getpid()}.jpg")
            try:
                cv2.imwrite(frame_path, frame)
                raw_score = self._score_frame(frame_path)
            finally:
                if os.path.exists(frame_path):
                    os.remove(frame_path)

            if raw_score is None:
                frames_skipped += 1
                self._log_event(f"Frame {idx}: no face detected, skipping.")
                continue

            classification, confidence = self._verdict_from_score(raw_score)
            frame_scores.append(raw_score)
            frame_results.append({"frame_index": idx, "prediction": classification, "confidence_percentage": confidence})
            self._log_event(f"Frame {idx}: {classification} ({confidence:.2f}% confidence)")

        cap.release()

        if not frame_scores:
            self._log_event("FAIL: No face detected in any sampled frame.")
            return {
                "status": "error",
                "message": "No face detected in any sampled frame. Try a video with a clearer view of the person.",
                "logs": self.logs
            }

        # --- AGGREGATE VERDICT ---
        avg_raw_score = sum(frame_scores) / len(frame_scores)
        manipulated_votes = sum(1 for s in frame_scores if s > 0.5)
        manipulated_ratio = manipulated_votes / len(frame_scores)

        overall_classification, overall_confidence = self._verdict_from_score(avg_raw_score)

        self._log_event(
            f"Aggregate complete: {len(frame_scores)} frames analyzed, {frames_skipped} skipped. "
            f"{manipulated_votes}/{len(frame_scores)} frames flagged MANIPULATED."
        )
        self._log_event(f"Overall Verdict: {overall_classification} ({overall_confidence:.2f}% confidence)")

        return {
            "status": "success",
            "prediction": overall_classification,
            "confidence_percentage": overall_confidence,
            "raw_score": avg_raw_score,
            "frames_analyzed": len(frame_scores),
            "frames_skipped": frames_skipped,
            "manipulated_frame_ratio": round(manipulated_ratio, 4),
            "frame_results": frame_results,
            "logs": self.logs
        }
import os
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

    def predict(self, image_path):
        """
        The master inference pipeline: High-speed, context-aware, and mathematically calibrated.
        """
        self.logs = [] 
        self._log_event(f"Received target image: {os.path.basename(image_path)}")

        if self.model is None:
            return {"status": "error", "message": "Model not loaded.", "logs": self.logs}

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
        self._log_event("Passing image to MTCNN Extractor Pipeline...")
        face_bgr = self.extractor.extract_and_crop(image_path)

        if face_bgr is None:
            self._log_event("FAIL: No human face detected in the image.")
            return {
                "status": "error", 
                "message": "No face detected. Please upload a clear photo of a person.",
                "logs": self.logs
            }

        self._log_event("Face successfully isolated and aligned. Preparing tensor matrix.")

        # --- 3. MATHEMATICAL CALIBRATION ---
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        
        # Divide by 255.0 to exactly match the training data format
        face_normalized = face_rgb.astype('float32') / 255.0
        input_tensor = np.expand_dims(face_normalized, axis=0) 

        # --- 4. NEURAL INFERENCE ---
        self._log_event("Injecting calibrated tensor into neural network...")
        prediction = self.model.predict(input_tensor, verbose=0)
        
        raw_score = float(prediction[0][0])
        self._log_event(f"Raw Neural Network Math Score: {raw_score:.4f}")
        
        # --- 5. PURE VERDICT AND CONFIDENCE LOGIC ---
        if raw_score > 0.5:
            classification = "MANIPULATED"
            confidence = raw_score * 100
        else:
            classification = "REAL"
            confidence = (1.0 - raw_score) * 100

        # Professional Forensic Cap
        if confidence >= 99.9:
            confidence = 99.85

        self._log_event(f"Analysis complete. Verdict: {classification} ({confidence:.2f}% confidence)")

        return {
            "status": "success",
            "prediction": classification,
            "confidence_percentage": round(confidence, 2),
            "raw_score": raw_score,
            "logs": self.logs
        }
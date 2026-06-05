import cv2
from mtcnn import MTCNN
import logging

logger = logging.getLogger(__name__)

class FaceExtractor:
    def __init__(self):
        # We only need to load the MTCNN model into memory once
        self.detector = MTCNN()

    def extract_and_crop(self, image_path):
        try:
            # EDGE CASE 1: Corrupted or missing file
            img = cv2.imread(image_path)
            if img is None:
                logger.error("Shield Activated: Image could not be read or is completely empty.")
                return None

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # EDGE CASE 2: The "Dog" Scenario (No human faces)
            results = self.detector.detect_faces(img_rgb)
            if not results:
                logger.warning("Shield Activated: No human faces detected in the image.")
                return None

            # Get the bounding box of the most prominent face (the first one)
            bounding_box = results[0]['box']
            x, y, w, h = bounding_box
            
            # EDGE CASE 3: Face is cut off at the edge of the photo
            x, y = max(0, x), max(0, y)
            
            face_crop = img[y:y+h, x:x+w]
            
            # EDGE CASE 4: Math error resulted in a 0-pixel crop
            if face_crop.size == 0:
                logger.warning("Shield Activated: Face crop resulted in an empty image.")
                return None
                
            # Resize to match the 224x224 your EfficientNet expects
            face_resized = cv2.resize(face_crop, (224, 224))
            return face_resized

        except Exception as e:
            # The Ultimate Catch-All: If anything goes wrong, do not crash the server.
            logger.error(f"Fatal Extraction Error Caught: {str(e)}")
            return None
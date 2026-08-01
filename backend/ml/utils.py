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

            # MARGIN FIX: MTCNN gives a very tight box (just eyes/nose/mouth/chin).
            # The training dataset's images had a looser crop (forehead, hair, some
            # shoulder). Without matching that framing, the model sees an unfamiliar
            # zoom level at inference time and predictions become unreliable.
            # Expand the box by ~35% on each side (a bit more on top for forehead/hair).
            margin_x = int(w * 0.35)
            margin_top = int(h * 0.45)
            margin_bottom = int(h * 0.35)

            img_h, img_w = img.shape[:2]
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_top)
            x2 = min(img_w, x + w + margin_x)
            y2 = min(img_h, y + h + margin_bottom)

            # EDGE CASE 3: Face is cut off at the edge of the photo
            x1, y1 = max(0, x1), max(0, y1)

            face_crop = img[y1:y2, x1:x2]
            
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
import os
import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def normalize_dataset(source_dir, target_dir, target_size=(224, 224)):
    """
    Scans the raw dataset, resizes every image to 224x224, 
    and saves them in a clean, uniform directory.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # We assume your source_dir has 'real' and 'fake' subfolders
    for category in ['real', 'fake']:
        category_source = os.path.join(source_dir, category)
        category_target = os.path.join(target_dir, category)
        
        if not os.path.exists(category_source):
            logging.warning(f"Could not find folder: {category_source}")
            continue
            
        if not os.path.exists(category_target):
            os.makedirs(category_target)

        logging.info(f"Normalizing '{category}' images...")
        
        # Loop through all images
        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        files = [f for f in os.listdir(category_source) if f.lower().endswith(valid_extensions)]
        
        for i, filename in enumerate(files):
            img_path = os.path.join(category_source, filename)
            save_path = os.path.join(category_target, filename)
            
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize to exactly 224x224
                    resized_img = cv2.resize(img, target_size)
                    cv2.imwrite(save_path, resized_img)
                    
                    if i % 1000 == 0:
                        logging.info(f"Processed {i}/{len(files)} images in '{category}'...")
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")

    logging.info("Dataset normalization complete! Data is ready for the engine.")

if __name__ == "__main__":
    # Define paths based on your TrueLens structure
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Where your current mixed-resolution data lives
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'train') 
    
    # Where the new, perfect 224x224 data will be saved
    CLEAN_DATA_DIR = os.path.join(BASE_DIR, 'data', 'train_clean')
    
    normalize_dataset(RAW_DATA_DIR, CLEAN_DATA_DIR)
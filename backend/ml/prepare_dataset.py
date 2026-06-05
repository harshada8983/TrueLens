import os
import glob
from tqdm import tqdm
from utils import FaceExtractor

def process_directory(input_dir, output_dir):
    """
    Scans a directory for images, extracts the faces, and saves them to a new directory.
    """
    extractor = FaceExtractor()
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all jpg/png files in the input directory
    image_paths = glob.glob(os.path.join(input_dir, '*.[jp][pn]g'))
    
    print(f"\nProcessing {len(image_paths)} images from {input_dir}...")
    
    successful_crops = 0
    # tqdm provides a professional progress bar in the terminal
    for img_path in tqdm(image_paths):
        try:
            # We don't use output_dir in the function call because we want to rename the file slightly
            face = extractor.extract_and_crop(img_path)
            
            if face is not None:
                filename = os.path.basename(img_path)
                save_path = os.path.join(output_dir, filename)
                import cv2
                cv2.imwrite(save_path, face)
                successful_crops += 1
        except Exception as e:
            # Skip corrupted images without crashing the whole script
            pass 

    print(f"Successfully extracted {successful_crops}/{len(image_paths)} faces.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define our raw data paths
    raw_real_dir = os.path.join(base_dir, 'data', 'train', 'real')
    raw_fake_dir = os.path.join(base_dir, 'data', 'train', 'fake')
    
    # Define where the clean, cropped data will go
    processed_real_dir = os.path.join(base_dir, 'data', 'processed_train', 'real')
    processed_fake_dir = os.path.join(base_dir, 'data', 'processed_train', 'fake')
    
    # 1. Process Real Images
    if os.path.exists(raw_real_dir):
        process_directory(raw_real_dir, processed_real_dir)
    else:
        print(f"Directory not found: {raw_real_dir}")
        
    # 2. Process Fake Images
    if os.path.exists(raw_fake_dir):
        process_directory(raw_fake_dir, processed_fake_dir)
    else:
         print(f"Directory not found: {raw_fake_dir}")
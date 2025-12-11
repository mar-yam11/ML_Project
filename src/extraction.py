import cv2
import os
import numpy as np
from pathlib import Path
from skimage.feature import hog

def list_images(folder):
    extentions = {'.jpg','.png','.jpeg'}
    images = []
    all_folders = os.listdir(folder)
    for folder_name in all_folders:
        folder_extension = Path(folder_name).suffix.lower()
        if folder_extension in extentions:
            images.append(folder_name) 
            
    return images

def compute_hog_feature(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.resize(gray, (64, 64)) 

    features = hog(gray, orientations= 9, pixels_per_cell= (8,8),cells_per_block=(2,2),block_norm='L2-Hys'
                   ,visualize=False,feature_vector=True)
    return features

def extract_features(filePath):
    raw_data = Path(filePath)
    output = raw_data.parent / 'features'
    output_file = output / 'data.npz'
    
    folders = []

    for item in raw_data.iterdir():
        if item.is_dir():
            folders.append(item)
            
    X = [] 
    y = [] 
    class_map = {} 

    for idx, folder in enumerate(folders):
        class_map[idx] = folder.name
        images = list_images(folder)
        
        for img in images:
            feat = compute_hog_feature(folder / img)
            if feat is not None:
                X.append(feat)
                y.append(idx)

    
    X = np.array(X)
    y = np.array(y)
   
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, X=X, y=y, class_map=class_map)
    
    print(f"\nPipeline Finished Successfully!")
    print(f"Features saved to: {output}")
    print(f"Data Shape: {X.shape}, Labels Shape: {y.shape}")
    print(f"Class Map: {class_map}")


if __name__ == "__main__":
    extract_features('data/augmented')
import random
import os 
from PIL import Image
from pathlib import Path
from torchvision import transforms

def list_images(folder):
    extentions = {'.jpg','.png','.jpeg'}
    images = []
    all_folders = os.listdir(folder)
    for folder_name in all_folders:
        folder_extension = Path(folder_name).suffix.lower()
        if folder_extension in extentions:
            images.append(folder_name) 
            
    return images


def augment_image(img):
    
    transform_pipeline = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((64, 64)) 
    ])
    
    return transform_pipeline(img)

    

def augment_class_folder(folder, output, class_count):
    images = list_images(folder)
    
    for img in images:
        src = Path(folder) / img
        dst = Path(output) / Path(img).stem
        dst = dst.with_suffix('.jpg')
        if not (Path(output) / dst.name).exists():
            try: 
                Image.open(src).convert('RGB').save(Path(output) / dst.name, format='JPEG', quality=90)
            except Exception:
                continue
            
    current_files = list_images(output)
    current_len = len(current_files)
    if current_len == 0:
        print(f"Warning: no valid images in {folder}.")
        return 0
    idx = 0
    while current_len < class_count:
        src_name = current_files[idx % len(current_files)]
        src_path = Path(output) / src_name
        try:
            img = Image.open(src_path).convert('RGB')
        except Exception:
            idx += 1
            continue
        aug = augment_image(img)
        new_name = f"aug_{current_len}_{random.randint(0,99999)}.jpg"
        save_path = Path(output) / new_name
        aug.save(save_path, format='JPEG', quality=90)
        current_len += 1
        idx += 1
    return current_len



def augment_data(filePath, class_count = 600):
    raw_data = Path(filePath)
    output_data = raw_data.parent / 'augmented'
   
    folders = []

    for item in raw_data.iterdir():
        if item.is_dir():
            folders.append(item)
    
    for folder in folders:
        output = output_data / folder.name
        output.mkdir(parents=True, exist_ok=True)
        final_clases = augment_class_folder(folder, output, class_count)
        print(f"Class {folder.name} -> {final_clases} images")
    
    print('Augmentation completed!')
        

augment_data('data/cleaned')
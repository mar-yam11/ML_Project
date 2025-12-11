import os 
from PIL import Image, UnidentifiedImageError
from pathlib import Path

def list_images(folder):
    extentions = {'.jpg','.png','.jpeg'}
    images = []
    all_folders = os.listdir(folder)
    for folder_name in all_folders:
        folder_extension = Path(folder_name).suffix.lower()
        if folder_extension in extentions:
            images.append(folder_name) 
            
    return images

def is_image_valid(img_path):
    try:
        img = Image.open(img_path)
        img.verify() 
        img = Image.open(img_path) 
        if img.size[0] > 0 and img.size[1] > 0:
            return True
            
    except (IOError, SyntaxError, UnidentifiedImageError):
        print(f"Corrupt image found: {img_path}") 
        return False
        
    return False

def clean_data (filePath):
    raw_data = Path(filePath)
    output_data = raw_data.parent / 'cleaned'

    folders = []
    
    for item in raw_data.iterdir():
        if item.is_dir():
            folders.append(item) 
    
    for folder in folders:
        output = output_data / folder.name
        output.mkdir(parents=True, exist_ok=True)
        for img in list_images(folder): 
            img_path = folder / img
            if is_image_valid(img_path):
                image = Image.open(img_path).convert('RGB')
                save_path = output / (Path(img).stem + '.jpg')
                image.save(save_path, format='JPEG', quality=90)

                    

clean_data ('data/raw')
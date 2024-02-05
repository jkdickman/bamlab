import os
from PIL import Image, ImageDraw
import numpy as np

def process_dataset(input_dir, output_dir, background_color=(135, 206, 235), shrink_factor=0.5)):
    for dataset_type in ['train', 'test', 'val']:
        img_dir = os.path.join(input_dir, dataset_type, 'images')
        label_dir = os.path.join(input_dir, dataset_type, 'labels')
        
        output_img_dir = os.path.join(output_dir, dataset_type, 'images')
        output_label_dir = os.path.join(output_dir, dataset_type, 'labels')
        
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        
        for img_name in os.listdir(img_dir):
            if not img_name.endswith('.jpg'):
                continue  # Skip non-JPEG files
            
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
            
            with Image.open(img_path) as img:
                w, h = img.size
                
                # Create a new image with sky blue background
                new_img = Image.new('RGB', (w, h), color=background_color)
                
                # Process label file
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            class_id, x_center, y_center, width, height = map(float, line.split())
                            
                            # Convert normalized coords to absolute coords
                            box_w, box_h = width * w, height * h
                            box_x1, box_y1 = (x_center * w) - (box_w / 2), (y_center * h) - (box_h / 2)
                            box_x2, box_y2 = box_x1 + box_w, box_y1 + box_h
                            
                            # Crop and resize drone image
                            drone_img = img.crop((box_x1, box_y1, box_x2, box_y2))
                            drone_img = drone_img.resize((int(box_w * shrink_factor), int(box_h * shrink_factor)))  
                                                class_id, x_center, y_center, width, height = map(float, line.split())
                            new_width = width * shrink_factor
                            new_height = height * shrink_factor
                            new_x_center = 0.5  # Centered in the middle when normalized
                            new_y_center = 0.5  # Centered in the middle when normalized
                            
                            # Calculate new position (example: center of the image)
                            new_x, new_y = (w / 2) - (drone_img.width / 2), (h / 2) - (drone_img.height / 2)
                            new_label_file.write(f"{class_id} {new_x_center} {new_y_center} {new_width} {new_height}\n")

                            # Paste drone image on new background
                            new_img.paste(drone_img, (int(new_x), int(new_y)))
                
                # Save new image
                new_img_path = os.path.join(output_img_dir, img_name)
                new_img.save(new_img_path)
                
                # Optionally, save a new label file if necessary
                # This part depends on how you want to handle labels for the modified images
                


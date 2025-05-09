import cv2
import json
import numpy as np
import random
import glob
import os
from rembg import remove
from PIL import Image
import itertools
# Paths
COCO_IMAGES_DIR = "../coco128/images/train2017/"
COCO_LABELS_DIR = "../coco128/labels/train2017/"
CUSTOM_CLASSES_DIR = "../your-own-dataset/train/images/"
OUTPUT_IMAGES_DIR = "augmented_coco128_v4/images/"
OUTPUT_LABELS_DIR = "augmented_coco128_v4/labels/"

# Ensure output directories exist
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# Load COCO128 images and labels
image_paths = glob.glob(f"{COCO_IMAGES_DIR}/*.jpg")
label_paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f"{COCO_LABELS_DIR}/*.txt")}

# Class folder mapping (folders are named after class)
class_mapping = {"fan": 80, "lamp": 81, "pet_feeder-new": 82}

# Load object images for augmentation, grouped by class
object_images_by_class = {
    key: list(itertools.chain(
        glob.glob(f"{CUSTOM_CLASSES_DIR}/{key}/*.png"),
        glob.glob(f"{CUSTOM_CLASSES_DIR}/{key}/*.jpg")
    ))
    for key in class_mapping
}

# Process each image
for img_path in image_paths:
    img_name = os.path.basename(img_path)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = label_paths.get(os.path.splitext(img_name)[0], None)
    
    # Load the image
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    
    # Load existing labels
    new_labels = []
    if label_path:
        with open(label_path, "r") as f:
            new_labels = f.readlines()
    
    # Randomly choose objects (0-4) to paste
    num_objects = random.randint(0, 4)
    for _ in range(num_objects):
        # Randomly select a class
        class_name = random.choice(list(class_mapping.keys()))
        obj_class = class_mapping[class_name]

        # Select a random object from the corresponding class folder
        obj_path = random.choice(object_images_by_class[class_name])
        
        # Load and remove background from object
        obj_img = Image.open(obj_path)
        obj_no_bg = remove(obj_img)
        obj_no_bg.save("temp.png")  # Save temporary file
        obj_cv = cv2.imread("temp.png", cv2.IMREAD_UNCHANGED)
        
        # Get background dimensions
        bg_h, bg_w = img.shape[:2]

        # Determine object size based on background size (10-30% of background width)
        scale_factor = random.uniform(0.1, 0.3)  
        obj_w = int(bg_w * scale_factor)
        obj_h = int(obj_w * (obj_cv.shape[0] / obj_cv.shape[1]))  # Maintain aspect ratio

        # Ensure object fits in background
        obj_h = min(obj_h, bg_h)
        obj_w = min(obj_w, bg_w)

        obj_resized = cv2.resize(obj_cv, (obj_w, obj_h))
        
        # Random position
        x_offset = random.randint(0, img_w - obj_w)
        y_offset = random.randint(0, img_h - obj_h)
        
        # Blend object onto background
        for c in range(3):
            img[y_offset:y_offset+obj_h, x_offset:x_offset+obj_w, c] = (
                obj_resized[:, :, c] * (obj_resized[:, :, 3] / 255.0) +
                img[y_offset:y_offset+obj_h, x_offset:x_offset+obj_w, c] * (1 - obj_resized[:, :, 3] / 255.0)
            )
        
        # Normalize coordinates and update labels
        x_center = (x_offset + obj_w / 2) / img_w
        y_center = (y_offset + obj_h / 2) / img_h
        w_norm = obj_w / img_w
        h_norm = obj_h / img_h
        new_labels.append(f"{obj_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    # Save the new image
    output_img_path = os.path.join(OUTPUT_IMAGES_DIR, img_name)
    cv2.imwrite(output_img_path, img)
    
    # Save the new label
    output_label_path = os.path.join(OUTPUT_LABELS_DIR, label_name)
    with open(output_label_path, "w") as f:
        f.writelines(new_labels)

print("Augmentation complete! Synthetic images and labels saved.")

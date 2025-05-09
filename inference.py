import os
from ultralytics import YOLO

# Define directories
dir = "test_images/IMG_1288_frame_8.jpg"
label_dir = "test_images/frame_labels/"
out_dir = "test_images/frame_results/"

# Create directories if they don't exist
os.makedirs(label_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

model = YOLO("runs/detect/exp1/weights/best.pt")

results = model(dir)

# Process results
for result in results:
    # Extract filename from result.path
    image_filename = os.path.basename(result.path)
    label_filename = os.path.splitext(image_filename)[0] + ".txt"  # Save as TXT

    label_path = os.path.join(label_dir, label_filename)
    
    with open(label_path, "w") as f:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Class ID
            x, y, w, h = box.xywhn[0].tolist()  # YOLO format (normalized x_center, y_center, width, height)
            
            # Write label in YOLO format: class_id x_center y_center width height
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print(f"✅ Saved labels for {image_filename} -> {label_path}")

print("🎯 Inference complete! Labels saved in", label_dir)

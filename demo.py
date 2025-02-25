from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("runs/detect/train5/weights/best.pt")
# model = YOLO("yolov8n.pt")


# Define path to the image file
# source = "train/images/1_jpg.rf.8d19eab997437013d01bdf8bcf678a5d.jpg"
source = "valid/images/4_jpg.rf.8198f294a9e2450e2aa5bc2a3be548e0.jpg"


# Run inference on the source
results = model(source)  # list of Results objects

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.save(filename="result.jpg")  # save to disk
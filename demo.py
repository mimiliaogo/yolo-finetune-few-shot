from ultralytics import YOLO

# Load the fine-tuned model
model = YOLO("runs/detect/exp1/weights/best.pt")


# Define path to the image file
source = "test_images/IMG_1288_frame_8.jpg"



# Run inference on the source
results = model(source)  # list of Results objects

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.save(filename="result2.jpg")  # save to disk
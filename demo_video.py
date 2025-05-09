from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("runs/detect/coco128v4_augment_83_classes/weights/best.pt")

# Run inference on the entire video
results = model("your_video.mp4", batch=5, save=True, show=False, name="coco128v4_augment_83_classes", conf=0.5)  # save=True saves the output video


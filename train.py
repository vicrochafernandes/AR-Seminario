from ultralytics import YOLO
from dataset import dataset

model = YOLO("yolov8n.pt")
model.train(data=f"{dataset.location}/data.yaml", epochs=50, imgsz=640)
# weights: runs/detect/train/weights/best.pt
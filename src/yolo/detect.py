# src/yolo/detect.py

from ultralytics import YOLO
from src.utils.path import from_root  # your helper function to resolve paths

def run_custom_yolo(image_path: str):
    model = YOLO(from_root("models/yolo-trained.pt"))
    results = model(from_root(image_path))

    boxes_data = []
    class_ids = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_names = model.names

    for box, cls_id, conf in zip(results[0].boxes.xyxy, class_ids, confidences):
        coords = box.tolist()
        label = class_names[int(cls_id)]
        boxes_data.append({
            "label": label,
            "confidence": float(conf),
            "coords": coords
        })

    return boxes_data
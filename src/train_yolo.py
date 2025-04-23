from ultralytics import YOLO

model = YOLO("yolov8m.pt")

train_results = model.train(
    data="datasets/dataset.yml",
    epochs=10,
    imgsz=640,
    device="cpu"
)

results = model("datasets/test/images/1.png")

boxes = results[0].boxes  
class_ids = boxes.cls.cpu().numpy()
confidences = boxes.conf.cpu().numpy()

class_names = model.names

for cls_id, conf in zip(class_ids, confidences):
    label = class_names[int(cls_id)]
    print(f"Detected: {label} (confidence: {conf:.2f})")

model.export(format="onnx", dynamic=True)
# results[0].save(filename="predicted_image.jpg")
from ultralytics import YOLO

model = YOLO("yolo-trained.pt")

results = model("datasets/test/images/1.png")

results[0].show()
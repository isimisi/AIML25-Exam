#%%
from ultralytics import YOLO

#%%
model = YOLO("yolov8m-seg.pt")

train_results = model.train(
    data="datasets/dataset.yml",
    epochs=10,
    imgsz=640,
    device="cpu"
)
#%%
model.export(format="onnx", dynamic=True)
# results[0].save(filename="predicted_image.jpg")
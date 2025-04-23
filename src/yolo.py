#%%
from ultralytics import YOLO
from utils.path import from_root
#%%
model = YOLO(from_root("yolo-trained.pt"))

#%%
results = model(from_root("datasets/test/images/1.png"))

results[0].show()
#%%
# import os
# from pathlib import Path

# while Path.cwd().name != "src" and "src" in str(Path.cwd()):
#     os.chdir("..")  # Move up one directory
# print(f"Working directory set to: {Path.cwd()}")

# %%
import sys
from pathlib import Path

# Add root of the project (AIML25-EXAM) to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

# %%
from ultralytics import YOLO
from src.utils.path import from_root
#%%
model = YOLO(from_root("models/yolo-trained.pt"))

#%%
results = model(from_root("datasets/test/images/1.png"))

results[0].show()
# %%
results
# %%
boxes = results[0].boxes  
class_ids = boxes.cls.cpu().numpy()
confidences = boxes.conf.cpu().numpy()

#%%
boxes[0].xyxy.tolist()[0]
print(boxes[0])
#%%
class_names = model.names

for cls_id, conf in zip(class_ids, confidences):
    label = class_names[int(cls_id)]
    print(f"Detected: {label} (confidence: {conf:.2f})")

# %%

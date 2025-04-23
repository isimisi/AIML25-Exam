from ultralytics import YOLO

class Yolo:
    def __init__(self, model: YOLO):
        self.model = model
        self.results = []
    
    def predict(self, file: str):
        self.results = self.model(file)

    def getBBoxes(self) -> list:
        bboxes = []
        result = self.results[0]
        for box in result.boxes:
            bboxes.append({
                    "xywh": box.xywh.tolist()[0],
                    "confidence": float(box.conf),
                    "class_id": int(box.cls),
                    "xyxy": box.xyxy.tolist()[0]
                })
        return bboxes

    def show(self):
        self.results.show()
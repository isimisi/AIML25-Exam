from ultralytics import YOLO
from PIL import Image
from typing import List
from typing_extensions import TypedDict

class Detection(TypedDict):
    xywh: List[float]
    confidence: float
    class_id: int
    xyxy: List[float]
class Yolo:
    _file: str
    _results: list
    _model: YOLO

    def __init__(self, model: YOLO):
        self._model = model
        self._results = []
        self._file = ""
    
    def predict(self, file: str):
        self._results = self._model(file)
        self._file = file
        return self._results

    def getBBoxes(self) -> List[Detection]:
        bboxes = []
        result = self._results[0]
        for box in result.boxes:
            bboxes.append({
                    "xywh": box.xywh.tolist()[0],
                    "confidence": float(box.conf),
                    "class_id": int(box.cls),
                    "xyxy": box.xyxy.tolist()[0]
                })
        return bboxes

    def show(self) -> None:
        self._results.show()
    
    def cropImages(self, bboxes: List[Detection]):
        images: List[Image.Image] = []
        for bbox in bboxes:
            img = Image.open(self._file)
            [left, top, right, bottom] = bbox["xyxy"]
            cropped = img.crop((left, top, right, bottom))
            images.append(cropped)

        return images

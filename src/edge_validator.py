from src.graph import Edge
from typing import List, Dict, Any
import json
import re


class EdgeValidator:
    true_edges: List[Edge]
    predicted_edges: List[Edge]

    def __init__(self, true_edges: List[Edge], predicted_edges: List[Edge]) -> None:
        self.true_edges = true_edges
        self.predicted_edges = predicted_edges

    @classmethod
    def from_json(cls, true_edges_json: List[Dict[str, Any]], predicted_edges: List[Edge]) -> "EdgeValidator":
        true_edges = [Edge(item["source"], item["target"])
                      for item in true_edges_json]
        return cls(true_edges, predicted_edges)

    @classmethod
    def from_json_file(cls, file_path: str, predicted_edges: List[Edge]) -> "EdgeValidator":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                "JSON file must contain a list of edge dictionaries")

        return cls.from_json(data, predicted_edges)

    def validate(self) -> dict:
        true_ids = {
            re.sub(r'\s+', '', edge.id).lower()
            for edge in self.true_edges
        }
        pred_ids = {
            re.sub(r'\s+', '', edge.id).lower()
            for edge in self.predicted_edges
        }

        tp_ids = true_ids & pred_ids
        fp_ids = pred_ids - true_ids

        precision = len(tp_ids) / len(pred_ids) if pred_ids else 0.0
        recall = len(tp_ids) / len(true_ids) if true_ids else 0.0
        f1 = (2 * precision * recall) / (precision +
                                         recall) if (precision + recall) else 0.0

        return {
            "true_positives": list(tp_ids),
            "false_positives": list(fp_ids),
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def get_true_positive_edges(self) -> List[Edge]:
        tp_ids = {edge.id.lower() for edge in self.true_edges} & {
            edge.id.lower() for edge in self.predicted_edges}
        return [edge for edge in self.predicted_edges if edge.id.lower() in tp_ids]

    def get_false_positive_edges(self) -> List[Edge]:
        tp_ids = {edge.id.lower() for edge in self.true_edges} & {
            edge.id.lower() for edge in self.predicted_edges}
        return [edge for edge in self.predicted_edges if edge.id.lower() not in tp_ids]

    def get_false_negative_edges(self) -> List[Edge]:
        tp_ids = {edge.id.lower() for edge in self.true_edges} & {
            edge.id.lower() for edge in self.predicted_edges}
        return [edge for edge in self.true_edges if edge.id.lower() not in tp_ids]

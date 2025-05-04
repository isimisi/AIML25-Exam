from src.llm_detector import BaseDetector, EdgeResponse, Edge
from typing import List


class MermaidToJSON(BaseDetector):
    def convert(self, diagram: str):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert assistant that converts Mermaid diagram definitions "
                    "into a structured JSON representation."
                    f"Repond in the following format: {self.prepare_response_type(EdgeResponse)}"
                )
            },
            {
                "role": "user",
                "content": f"Convert the following Mermaid diagram into JSON:\n\n{diagram}"
            }
        ]

        # Invoke the LLM
        response = self.model.invoke(
            messages=messages, response_model=EdgeResponse)

        edges: List[Edge] = []
        for edge in response.answer:
            edges.append(Edge(edge["source"], edge["target"]))

        return edges

from pydantic import BaseModel, Field
from src.graph import Edge
from typing import List, Dict, Any
import base64
import litellm
from litellm import completion, ChatCompletionImageObject
from instructor import Mode, from_litellm
from PIL import Image
import io
import json


class ValidationResponse(BaseModel):
    reasoning: str = Field(
        description="A short reasoning for the given precision_score")
    precision_score: float = Field(
        description="A floating score between 0 - 1, indicating how similar the predicted edges are to the true edges")


class LLMEdgeValidator:
    def __init__(self, api_key: str, project_id: str, api_url: str, model_id: str, params: dict[str, Any] = {}):
        """
            Stolen from ma3
        """

        self.api_key = api_key
        self.project_id = project_id
        self.api_url = api_url
        self.model_id = model_id
        self.params = params

        litellm.drop_params = True
        self.client = from_litellm(completion, mode=Mode.JSON)

    def prepare_prompt_content(self, true_edges: List[Edge], predicted_edges: List[Edge]):
        example_true = [{
            "source": "owner",
            "target": "company 1",
        }, {
            "source": "owner",
            "target": "company 2",
        }, {
            "source": "company 2",
            "target": "company 3",
        }, {
            "source": "company 2",
            "target": "company 4",
        }, {
            "source": "entity",
            "target": "company 1",
        }, {
            "source": "fund",
            "target": "company 1",
        }]

        example_predicted = [{
            "source": "Owner (1)",
            "target": "company 2",
        }, {
            "source": "company 2",
            "target": "company 3",
        }, {
            "source": "Company 2",
            "target": "company 4",
        }, {
            "source": "Company 1",
            "target": "entity",
        }, {
            "source": "Fund",
            "target": "company 3",
        }]
        prompt = f"""
            Objective:
                You're a highly advanced AI specialized in validating JSON similarity of edges.
                Your task is to give a precision score based on how similar the predicted edges are to the true edges.
                The predicted edges are generated by an LLM, and therefore may swerve a bit from the true edges.
            
            Input Content:
                You're given 2 lists:
                    true edges and predicted edges
                An edge has the following format {"{ source: str, target: str }"}

            Example:
                true_edges: {example_true}

                predicted_edges: {example_predicted}

                precision_score: 0.67

            Input:
                true_edges: {true_edges}

                predicted_edges: {predicted_edges}

            Provide your answer as an object of {type(ValidationResponse)}
        """

        return {"type": "text", "text": prompt}

    def get_edges_from_json(self, true_edges_json: List[Dict[str, Any]]):
        true_edges = [Edge(item["source"], item["target"])
                      for item in true_edges_json]
        return true_edges

    def bruh():
        print("bruh")

    def get_json_from_file(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                "JSON file must contain a list of edge dictionaries")

        return data

    def invoke(self, true_edges: List[Edge], predicted_edges: List[Edge], **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [self.prepare_prompt_content(true_edges, predicted_edges)],
                }
            ],
            project_id=self.project_id,
            apikey=self.api_key,
            api_base=self.api_url,
            response_model=ValidationResponse,
            **kwargs
        )

        return response

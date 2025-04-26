from pydantic import BaseModel, Field
from src.graph_factory import RawEdge
from typing import List, TypeVar, Any, Union
import base64
import litellm
from litellm import completion, ChatCompletionImageObject
from instructor import Mode, from_litellm
from PIL import Image
import io


class EdgeResponse(BaseModel):
    reasoning: str = Field(
        description="A short reasoning to why the selected edges are valid")
    answer: List[RawEdge] = Field(
        description="Your answer as a valid JSON array containing only valid edges")


ResponseType = TypeVar('ResponseType', bound=BaseModel)


class EdgeDetectorLLM:
    def prepare_image_content(self, image: Union[str, Image.Image], mime_type: str = "image/png") -> ChatCompletionImageObject:
        if isinstance(image, str):
            with open(image, "rb") as f:
                img_bytes = f.read()
        elif isinstance(image, Image.Image):
            buffered = io.BytesIO()
            image.save(buffered, format=mime_type.split("/")[-1].upper())
            img_bytes = buffered.getvalue()
        else:
            raise ValueError(
                "Input must be a filepath (str) or PIL.Image.Image object.")

        base64_image = base64.b64encode(img_bytes).decode('utf-8')

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}",
                "detail": "high"
            },
        }

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

    def prepare_prompt_content(self, found_edges: List[RawEdge]):
        prompt = f"""
        Objective
            You are a highly intelligent AI specialized in image recognition and data transformation of flowchart structures. Your task is to analyze an input image containing flowchart elements (nodes and edges) and transform the recognized content into a structured JSON format compatible with specific interfaces used in a diagramming application.
            
        Image Content:
            Nodes may include shapes, text, or small illustrations.
            Edges describe relationships and are represented as lines/arrows between nodes.
            The image is part of a larger flowchart, which means that partial connections may appear.
        
        Input Contents:
            Existing JSON structure with defined edges from previous images of the full structure. Use this to update or expand the structure with new connections detected in the image.

        Found edges:
            {found_edges}

        Provide your answer as an object of {type(EdgeResponse)}
        """

        return {"type": "text", "text": prompt}

    def invoke(self, image: Union[str, Image.Image], found_edges: List[RawEdge], **kwargs) -> ResponseType:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [self.prepare_prompt_content(found_edges), self.prepare_image_content(image)],
                }
            ],
            project_id=self.project_id,
            apikey=self.api_key,
            api_base=self.api_url,
            response_model=EdgeResponse,
            **kwargs
        )

        return response

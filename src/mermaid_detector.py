from pydantic import BaseModel, Field
from src.graph import RawEdge, Node, Edge
from typing import List, TypeVar, Any, Union, TypedDict
import base64
from PIL import Image
import io
from src.llm_caller import LLMCaller
from src.yolo.yolo import Yolo
from src.graph import Graph
from src.mermaid_to_json import MermaidToJSON


class GraphImage(TypedDict):
    image: Image.Image
    id: str
    nodes: List[str]
    mermaid: str
    edges: List[Edge]


class NodeResponse(BaseModel):
    reasoning: str = Field(
        description="A short reasoning to why the selected nodes are a valid mermaid format")
    answer: List[str] = Field(
        description="Your answer a list of strings of detected nodes as mermaid nodes. Example: [Person[\"person\"], CompanyOwner[\"Company Owner\"]]")


class EdgeResponse(BaseModel):
    reasoning: str = Field(
        description="A short reasoning to why the answer is in correct mermaid diagram format")
    answer: str = Field(
        description="Your answer as a mermaid diagram")


ResponseType = TypeVar('ResponseType', bound=BaseModel)


class BaseDetector:
    def __init__(self, model: LLMCaller):
        self.model = model

    def prepare_image_content(
        self,
        image: Union[str, Image.Image],
        mime_type: str = "image/png",
    ) -> Any:
        if isinstance(image, str):
            with open(image, "rb") as f:
                img_bytes = f.read()
        elif isinstance(image, Image.Image):
            buffered = io.BytesIO()
            image.save(buffered, format=mime_type.split("/")[-1].upper())
            img_bytes = buffered.getvalue()
        else:
            raise ValueError(
                "Input must be a filepath (str) or PIL.Image.Image object."
            )

        base64_image = base64.b64encode(img_bytes).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}",
                "detail": "high",
            },
        }

    def prepare_response_type(self, response_type: ResponseType):
        return "\n".join(
            f"{name} ({field.annotation.__name__}): {field.description or 'No description'}"
            for name, field in response_type.model_fields.items()
        )


class NodeDetector(BaseDetector):
    def prepare_prompt(self):
        prompt = f"""
            Objective:
                You are a highly intelligent AI specialized in image recognition and data transformation of legal structures. 
                Your task is to analyze an input image containing flowchart elements and extract all **nodes (legal entities)** from it. 

            Input Details:
                The input will consist of an image with flowchart elements, including **nodes (legal entities)**.

            Image Content:
                Nodes may include shapes, text, or small illustrations.

            Output:
                {self.prepare_response_type(NodeResponse)}    
        """

        return {"type": "text", "text": prompt}

    def invoke(self, image: Image.Image):
        return self.model.invoke(messages=[{
            "role": "user",
            "content": [self.prepare_prompt(
            ), self.prepare_image_content(image)]
        }], response_model=NodeResponse)


class EdgeDetector(BaseDetector):
    def example(self):
        return """
        graph TD
            A["A"]
            B["B"]
            B_sub["B Sub"]
            C["C"]
            C_sub["C Sub"]

            A --> B
            A --> C
            B --> B_sub
            C --> C_sub"""

    def output(self, response_type: Union[EdgeResponse, None]):
        if (response_type == None):
            return f"""
                Output:
                    - Must only contain a TD mermaid diagram with no subgroups
                    - Must be just a string of the mermaid diagram

                Example of a full output:
                    {self.example()}
                """
        else:
            return f"""
                Output:
                    {self.prepare_response_type(EdgeResponse)}

                Example:
                    {self.example()}
                """

    def prepare_prompt(self, nodes: List[str], response_model: Union[EdgeResponse, None]):

        prompt = ""

        if len(nodes) == 0:
            prompt = f"""
            Objective:
                You are a highly intelligent AI specialized in image recognition and data transformation of legal structures. 
                Your task is to analyze an input image containing flowchart elements and cunstruct a mermaid diagram.
                
            Image Content:
                Nodes may include shapes, text, or small illustrations.
                Edges describe relationships and are represented as lines/arrows between nodes.
                The nodes' labels has no semantic meaning and therefore the labels do not have a relation to each other. 

            {self.output(response_model)}
            """
        else:
            prompt = f"""
            Objective:
                You are a highly intelligent AI specialized in image recognition and data transformation of legal structures. 
                Your task is to analyze an input image containing flowchart elements and create a mermaid diagram.
                You should use the provided **pre-detected nodes** to build the mermaid diagram. 
                The mermaid diagram should be one whole diagram (no subgroups).
                
            Input Details:
                You are provided with all **pre-detected nodes** (entities), listed below:
                {nodes}

                The nodes' labels has no semantic meaning and therefore the labels do not have a relation to each other. 

            Image Content:
                Edges describe relationships and are represented as lines/arrows between nodes.

            {self.output(response_model)}
            """

        return {"type": "text", "text": prompt}

    def invoke(self, image: Image.Image, nodes: List[str], use_pydantric: bool):
        if use_pydantric:
            response_model = EdgeResponse
        else:
            response_model = None

        return self.model.invoke(messages=[{
            "role": "user",
            "content": [self.prepare_prompt(nodes, response_model), self.prepare_image_content(image)]
        }], response_model=response_model)


class MermaidDetector:
    graph_images: List[GraphImage] = []

    def __init__(self, model: LLMCaller, yolo: Yolo, serializer: MermaidToJSON):
        self.edge_detector = EdgeDetector(model)
        self.node_detector = NodeDetector(model)
        self.yolo = yolo
        self.serializer = serializer

    def initiate_image(self, file: str, should_crop: bool = True):
        filename = file.split("/")[-1]
        if should_crop:
            self.yolo.predict(file)
            bboxes = self.yolo.getBBoxes()
            images = self.yolo.cropImages(bboxes)
            print(f"totale images: {len(images)}")
            self.graph_images = [{"mermaid": "", "id": filename + str(
                index), "image": image, "nodes": [], "edges": []} for index, image in enumerate(images)]
        else:
            image = Image.open(file)
            self.graph_images = [
                {"image": image, "id": filename, "mermaid": "", "nodes": [], "edges": []}]

    def detect_nodes(self):
        for graph_image in self.graph_images:
            response = self.node_detector.invoke(graph_image["image"])
            graph_image["nodes"] = response.answer

    def detect_edges(self, use_pydantric=True):
        for graph_image in self.graph_images:
            response = self.edge_detector.invoke(
                graph_image["image"], graph_image["nodes"], use_pydantric)
            if type(response) == str:
                graph_image["mermaid"] = response
            else:
                graph_image["mermaid"] = response.answer

    def convert_edges(self):
        for graph_image in self.graph_images:
            graph_image["edges"] = self.serializer.convert(
                graph_image["mermaid"])

    def get_graph(self):

        all_edges = []
        for graph_image in self.graph_images:
            all_edges.extend(graph_image["edges"])

        graph = Graph(all_edges, [])
        graph.nodes = graph.unique(graph.create_nodes_from_edges(all_edges))

        return graph

from pydantic import BaseModel, Field
from src.graph import RawEdge, Node, Edge
from typing import List, TypeVar, Any, Union, TypedDict
import base64
from PIL import Image
import io
from src.llm_caller import LLMCaller
from src.yolo.yolo import Yolo
from src.graph import Graph


class GraphImage(TypedDict):
    image: Image.Image
    id: str
    nodes: List[Node]
    edges: List[Edge]


class NodeResponse(BaseModel):
    reasoning: str = Field(
        description="A short reasoning to why the selected edges are valid")
    answer: List[str] = Field(
        description="Your answer as a list of strings of detected nodes, using the node labels.")


class EdgeResponse(BaseModel):
    reasoning: str = Field(
        description="A short reasoning to why the selected edges are valid")
    answer: List[RawEdge] = Field(
        description="Your answer as a valid JSON array containing only valid edges")


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


class NodeDetector(BaseDetector):
    def prepare_prompt(self):
        prompt = f"""
            Objective:
                You are a highly intelligent AI specialized in image recognition and data transformation of legal structures. 
                Your task is to analyze an input image containing flowchart elements and extract all **nodes (legal entities)** from it. 
                Your output must be an array of strings containing only the recognized nodes.

            Input Details:
                The input will consist of an image with flowchart elements, including **nodes (legal entities)**.

            Image Content:
                Nodes may include shapes, text, or small illustrations.
                The image is part of a larger flowchart, which means that partial connections may appear.

            Output:
                {type(NodeResponse)}    
        """

        return {"type": "text", "text": prompt}

    def invoke(self, image: Image.Image):
        return self.model.invoke(messages=[{
            "role": "user",
            "content": [self.prepare_prompt(
            ), self.prepare_image_content(image)]
        }], response_model=NodeResponse)


class EdgeDetector(BaseDetector):
    def prepare_prompt(self, nodes: List[str]):
        prompt = ""
        if len(nodes) == 0:
            prompt = f"""
            Objective:
                You are a highly intelligent AI specialized in image recognition and data transformation of legal structures. 
                Your task is to analyze an input image containing flowchart elements and extract all **edges (relationships between nodes)** from it. 
                Your output must be a structured JSON array containing only the edges, compatible with the diagramming application format.
                
            Image Content:
                Nodes may include shapes, text, or small illustrations.
                Edges describe relationships and are represented as lines/arrows between nodes.
                The image is part of a larger flowchart, which means that partial connections may appear.
            
            Instructions:
                - For each connection (line/arrow) between nodes in the image:
                - Identify the connected nodes using their labels.
                - Ensure output is valid JSON and contains only **edge objects** (no nodes).

            Output:
                {type(EdgeResponse)}
            """
        else:
            prompt = f"""
            Objective:
                You are a highly intelligent AI specialized in image recognition and data transformation of legal structures. 
                Your task is to analyze an input image containing flowchart elements and extract all **edges (relationships between nodes)** from it. 
                Your output must be a structured JSON array containing only the edges, compatible with the diagramming application format.
                
            Input Details:
                You are provided with all **pre-detected nodes** (entities), listed below:
                {nodes}
            
            Instructions:
                - For each connection (line/arrow) between nodes in the image:
                - Identify the connected nodes using their labels.
                - Match source and target by their node labels from the input list.
                - Ensure output is valid JSON and contains only **edge objects** (no nodes).

            Output:
                {type(EdgeResponse)}
            """

        return {"type": "text", "text": prompt}

    def invoke(self, image: Image.Image, nodes: List[Node]):
        return self.model.invoke(messages=[{
            "role": "user",
            "content": [self.prepare_prompt([node.id for node in nodes]), self.prepare_image_content(image)]
        }], response_model=EdgeResponse)


class Detector:
    graph_images: List[GraphImage] = []

    def __init__(self, model: LLMCaller, yolo: Yolo):
        self.edge_detector = EdgeDetector(model)
        self.node_detector = NodeDetector(model)
        self.yolo = yolo

    def initiate_image(self, file: str):
        self.yolo.predict(file)
        bboxes = self.yolo.getBBoxes()
        images = self.yolo.cropImages(bboxes)
        print(f"totale images: {len(images)}")
        filename = file.split("/")[-1]
        self.graph_images = [{"edges": [], "id": filename + str(
            index), "image": image, "nodes": []} for index, image in enumerate(images)]

    def detect_nodes(self):
        for graph_image in self.graph_images:
            response = self.node_detector.invoke(graph_image["image"])
            graph_image["nodes"] = [Node(label) for label in response.answer]

    def detect_edges(self):
        for graph_image in self.graph_images:
            response = self.edge_detector.invoke(
                graph_image["image"], graph_image["nodes"])
            graph_image["edges"] = [
                Edge(edge["source"], edge["target"]) for edge in response.answer]

    def get_graph(self):
        all_nodes = []
        all_edges = []
        for graph_image in self.graph_images:
            all_nodes.extend(graph_image["nodes"])
            all_edges.extend(graph_image["edges"])
        graph = Graph(all_edges, all_nodes)

        if (len(graph.nodes) == 0):
            graph.nodes = graph.create_nodes_from_edges(graph.edges)

        return graph

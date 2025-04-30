import json
from typing import List, TypedDict, Tuple, Union
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from thefuzz import fuzz


class RawEdge(TypedDict):
    source: str
    target: str


class Edge:
    source: str
    target: str
    id: str

    def __init__(self, source, target):
        self.id = f"{source}-{target}"
        self.source = source
        self.target = target

    def __repr__(self):
        return json.dumps({
            "id": self.id,
            "source": self.source,
            "target": self.target
        }, indent=2)

    def __str__(self):
        return f"{self.source} -> {self.target} (ID: {self.id})"


class Node:
    id: str
    x: float
    y: float
    width: int
    height: int

    def __init__(self, id: str):
        self.id = id
        self.set_position(0, 0)
        self.set_dimensions(*self.get_node_dimensions())

    def __repr__(self):
        return json.dumps({
            "id": self.id,
            "position": {
                "x": self.x,
                "y": self.y
            },
            "width": self.width,
            "height": self.height
        }, indent=2)

    def set_position(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def set_dimensions(self, width: int, height: int) -> None:
        self.height = height
        self.width = width

    def get_node_dimensions(self, max_width: int = 100) -> Tuple[int, int]:
        char_width = 8
        base_padding = 20
        line_height = 20

        raw_width = len(self.id) * char_width + base_padding

        width = min(raw_width, max_width)
        usable_width = max(1, width - base_padding)
        max_chars_per_line = max(1, usable_width // char_width)
        total_lines = -(-len(self.id) // max_chars_per_line)

        height = total_lines * line_height

        return width, height


class Graph:
    nodes: List[Node]
    edges: List[Edge]
    fuzzy_threshold = 90

    def __init__(self, edges: List[Edge], nodes: List[Node]):
        self.edges = self.unique(edges)
        self.nodes = self.unique(nodes)

    def unique(self, elements: Union[List[Edge] | List[Node]]) -> List[Edge]:
        unique_dict = {}
        for element in elements:
            unique_dict[element.id] = element
        return list(unique_dict.values())

    def fuzzy_unique(self, edges: List[Edge]) -> List[Edge]:
        """
        This don't work well, as nodes have very similar labels
        """

        unique_edges: List[Edge] = []
        seen_ids: List[str] = []

        for edge in edges:
            is_duplicate = False
            for seen_id in seen_ids:
                ratio = fuzz.ratio(edge.id, seen_id)
                if ratio > self.fuzzy_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_edges.append(edge)
                seen_ids.append(edge.id)

        return unique_edges

    def create_nodes_from_edges(self, edges: List[Edge]) -> List[Node]:
        nodes = {}
        for edge in edges:
            nodes[edge.source] = Node(edge.source)
            nodes[edge.target] = Node(edge.target)
        return list(nodes.values())

    def create_digraph(self, direction: str = "TB"):
        digraph = nx.DiGraph()
        digraph.graph['rankdir'] = direction

        for node in self.nodes:
            digraph.add_node(node.id, width=node.width, height=node.height)

        for edge in self.edges:
            digraph.add_edge(edge.source, edge.target, id=edge.id)

        return digraph

    def plot_digraph(self, digraph: nx.DiGraph):
        pos = graphviz_layout(digraph, prog="dot")
        for node in self.nodes:
            node.set_position(*pos[node.id])

        nx.draw(digraph, pos, with_labels=True, node_size=2000,
                node_color='lightblue', font_size=12, arrows=True, node_shape="s")
        return plt

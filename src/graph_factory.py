import json
from typing import List, TypedDict, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

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


class GraphFactory:
    nodes: List[Node]
    edges: List[Edge]

    def __init__(self, raw: List[RawEdge]):
        self.edges = self.unique(self.set_id(raw))
        self.nodes = self.attach_nodes(self.edges)
    
    def unique(self, edges: List[Edge]) -> List[Edge]:
        unique_dict = {}
        for edge in edges:
            unique_dict[edge.id] = edge
        return list(unique_dict.values())
        

    def set_id(self, edges: List[RawEdge]) -> List[Edge]:
        return [Edge(edge["source"], edge["target"]) for edge in edges]
    
    def attach_nodes(self, edges: List[Edge]) -> List[Node]:
        nodes = {}
        for edge in edges:
            nodes[edge.source] = Node(edge.source)
            nodes[edge.target] = Node(edge.target)
        return list(nodes.values())
    
    def create_graph(self, direction: str = "TB"):
        graph = nx.DiGraph()
        graph.graph['rankdir'] = direction

        for node in self.nodes:
            graph.add_node(node.id, width=node.width, height=node.height)

        for edge in self.edges:
            graph.add_edge(edge.source, edge.target, id=edge.id)

        return graph
    
    def plot_graph(self, graph: nx.DiGraph):
        pos = graphviz_layout(graph, prog="dot") 
        nx.draw(graph, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12, arrows=True, node_shape="s")
        plt.show()


    
test_edges: List[RawEdge] = [
        {
            "source": "Co 2 Limited",
            "target": "Co 6 Limited",
        },
        {
            "source": "Co 6 Limited",
            "target": "Co 7 Limited",
        },
        {
            "source": "Co 7 Limited",
            "target": "Co 8 Limited",
        },
        {
            "source": "Co 7 Limited",
            "target": "Co 9 Limited",
        },
        {
            "source": "Co 9 Limited",
            "target": "Co 10 Limited",
        },
        {
            "source": "Co 8 Limited",
            "target": "Other Subs",
        },
        {
            "source": "Co 1 Limited",
            "target": "building",
        },
        {
            "source": "Main Group TopCo",
            "target": "GroupCo 3",
        },
        {
            "source": "Main Group TopCo",
            "target": "GroupCo 4",
        },
        {
            "source": "GroupCo 4",
            "target": "Co 4 Limited",
        },
        {
            "source": "GroupCo 4",
            "target": "Co 5 Limited",
        },
        {
            "source": "Main Group TopCo",
            "target": "Facility 1",
        },
        {
            "source": "Main Group TopCo",
            "target": "Facility 2",
        },
        {
            "source": "Main Group TopCo",
            "target": "Facility 3",
        },
        {
            "source": "Main Group TopCo",
            "target": "Facility 4",
        },
        {
            "source": "Facility 1",
            "target": "Facility 1 Lenders",
        },
        {
            "source": "Facility 2",
            "target": "Facility 2 Lenders",
        },
        {
            "source": "Facility 1",
            "target": "Facility 1 Lenders",
        },
        {
            "source": "Facility 2",
            "target": "Facility 2 Lenders",
        },
        {
            "source": "Facility 1",
            "target": "Facility 1 Lenders",
        },
        {
            "source": "Facility 2",
            "target": "Facility 2 Lenders",
        },
         {
            "source": "Main Group TopCo",
            "target": "GroupCo 3",
        },
        {
            "source": "Main Group TopCo",
            "target": "GroupCo 4",
        },
        {
            "source": "GroupCo 4",
            "target": "Co 4 Limited",
        },
        {
            "source": "GroupCo 4",
            "target": "Co 5 Limited",
        },
        {
            "source": "Main Group TopCo",
            "target": "Facility 1",
        },
        {
            "source": "Main Group TopCo",
            "target": "Facility 2",
        },
        {
            "source": "Main Group TopCo",
            "target": "Facility 3",
        },
        {
            "source": "Main Group TopCo",
            "target": "Facility 4",
        },
        {
            "source": "Facility 1",
            "target": "Facility 1 Lenders",
        },
        {
            "source": "Facility 2",
            "target": "Facility 2 Lenders",
        },
        {
            "source": "Life Interest Trust 1",
            "target": "Main Group TopCo",
        },
        {
            "source": "Life Interest Trust 2",
            "target": "Main Group TopCo",
        },
        {
            "source": "Life Interest Trust 1",
            "target": "BVI Group TopCo",
        },
        {
            "source": "Life Interest Trust 2",
            "target": "BVI Group TopCo",
        },
        {
            "source": "Main Group TopCo",
            "target": "GroupCo 1",
        },
        {
            "source": "Main Group TopCo",
            "target": "GroupCo 2",
        },
        {
            "source": "Main Group TopCo",
            "target": "GroupCo 3",
        },
        {
            "source": "BVI Group TopCo",
            "target": "BVI Group Sub 1",
        },
        {
            "source": "GroupCo 2",
            "target": "Co 1 Limited",
        },
        {
            "source": "GroupCo 2",
            "target": "Co 2 Limited",
        },
        {
            "source": "GroupCo 2",
            "target": "Co 3 Limited",
        },
    ]

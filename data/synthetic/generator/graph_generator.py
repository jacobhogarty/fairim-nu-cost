from abc import (
    ABC,
    abstractmethod,
)
from networkx import (
    spring_layout,
    draw,
)

import matplotlib.pyplot as plt


class GraphGenerator(ABC):
    def __init__(self):
        self.graph = None

    @abstractmethod
    def generate_graph(self, *args, **kwargs):
        """
        Abstract method for generating graphs
        """
        pass

    def visualise(self, node_size=500, with_labels=True):
        """
        Visualises the graph

        Args:
            node_size: Number of nodes
            with_labels: Labels of the graph
        """
        if self.graph is None:
            raise ValueError("No graph generated yet. Call generate_graph() first.")

        plt.figure(figsize=(8, 6))
        pos = spring_layout(self.graph)  # Default layout

        draw(
            self.graph,
            pos,
            with_labels=with_labels,
            node_size=node_size,
            node_color='skyblue',
            edge_color='gray',
            font_size=10,
            font_weight='bold',
        )

        plt.show()

    def get_graph(self):
        """
        Returns the generated graph

        Returns:
            The generated graph
        """
        return self.graph

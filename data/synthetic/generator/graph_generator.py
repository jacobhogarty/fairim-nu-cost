from abc import (
    ABC,
    abstractmethod,
)
from networkx import (
    spring_layout,
    draw_networkx,
    draw_networkx_edge_labels,
    get_edge_attributes,
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

    def visualise(self, node_size=500, with_labels=True, edge_labels=True):
        """
        Visualises the graph

        Args:
            node_size: Number of nodes
            with_labels: Labels of the graph
        """
        if self.graph is None:
            raise ValueError("No graph generated yet. Call generate_graph() first.")

        plt.figure(figsize=(12, 8))
        pos = spring_layout(
            self.graph,
            k=0.15,
            iterations=50,
        )

        draw_networkx(
            self.graph,
            pos,
            node_size=node_size,
            with_labels=with_labels,
            font_size=8,
            node_color="lightblue",
            edge_color="gray",
            alpha=0.7,
        )

        if edge_labels and get_edge_attributes(self.graph, 'weight'):
            edge_labels = get_edge_attributes(self.graph, 'weight')
            draw_networkx_edge_labels(
                self.graph,
                pos,
                edge_labels=edge_labels,
                font_size=7,
                font_color="red",
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
            )

        plt.show()

    def get_graph(self):
        """
        Returns the generated graph

        Returns:
            The generated graph
        """
        return self.graph

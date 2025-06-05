from abc import (
    ABC,
    abstractmethod,
)
import pickle
from pathlib import Path

from networkx import (
    Graph,
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

    def serialise(self, path: Path, filename: str) -> pickle:
        """
        Serialises the graph using pickle

        Args:
            path: Path object representing the path of the file
            filename: Name of the file

        Returns:
            Pickled graph
        """
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f'{filename}.pkl'

        with open(file_path, 'wb') as file:
            pickle.dump(
                obj=self.graph,
                file=file,
            )

        print(f'Graph serialised to {file_path}')

    def get_graph(self) -> Graph:
        """
        Returns the generated graph

        Returns:
            The generated graph
        """
        return self.graph

    def visualise(
            self,
            node_size: int = 500,
            with_labels: bool = True,
            edge_labels: bool = True,
    ):
        """
        Visualises the graph

        Args:
            node_size: Number of nodes
            with_labels: Labels of the graph
            edge_labels: Labels of the edges
        """
        if self.graph is None:
            raise ValueError('No graph generated yet. Call generate_graph() first')

        plt.figure(figsize=(12, 8))
        pos = spring_layout(
            self.graph,
            k=0.15,
            iterations=50,
        )

        draw_networkx(
            G=self.graph,
            pos=pos,
            node_size=node_size,
            with_labels=with_labels,
            font_size=8,
            node_color='lightblue',
            edge_color='gray',
            alpha=0.7,
        )

        if edge_labels and get_edge_attributes(self.graph, 'weight'):
            edge_labels = get_edge_attributes(self.graph, 'weight')
            draw_networkx_edge_labels(
                G=self.graph,
                pos=pos,
                edge_labels=edge_labels,
                font_size=7,
                font_color='red',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
            )

        plt.show()

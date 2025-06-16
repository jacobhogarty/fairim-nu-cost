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
    draw_networkx_nodes,
    draw_networkx_labels,
)

from community import best_partition

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class GraphGenerator(ABC):
    def __init__(self):
        self.graph = None
        self.node_costs = None
        self.communities = None

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

    def louvain_communities(self, ) -> list[int]:
        """
        Returns the louvain communities

        Returns:
            The list of louvain communities
        """
        # Uses louvain best communities
        partition = best_partition(
            self.graph.to_undirected(),
        )
        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)

        return list(communities.values())

    def visualise(
            self,
            node_size: int = 500,
            with_labels: bool = True,
            edge_labels: bool = True,
            node_labels: bool = True,
    ):
        """
        Visualises the graph

        Args:
            node_size: Number of nodes
            with_labels: Labels of the graph
            edge_labels: Labels of the edges
            node_labels: Labels of the nodes
        """
        if self.graph is None:
            raise ValueError('No graph generated yet. Call generate_graph() first')

        plt.figure(figsize=(12, 8))
        pos = spring_layout(
            self.graph,
            k=0.15,
            iterations=50,
            seed=42,
        )

        communities = getattr(self, 'communities', None)

        if communities is None:
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
        else:
            num_communities = len(communities)
            colour_map = cm.get_cmap('tab10', num_communities)

            for i, community in enumerate(communities):
                draw_networkx_nodes(
                    self.graph,
                    pos,
                    nodelist=community,
                    node_color=[colour_map(i)],
                    node_size=node_size,
                    label=f"Community {i}",
                    alpha=0.9
                )
            if with_labels:
                draw_networkx(
                    self.graph,
                    pos,
                    nodelist=self.graph.nodes(),
                    with_labels=True,
                    font_size=8,
                    node_color='none',
                    edge_color='gray',
                    alpha=0.5,
                )

        draw_networkx(
            self.graph,
            pos,
            edgelist=self.graph.edges(),
            edge_color='gray',
            alpha=0.7,
            node_size=0,
            with_labels=False,
        )

        if node_labels:
            label_offset = {node: (pos[node][0], pos[node][1] - 0.05) for node in self.graph.nodes()}
            cost_labels = {node: f"{self.node_costs.get(node, '?')}%.2f" for node in self.graph.nodes()}
            draw_networkx_labels(
                self.graph,
                pos=label_offset,
                labels=cost_labels,
                font_size=8,
                font_color='black',
            )

        if edge_labels and get_edge_attributes(self.graph, 'weight'):
            edge_label_dict = get_edge_attributes(self.graph, 'weight')
            draw_networkx_edge_labels(
                G=self.graph,
                pos=pos,
                edge_labels=edge_label_dict,
                font_size=7,
                font_color='red',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
            )

        if communities is not None:
            legend_handles = [
                Line2D(
                    xdata=[0],
                    ydata=[0],
                    marker='o',
                    color='w',
                    label=f'Community {i}',
                    markerfacecolor=colour_map(i),
                    markersize=6,
                )
                for i in range(num_communities)
            ]

            plt.legend(handles=legend_handles, scatterpoints=1)

        plt.axis('off')
        plt.show()

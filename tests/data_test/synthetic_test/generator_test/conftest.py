import pytest
import networkx as nx
from src.data import GraphGenerator


@pytest.fixture
def generate_complete_graph():
    """
    Generates a complete graph
    """

    class CompleteGraphGenerator(GraphGenerator):
        def generate_graph(self, n):
            """
            Generates a complete graph with n nodes
            """
            self.graph = nx.complete_graph(n)

    return CompleteGraphGenerator()

import pytest


@pytest.mark.parametrize(
    "nodes",
    [
        10,
    ],
)
def test__graph_generator(generate_complete_graph, nodes):
    """
    Tests the GraphGenerator class by testing generation of a complete graph.

    Args:
        generate_complete_graph: Complete graph to test
        nodes: Number of nodes to generate

    Asserts:
           Length of nodes is correct
           Length of edges is correct
    """
    generate_complete_graph.generate_graph(
        n=nodes,
    )

    assert len(generate_complete_graph.ba_100_graph) == nodes

    expected_edges = nodes * (nodes - 1) // 2
    assert len(generate_complete_graph.ba_100_graph.edges) == expected_edges

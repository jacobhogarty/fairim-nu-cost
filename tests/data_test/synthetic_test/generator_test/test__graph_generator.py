import pytest


@pytest.mark.parametrize(
    "nodes",
    [
        10,
    ],
)
def test__graph_generator(generate_complete_graph, nodes):
    generate_complete_graph.generate_graph(
        n=nodes,
    )

    assert len(generate_complete_graph.graph) == nodes

    expected_edges = nodes * (nodes - 1) // 2
    assert len(generate_complete_graph.graph.edges) == expected_edges

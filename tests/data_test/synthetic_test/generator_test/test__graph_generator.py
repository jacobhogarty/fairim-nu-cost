import pytest


@pytest.mark.parametrize(
    "nodes",
    [
        (
                10,
        ),
    ]
)
def test__graph_generator(generate_complete_graph, nodes):
    generate_complete_graph.generate_graph(10)

    assert len(generate_complete_graph.graph) == 10
    assert len(generate_complete_graph.graph.edges) == 45

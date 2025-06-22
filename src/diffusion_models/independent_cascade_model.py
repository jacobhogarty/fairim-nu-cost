"""
Implementation of Independent Cascade Model from Kempe et al. 2003.
Streamlined version with removed redundancy.
"""
from typing import Union

import networkx as nx
import numpy as np
from numba import jit, prange


def graph_to_arrays(graph: nx.Graph) -> tuple:
    """
    Convert NetworkX graph to efficient array representation for Numba.
    """
    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for idx, node in enumerate(nodes)}

    edges = []
    for u, v in graph.edges():
        edges.append([node_to_idx[u], node_to_idx[v]])
        if not graph.is_directed():
            edges.append([node_to_idx[v], node_to_idx[u]])

    edge_array = np.array(edges, dtype=np.int32) if edges else np.empty((0, 2), dtype=np.int32)
    return edge_array, node_to_idx, idx_to_node


@jit(nopython=True, cache=True)
def _build_adjacency_lists(edges: np.ndarray, num_nodes: int):
    """
    Build compressed adjacency list representation from edge array.
    """
    # Count outgoing edges for each node
    out_degrees = np.zeros(num_nodes, dtype=np.int32)
    for i in range(edges.shape[0]):
        out_degrees[edges[i, 0]] += 1

    # Build offset array
    offsets = np.zeros(num_nodes + 1, dtype=np.int32)
    for i in range(num_nodes):
        offsets[i + 1] = offsets[i] + out_degrees[i]

    # Fill neighbors array
    neighbors = np.zeros(edges.shape[0], dtype=np.int32)
    current_pos = offsets[:-1].copy()

    for i in range(edges.shape[0]):
        src = edges[i, 0]
        dst = edges[i, 1]
        neighbors[current_pos[src]] = dst
        current_pos[src] += 1

    return neighbors, offsets


@jit(nopython=True, cache=True)
def _independent_cascade_core(
        neighbors: np.ndarray,
        offsets: np.ndarray,
        seeds: np.ndarray,
        probability: float,
        max_steps: int = 0,
        random_state: int = 42,
) -> np.ndarray:
    """
    Core Independent Cascade implementation using Numba.
    """
    np.random.seed(random_state)

    num_nodes = len(offsets) - 1
    active = np.zeros(num_nodes, dtype=np.bool_)

    max_queue_size = min(num_nodes * 10, 100000)
    queue = np.zeros(max_queue_size * 2, dtype=np.int32)
    queue_size = 0
    queue_pos = 0

    # Initialize with seeds
    for seed in seeds:
        if seed < num_nodes and not active[seed]:
            active[seed] = True
            if queue_size < max_queue_size:
                queue[queue_size * 2] = seed
                queue[queue_size * 2 + 1] = 0
                queue_size += 1

    # Process queue
    while queue_pos < queue_size:
        current_node = queue[queue_pos * 2]
        current_step = queue[queue_pos * 2 + 1]
        queue_pos += 1

        if 0 < max_steps <= current_step:
            continue

        # Check all neighbors
        start_idx = offsets[current_node]
        end_idx = offsets[current_node + 1]

        for i in range(start_idx, end_idx):
            neighbor = neighbors[i]
            if not active[neighbor] and np.random.random() < probability:
                active[neighbor] = True
                if queue_size < max_queue_size:
                    queue[queue_size * 2] = neighbor
                    queue[queue_size * 2 + 1] = current_step + 1
                    queue_size += 1

    return np.where(active)[0]


@jit(nopython=True, parallel=True, cache=True)
def _estimate_influence_parallel(
        neighbors: np.ndarray,
        offsets: np.ndarray,
        seeds: np.ndarray,
        probability: float,
        num_simulations: int,
        max_steps: int = 0,
        base_seed: int = 42
) -> float:
    """
    Parallel estimation of influence spread using multiple simulations.
    """
    total_spread = 0.0

    for i in prange(num_simulations):
        thread_seed = base_seed + i * 12345
        activated = _independent_cascade_core(
            neighbors, offsets, seeds, probability, max_steps, thread_seed
        )
        total_spread += len(activated)

    return total_spread / num_simulations


@jit(nopython=True, parallel=True, cache=True)
def _estimate_influence_per_group_parallel(
        neighbors: np.ndarray,
        offsets: np.ndarray,
        seeds: np.ndarray,
        node_groups: np.ndarray,
        probability: float,
        num_simulations: int,
        num_groups: int,
        base_seed: int = 42,
) -> np.ndarray:
    """
    Parallel estimation of influence spread per group.
    """
    group_totals = np.zeros(num_groups, dtype=np.float64)

    for i in prange(num_simulations):
        thread_seed = base_seed + i * 12345
        activated = _independent_cascade_core(
            neighbors, offsets, seeds, probability, 0, thread_seed
        )

        if len(activated) > 0:
            # For rates: count per group divided by total activated in this simulation
            local_group_counts = np.zeros(num_groups, dtype=np.float64)
            for node in activated:
                if node < len(node_groups):
                    local_group_counts[node_groups[node]] += 1.0

            # Normalize by total activated in this simulation
            total_activated = float(len(activated))
            for g in range(num_groups):
                group_totals[g] += local_group_counts[g] / total_activated
        else:
            # For counts: just sum the counts
            for node in activated:
                if node < len(node_groups):
                    group = node_groups[node]
                    group_totals[group] += 1.0

    return group_totals / num_simulations


class IndependentCascadeModel:
    """
    Optimised Independent Cascade Model using Numba acceleration.
    """

    def __init__(self, graph: nx.Graph):
        self.original_graph = graph
        self.edges, self.node_to_idx, self.idx_to_node = graph_to_arrays(graph)
        self.num_nodes = len(self.node_to_idx)

        if len(self.edges) > 0:
            self.neighbors, self.offsets = _build_adjacency_lists(self.edges, self.num_nodes)
        else:
            self.neighbors = np.array([], dtype=np.int32)
            self.offsets = np.zeros(self.num_nodes + 1, dtype=np.int32)

    def _convert_seeds(self, seeds: Union[set, list]) -> np.ndarray:
        """
        Convert seed nodes to array indices.
        """
        seed_indices = []
        for seed in seeds:
            if seed in self.node_to_idx:
                seed_indices.append(self.node_to_idx[seed])
        return np.array(seed_indices, dtype=np.int32)

    def run_cascade(
            self,
            seeds: Union[set, list],
            probability: float = 0.1,
            max_steps: int = 0,
            random_state: int = 42
    ) -> set:
        """
        Run a single Independent Cascade simulation.

        Args:
            seeds: Set or list of initial active nodes
            probability: Activation probability for edges
            max_steps: Maximum propagation steps (0 for unlimited)
            random_state: Random seed for reproducibility

        Returns:
            Set of activated nodes
        """
        if not seeds or self.num_nodes == 0:
            return set()

        seed_indices = self._convert_seeds(seeds)
        if len(seed_indices) == 0:
            return set()

        activated_indices = _independent_cascade_core(
            self.neighbors, self.offsets, seed_indices,
            probability, max_steps, random_state
        )

        return {self.idx_to_node[idx] for idx in activated_indices}

    def estimate_influence(
            self,
            seeds: Union[set, list],
            num_simulations: int = 100,
            probability: float = 0.1,
            max_steps: int = 0,
            random_state: int = 42
    ) -> float:
        """
        Estimate expected influence spread via multiple parallel simulations.

        Args:
            seeds: Set of seed nodes
            num_simulations: Number of MC simulations
            probability: Activation probability
            max_steps: Maximum propagation steps
            random_state: Base random seed

        Returns:
            Expected influence spread
        """
        if not seeds or self.num_nodes == 0:
            return 0.0

        seed_indices = self._convert_seeds(seeds)
        if len(seed_indices) == 0:
            return 0.0

        return _estimate_influence_parallel(
            self.neighbors, self.offsets, seed_indices,
            probability, num_simulations, max_steps, random_state
        )

    def estimate_influence_by_community(
            self,
            seeds: Union[set, list],
            probability: float = 0.01,
            num_simulations: int = 100,
            random_state: int = 42,
    ) -> dict:
        """
        Estimate expected influence per community via parallel simulations.
        Communities are extracted from node 'community' attributes.

        Args:
            seeds: List of initial seed nodes
            probability: Probability of influence along an edge
            num_simulations: Number of simulations to run
            random_state: Base random seed

        Returns:
            Average rate of influenced nodes per community
        """
        if not seeds or self.num_nodes == 0:
            return {}

        # Extract communities from graph node attributes
        communities = nx.get_node_attributes(self.original_graph, 'community')
        if not communities:
            return {}

        seed_indices = self._convert_seeds(seeds)
        if len(seed_indices) == 0:
            unique_communities = set(communities.values())
            return {c: 0.0 for c in unique_communities}

        # Create community mapping
        unique_communities = list(set(communities.values()))
        community_to_idx = {community: idx for idx, community in enumerate(unique_communities)}
        node_communities = np.zeros(self.num_nodes, dtype=np.int32)

        for node, community in communities.items():
            if node in self.node_to_idx:
                node_idx = self.node_to_idx[node]
                community_idx = community_to_idx[community]
                node_communities[node_idx] = community_idx

        community_counts = _estimate_influence_per_group_parallel(
            self.neighbors, self.offsets, seed_indices, node_communities,
            probability, num_simulations, len(unique_communities), random_state,
        )

        # Create a dictionary from counts
        result: dict = {unique_communities[i]: count for i, count in enumerate(community_counts)}

        total_influenced = sum(result.values())
        if total_influenced > 0:
            result = {
                community: count / total_influenced for community, count in result.items()
            }
        else:
            result = {community: 0.0 for community in result}

        return result

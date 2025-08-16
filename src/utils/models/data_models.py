from dataclasses import dataclass


@dataclass
class CELFNode:
    """
    Data structure for CELF++ node with lazy evaluation metadata
    """
    node_id: int
    marginal_gain: float
    cost: float
    marginal_gain_per_cost: float
    iteration_updated: int

    def __lt__(self, other):
        return self.marginal_gain_per_cost > other.marginal_gain_per_cost
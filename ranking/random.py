import numpy as np
from attack_graph import StateAttackGraph
from ranking.ranking import RankingMethod


class RandomRankingMethod(RankingMethod):
    def __init__(self, graph: StateAttackGraph):
        super().__init__(list(graph.exploits))

    def get_score(self) -> float:
        return np.random.rand()

    def get_score_with_exploit_removed(self, id_exploit: int) -> float:
        return np.random.rand()

import numpy as np
from attack_graph import StateAttackGraph
from ranking.ranking import RankingMethod
from typing import Dict


class RandomRankingMethod(RankingMethod):
    def __init__(self, graph: StateAttackGraph):
        super().__init__(graph)

    def rank_exploits(self) -> Dict[int, int]:
        ids_exploits = [None] + list(self.graph.exploits)
        n_exploits = len(ids_exploits)
        ranks = np.random.choice(n_exploits, size=n_exploits, replace=False)
        ordering = dict([(ids_exploits[i], ranks[i])
                         for i in range(n_exploits)])
        return ordering

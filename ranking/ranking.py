from attack_graph import BaseGraph
from scipy.stats import rankdata
from typing import Dict, List, Tuple


class RankingMethod:
    def __init__(self, ids_exploits: List[int]):
        self.ids_exploits = ids_exploits

    def rank_exploits(self) -> Tuple[Dict[int, int], Dict[int, float]]:
        scores: Dict[int, float] = {}

        # Evaluate the score when removing no exploit
        scores[None] = self.get_score()

        # Evaluate the scores when removing one exploit
        for id_exploit in self.ids_exploits:
            scores[id_exploit] = self.get_score_with_exploit_removed(
                id_exploit)

        # Create the ordering of the exploits based on the corresponding scores
        ranks = rankdata(list(scores.values()), method="ordinal") - 1
        ordering = dict([(list(scores)[i], int(ranks[i]))
                         for i in range(len(self.ids_exploits) + 1)])

        return ordering, scores

    def get_score(self) -> float:
        return

    def get_score_with_exploit_removed(self, id_exploit: int) -> float:
        return

    @staticmethod
    def _get_pruned_graph(graph: BaseGraph, id_exploit: int) -> BaseGraph:
        ids_exploits = list(graph.exploits)
        ids_exploits.remove(id_exploit)

        pruned_graph: BaseGraph = graph.get_pruned_graph(ids_exploits)

        # Check that the pruned graph still has node. If not, it means that it
        # is impossible to obtain the goal proposition
        if len(list(pruned_graph.nodes)) == 0:
            return None
        else:
            return pruned_graph

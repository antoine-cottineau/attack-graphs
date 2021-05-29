from attack_graph import BaseGraph
from typing import Dict


class RankingMethod:
    def __init__(self, graph: BaseGraph):
        self.graph = graph

    def rank_exploits(self) -> Dict[int, float]:
        ids_exploits = list(self.graph.exploits)
        scores: Dict[int, float] = {}

        # Evaluate the score when removing no exploit
        print("Applying ranking method without removing any exploit")
        scores[None] = self._get_score()

        # Evaluate the scores when removing one exploit
        for id_exploit in ids_exploits:
            print("Applying ranking method with exploit {} removed".format(
                id_exploit))
            scores[id_exploit] = self._get_score_with_exploit_removed(
                id_exploit)

        return scores

    def _get_score(self) -> float:
        return

    def _get_score_for_graph(self, graph: BaseGraph) -> float:
        return

    def _get_score_with_exploit_removed(self, id_exploit: int) -> float:
        pruned_graph = self._get_pruned_graph(id_exploit)

        if pruned_graph is None:
            return float("-inf")
        else:
            return self._get_score_for_graph(pruned_graph)

    def _get_pruned_graph(self, id_exploit: int) -> BaseGraph:
        ids_exploits = list(self.graph.exploits)
        ids_exploits.remove(id_exploit)

        pruned_graph: BaseGraph = self.graph.get_pruned_graph(ids_exploits)

        # Check that the pruned graph still has node. If not, it means that it
        # is impossible to obtain the goal proposition
        if len(list(pruned_graph.nodes)) == 0:
            return None
        else:
            return pruned_graph

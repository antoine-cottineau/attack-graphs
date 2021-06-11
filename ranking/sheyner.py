import numpy as np
from attack_graph import StateAttackGraph
from ranking.ranking import RankingMethod
from typing import Dict, Tuple


class ValueIteration(RankingMethod):
    def __init__(self,
                 graph: StateAttackGraph,
                 precision: float = 1e-4,
                 lamb: float = 0.9):
        super().__init__(list(graph.exploits))
        self.graph = graph
        self.precision = precision
        self.lamb = lamb

    def apply(self) -> Tuple[Dict[int, float], Dict[int, int]]:
        values: Dict[int,
                     float] = dict([(node, 0) for node in self.graph.nodes])
        chosen_successors: Dict[int,
                                float] = dict([(node, 0)
                                               for node in self.graph.nodes])
        delta = np.inf

        while delta > self.precision:
            new_values: Dict[int, float] = dict([(node, 0)
                                                 for node in self.graph.nodes])
            for node in self.graph.nodes:
                node_value = values[node]
                reward = self._get_reward(node)
                successors = self._get_successors(node)

                # If the node is the final node, its value is always 1
                if len(successors) == 0:
                    new_values[node] = 1
                    chosen_successors[node] = node
                    continue

                # Find the best action
                best_value = -np.inf
                best_successor = None
                for successor, probability in successors.items():
                    successor_value = values[successor]

                    # The attacker either manages to perform the attack (with
                    # the above probability) or fails to. In the latter case,
                    # the attacker stays at the same node.
                    new_value = reward + self.lamb * (
                        probability * successor_value +
                        (1 - probability) * node_value)

                    if new_value > best_value:
                        best_value = new_value
                        best_successor = successor

                # Update the current values and chosen actions
                new_values[node] = best_value
                chosen_successors[node] = best_successor

            # Compute delta
            delta = ValueIteration._compute_delta(values, new_values)

            values = new_values.copy()

        return values, chosen_successors

    def get_score(self) -> float:
        values = self.apply()[0]
        score = values[0]
        return score

    def get_score_with_exploit_removed(self, id_exploit: int) -> float:
        pruned_graph = self._get_pruned_graph(self.graph, id_exploit)

        if pruned_graph is None:
            return float("-inf")
        else:
            score = ValueIteration(pruned_graph).get_score()
            return score

    def _get_successors(self, node: int) -> Dict[int, float]:
        result = dict([(s, self.graph.get_edge_probability(node, s))
                       for s in self.graph.successors(node)])
        return result

    def _get_reward(self, node: int) -> float:
        if node in self.graph.goal_nodes:
            return 1
        else:
            return 0

    @staticmethod
    def _compute_delta(before: Dict[int, float], after: Dict[int,
                                                             float]) -> float:
        return np.linalg.norm(np.array(list(before.values())) -
                              np.array(list(after.values())),
                              ord=2)

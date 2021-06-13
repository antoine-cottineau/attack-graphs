import numpy as np
from attack_graph import BaseGraph, DependencyAttackGraph, StateAttackGraph
from copy import deepcopy
from ranking.ranking import RankingMethod
from typing import Dict


class ValueIteration(RankingMethod):
    def __init__(self,
                 graph: BaseGraph,
                 precision: float = 1e-4,
                 lamb: float = 0.9):
        super().__init__(list(graph.exploits))
        self.graph = graph
        self.precision = precision
        self.lamb = lamb
        if isinstance(graph, DependencyAttackGraph):
            self.exploit_probabilities = graph.get_nodes_probabilities()

    def apply(self) -> Dict[int, float]:
        values: Dict[int,
                     float] = dict([(node, 0) for node in self.graph.nodes])
        delta = np.inf

        while delta > self.precision:
            new_values = deepcopy(values)

            for node in self.graph.nodes:
                node_value = values[node]
                reward = self._get_reward(node)
                successors = self._get_successors(node)

                # If the node is the final node, its value is always 1
                if len(successors) == 0:
                    new_values[node] = 1
                    continue

                # Find the best action
                best_value = -np.inf
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

                # Update the current values and chosen actions
                new_values[node] = best_value

            # Compute delta
            delta = ValueIteration._compute_delta(values, new_values)

            values = new_values

        return values

    def get_score(self) -> float:
        values = self.apply()
        if isinstance(self.graph, StateAttackGraph):
            score = values[0]
        elif isinstance(self.graph, DependencyAttackGraph):
            score = 0
            for node, id_proposition in self.graph.nodes(
                    data="id_proposition"):
                if id_proposition is None:
                    continue
                if not self.graph.propositions[id_proposition]["initial"]:
                    continue
                score += values[node]
        return score

    def get_score_with_exploit_removed(self, id_exploit: int) -> float:
        pruned_graph = self._get_pruned_graph(self.graph, id_exploit)

        if pruned_graph is None:
            return float("-inf")
        else:
            score = ValueIteration(pruned_graph).get_score()
            return score

    def _get_successors(self, node: int) -> Dict[int, float]:
        successors = list(self.graph.successors(node))

        if isinstance(self.graph, StateAttackGraph):
            result = dict([(s, self.graph.get_edge_probability(node, s))
                           for s in successors])
        elif isinstance(self.graph, DependencyAttackGraph):
            result = {}
            data = self.graph.nodes[node]
            if "id_proposition" in data:
                for successor in successors:
                    result[successor] = 1
            else:
                result[successors[0]] = self.exploit_probabilities[node]

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

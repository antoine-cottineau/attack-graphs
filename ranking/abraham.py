import numpy as np
from attack_graph import StateAttackGraph
from ranking.ranking import RankingMethod
from typing import Tuple


class ExpectedPathLength(RankingMethod):
    def __init__(self, graph: StateAttackGraph):
        super().__init__(graph)

    def apply(self) -> float:
        self.Q, self.R = self._create_Q_and_R()
        self.N = self._create_N()
        self.B = self._create_B()

        # We only return the expected path length from the initial state
        result = self.B.sum()
        return result

    def _get_score(self) -> float:
        score = self.apply()
        return score

    def _get_score_for_graph(self, graph: StateAttackGraph) -> float:
        score = ExpectedPathLength(graph)._get_score()
        return score

    def _get_edge_probability(self, src: int, dst: int) -> float:
        ids_exploits = self.graph.edges[src, dst]["ids_exploits"]

        # The probability of the action being successful is equal to the
        # max of the probilities of the exploits
        probabilities = []
        for id_exploit in ids_exploits:
            # The probability is equal to the CVSS score divided by 10 (to
            # get a value between 0 and 1)
            probability = self.graph.exploits[id_exploit]["cvss"] / 10
            probabilities.append(probability)
        return max(probabilities)

    def _create_Q_and_R(self) -> Tuple[np.ndarray, np.ndarray]:
        transient_nodes = set(self.graph.nodes) - set(self.graph.goal_nodes)
        absorbing_states = set(self.graph.goal_nodes)

        # Q is a sub matrix of P that contains transitions from transient
        # nodes to transient nodes
        Q = np.zeros((len(transient_nodes), len(transient_nodes)))

        # R is a sub matrix of P that contains transitions from transient
        # nodes to absorbing nodes
        R = np.zeros((len(transient_nodes), len(absorbing_states)))

        all_nodes = list(transient_nodes) + list(absorbing_states)
        node_ordering = dict([(all_nodes[i], i)
                              for i in range(len(all_nodes))])

        # Create Q and R
        for node in transient_nodes:
            i = node_ordering[node]

            # Find the successors of the node
            successors = set(self.graph.successors(node))

            # Compute the probability of each outgoing edge
            probabilities = dict([(s, self._get_edge_probability(node, s))
                                  for s in successors])
            sum_probabilities = sum(probabilities.values())

            # Add the normalized probability to Q or R
            for successor, probability in probabilities.items():
                j = node_ordering[successor]
                transition = probability / sum_probabilities
                if successor in transient_nodes:
                    Q[i, j] = transition
                else:
                    R[i, j - len(transient_nodes)] = transition

        return Q, R

    def _create_N(self) -> np.ndarray:
        N = np.identity(len(self.Q))
        Q_power_i = np.identity(len(self.Q))
        while np.linalg.norm(Q_power_i, ord=2) > 1e-5:
            Q_power_i = Q_power_i @ self.Q
            N += Q_power_i
        return N

    def _create_B(self) -> np.ndarray:
        B = self.N @ self.R
        return B

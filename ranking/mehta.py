import numpy as np
from attack_graph import StateAttackGraph
from ranking.ranking import RankingMethod
from scipy.sparse import csr_matrix
from typing import Dict


class PageRankMethod(RankingMethod):
    def __init__(self, graph: StateAttackGraph, d: float = 0.85):
        super().__init__(list(graph.exploits))
        self.graph = graph
        self.d = d

    def _compute_normalized_adjacency_matrix(self) -> np.ndarray:
        N = self.graph.number_of_nodes()
        Z = np.zeros((N, N))
        node_ordering = self.graph.get_node_ordering()
        for j in self.graph.nodes():
            # Add an edge with probability 1-d to the starting node
            Z[node_ordering[0], node_ordering[j]] = 1 - self.d

            probabilities = dict([(i, self.graph.get_edge_probability(j, i))
                                  for i in self.graph.successors(j)])

            if len(probabilities) == 0:
                # The node is a goal node, add an edge to itself
                Z[node_ordering[0], node_ordering[j]] = self.d
            else:
                normalization_constant = sum(list(probabilities.values()))
                for i, probability in probabilities.items():
                    Z[node_ordering[i], node_ordering[
                        j]] = self.d * probability / normalization_constant
        return Z

    def _compute_rank_vector(self,
                             Z: np.ndarray,
                             eps: float = 1e-4) -> np.ndarray:
        R = np.ones(
            self.graph.number_of_nodes()) / self.graph.number_of_nodes()
        distance = np.inf
        while distance > eps:
            new_R = Z.dot(R)
            distance = np.linalg.norm(R - new_R, ord=2)
            R = new_R
        return R

    def apply(self) -> Dict[int, float]:
        Z = self._compute_normalized_adjacency_matrix()
        R = self._compute_rank_vector(Z)

        ids_nodes = list(self.graph.nodes)
        return dict([(ids_nodes[i], float(R[i])) for i in range(len(R))])

    def get_score(self) -> float:
        ranks = self.apply()
        score = 0
        for node in self.graph.goal_nodes:
            score += ranks[node]
        return score

    def get_score_with_exploit_removed(self, id_exploit: int) -> float:
        pruned_graph = self._get_pruned_graph(self.graph, id_exploit)

        if pruned_graph is None:
            return float("-inf")
        else:
            score = PageRankMethod(pruned_graph).get_score()
            return score


class KuehlmannMethod(RankingMethod):
    def __init__(self, graph: StateAttackGraph, eta: float = 0.85):
        super().__init__(list(graph.exploits))
        self.graph = graph
        self.eta = eta

    def _compute_transition_probability_matrix(self) -> csr_matrix:
        N = self.graph.number_of_nodes()
        P = np.zeros((N, N))
        node_ordering = self.graph.get_node_ordering()
        for i in self.graph.nodes():
            probabilities = dict([(j, self.graph.get_edge_probability(i, j))
                                  for j in self.graph.successors(i)])
            normalization_constant = sum(list(probabilities.values()))
            for j, probability in probabilities.items():
                P[node_ordering[i],
                  node_ordering[j]] = probability / normalization_constant
        P = csr_matrix(P)
        return P

    def apply(self, max_m: int = 100) -> Dict[int, float]:
        P = self._compute_transition_probability_matrix()
        s = np.zeros(self.graph.number_of_nodes())
        s[0] = 1

        powers_eta = np.power(self.eta, np.arange(max_m + 1))
        r = np.zeros(self.graph.number_of_nodes())

        power_P = csr_matrix(np.identity(self.graph.number_of_nodes()))
        current_sum = s
        m = 1
        stop = False
        while m <= max_m and not stop:
            power_P = csr_matrix(power_P.dot(P))
            to_add = csr_matrix(power_P.dot(s))
            current_sum += to_add.toarray().flatten()
            r += powers_eta[m] * current_sum

            m += 1
            stop = to_add.sum() < 1e-15

        r *= (1 - self.eta) / self.eta
        values = dict([(list(self.graph.nodes)[i], float(r[i]))
                       for i in range(len(r))])
        return values

    def get_score(self) -> float:
        ranks = self.apply()
        score = 0
        for node in self.graph.goal_nodes:
            score += ranks[node]
        return score

    def get_score_with_exploit_removed(self, id_exploit: int) -> float:
        pruned_graph = self._get_pruned_graph(self.graph, id_exploit)

        if pruned_graph is None:
            return float("-inf")
        else:
            score = KuehlmannMethod(pruned_graph).get_score()
            return score

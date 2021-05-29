import numpy as np
from attack_graph import StateAttackGraph
from ranking.ranking import RankingMethod
from typing import Dict


class PageRankMethod(RankingMethod):
    def __init__(self, graph: StateAttackGraph, d: float = 0.85):
        super().__init__(graph)
        self.d = d

    def _compute_normalized_adjacency_matrix(self):
        Z = np.zeros(
            (self.graph.number_of_nodes(), self.graph.number_of_nodes()))
        node_ordering = self.graph.get_node_ordering()
        for i in self.graph.nodes():
            for j in self.graph.predecessors(i):
                Z[node_ordering[i],
                  node_ordering[j]] = 1 / len(list(self.graph.successors(j)))

        # In the paper, Mehta et al. indicated that a link with probability
        # 1-d should be added from each state toward the initial state.
        toward_initial_state = np.zeros(
            (self.graph.number_of_nodes(), self.graph.number_of_nodes()))
        toward_initial_state[0, :] = 1 - self.d

        return toward_initial_state + self.d * Z

    def _compute_rank_vector(self, Z: np.array, eps: float = 1e-4):
        R = np.ones(
            self.graph.number_of_nodes()) / self.graph.number_of_nodes()
        distance = np.inf
        while distance > eps:
            new_R = Z.dot(R)
            distance = np.linalg.norm(R - new_R, ord=2)
            R = new_R
        R[0] = 0
        return R

    def apply(self) -> Dict[int, float]:
        Z = self._compute_normalized_adjacency_matrix()
        R = self._compute_rank_vector(Z)

        ids_nodes = list(self.graph.nodes)
        return dict([(ids_nodes[i], float(R[i])) for i in range(len(R))])

    def _get_score(self) -> float:
        score = sum(list(self.apply().values()))
        return score

    def _get_score_for_graph(self, graph: StateAttackGraph) -> float:
        return PageRankMethod(graph)._get_score()


class KuehlmannMethod(RankingMethod):
    def __init__(self, graph: StateAttackGraph, eta: float = 0.85):
        super().__init__(graph)
        self.eta = eta

    def _compute_transition_probability_matrix(self):
        P = np.zeros(
            (self.graph.number_of_nodes(), self.graph.number_of_nodes()))
        node_ordering = self.graph.get_node_ordering()
        for i in self.graph.nodes():
            for j in self.graph.predecessors(i):
                P[node_ordering[i],
                  node_ordering[j]] = 1 / len(list(self.graph.successors(j)))
        return P

    def apply(self, max_m: int = 100) -> Dict[int, float]:
        P = self._compute_transition_probability_matrix()
        s = np.zeros(self.graph.number_of_nodes())
        s[0] = 1

        powers_P = np.zeros((max_m + 1, self.graph.number_of_nodes(),
                             self.graph.number_of_nodes()))
        powers_eta = np.power(self.eta, np.arange(1, max_m + 1))

        r = np.zeros(self.graph.number_of_nodes())

        powers_P[0] = P
        for m in range(1, max_m + 1):
            powers_P[m] = P.dot(powers_P[m - 1])
            r += powers_eta[m - 1] * np.sum(powers_P[:m + 1], axis=0).dot(s)

        r *= (1 - self.eta) / self.eta

        ids_nodes = list(self.graph.nodes)
        return dict([(ids_nodes[i], float(r[i])) for i in range(len(r))])

    def _get_score(self) -> float:
        score = sum(list(self.apply().values()))
        return score

    def _get_score_for_graph(self, graph: StateAttackGraph) -> float:
        return KuehlmannMethod(graph)._get_score()

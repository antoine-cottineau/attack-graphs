import numpy as np
from attack_graph import StateAttackGraph
from ranking.ranking import RankingMethod
from scipy.sparse import csr_matrix
from typing import Dict


class PageRankMethod(RankingMethod):
    def __init__(self, graph: StateAttackGraph, d: float = 0.85):
        super().__init__(graph)
        self.d = d

    def _compute_normalized_adjacency_matrix(self) -> np.ndarray:
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

    def _compute_rank_vector(self,
                             Z: np.array,
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

    def _get_score(self) -> float:
        score = sum(list(self.apply().values()))
        return score

    def _get_score_for_graph(self, graph: StateAttackGraph) -> float:
        return PageRankMethod(graph)._get_score()


class KuehlmannMethod(RankingMethod):
    def __init__(self, graph: StateAttackGraph, eta: float = 0.85):
        super().__init__(graph)
        self.eta = eta

    def _compute_transition_probability_matrix(self) -> csr_matrix:
        P = np.zeros(
            (self.graph.number_of_nodes(), self.graph.number_of_nodes()))
        node_ordering = self.graph.get_node_ordering()
        for i in self.graph.nodes():
            for j in self.graph.predecessors(i):
                P[node_ordering[i],
                  node_ordering[j]] = 1 / len(list(self.graph.successors(j)))
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

    def _get_score(self) -> float:
        score = sum(list(self.apply().values()))
        return score

    def _get_score_for_graph(self, graph: StateAttackGraph) -> float:
        return KuehlmannMethod(graph)._get_score()

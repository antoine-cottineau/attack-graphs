import numpy as np

from attack_graph import AttackGraph


class RankingMethod:
    def __init__(self, ag: AttackGraph):
        self.ag = ag

    def update_ranking_scores(self, ranking_scores: np.array):
        # Find the minimum and maximum of the ranking
        min_ = ranking_scores[0]
        max_ = ranking_scores[0]

        # Add a ranking score to each vertex in the graph
        for i, node in self.ag.nodes(data=True):
            node["ranking_score"] = ranking_scores[i]

            # Update the extrema
            if ranking_scores[i] < min_:
                min_ = ranking_scores[i]
            elif ranking_scores[i] > max_:
                max_ = ranking_scores[i]

        # Add the extrema to the attack graph
        self.ag.ranking_min = min_
        self.ag.ranking_max = max_

        # Update the colors of the nodes
        self.ag.update_colors_based_on_ranking()


class PageRankMethod(RankingMethod):
    def __init__(self, ag: AttackGraph, d: float = 0.85):
        super().__init__(ag)
        self.d = d

    def compute_normalized_adjacency_matrix(self):
        Z = np.zeros((self.ag.number_of_nodes(), self.ag.number_of_nodes()))
        for i in self.ag.nodes():
            for j in self.ag.predecessors(i):
                Z[i, j] = 1 / len(list(self.ag.successors(j)))

        # In the paper, Mehta et al. indicated that a link with probability
        # 1-d should be added from each state toward the initial state.
        toward_initial_state = np.zeros(
            (self.ag.number_of_nodes(), self.ag.number_of_nodes()))
        toward_initial_state[0, :] = 1 - self.d

        return toward_initial_state + self.d * Z

    def compute_page_rank_vector(self, Z: np.array, eps: float = 1e-4):
        R = np.ones(self.ag.number_of_nodes()) / self.ag.number_of_nodes()
        distance = np.inf
        while distance > eps:
            new_R = Z.dot(R)
            distance = np.linalg.norm(R - new_R, ord=2)
            R = new_R
        R[0] = 0
        return R

    def apply(self):
        Z = self.compute_normalized_adjacency_matrix()
        R = self.compute_page_rank_vector(Z)

        self.update_ranking_scores(R)


class KuehlmannMethod(RankingMethod):
    def __init__(self, ag: AttackGraph, eta: float = 0.85):
        super().__init__(ag)
        self.eta = eta

    def compute_transition_probability_matrix(self):
        P = np.zeros((self.ag.number_of_nodes(), self.ag.number_of_nodes()))
        for i in self.ag.nodes():
            for j in self.ag.predecessors(i):
                P[i, j] = 1 / len(list(self.ag.successors(j)))
        return P

    def apply(self, max_m: int = 100):
        P = self.compute_transition_probability_matrix()
        s = np.zeros(self.ag.number_of_nodes())
        s[0] = 1

        powers_P = np.zeros(
            (max_m + 1, self.ag.number_of_nodes(), self.ag.number_of_nodes()))
        powers_eta = np.power(self.eta, np.arange(1, max_m + 1))

        r = np.zeros(self.ag.number_of_nodes())

        powers_P[0] = P
        for m in range(1, max_m + 1):
            powers_P[m] = P.dot(powers_P[m - 1])
            r += powers_eta[m - 1] * np.sum(powers_P[:m + 1], axis=0).dot(s)

        r *= (1 - self.eta) / self.eta

        self.update_ranking_scores(r)

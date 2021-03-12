from attack_graph import AttackGraph
import numpy as np


class RankingMethod:
    """
    Base class for ranking methods.

    :param AttackGraph ag: The attack graph to consider.
    """
    def __init__(self, ag: AttackGraph):
        self.ag = ag

    def update_ranking_scores(self, ranking_scores: np.array):
        """
        Add a new attribute called ranking_score to the states of the attack
        graph.
        The attack graph will also receive two attributes namely ranking_min
        and ranking_max.

        :param ndarray ranking_scores: The array of the rankings.
        """
        # Find the minimum and maximum of the rankings
        min_ = ranking_scores[0]
        max_ = ranking_scores[0]

        # Add a ranking score to each vertex in the graph
        for i in range(self.ag.N):
            state = self.ag.states[i]
            state.ranking_score = ranking_scores[i]

            # Update the extrema
            if ranking_scores[i] < min_:
                min_ = ranking_scores[i]
            elif ranking_scores[i] > max_:
                max_ = ranking_scores[i]

        # Add the extrema to the attack graph
        self.ag.ranking_min = min_
        self.ag.ranking_max = max_


class PageRankMethod(RankingMethod):
    """
    Implementation of the first method introduced by Mehta et al. in Ranking
    Attack Graphs (2006).
    It was inspired by Google's PageRank.

    :param AttackGraph ag: The attack graph to consider.
    :param float d: The damping factor.
    """
    def __init__(self, ag: AttackGraph, d=0.85):
        RankingMethod.__init__(self, ag)
        self.d = d

    def compute_normalized_adjacency_matrix(self):
        """
        Compute the normalized adjacency matrix Z.

        :return ndarray Z: The normalized adjacency matrix.
        """
        Z = np.zeros((self.ag.N, self.ag.N))
        for i in range(self.ag.N):
            state_i = self.ag.states[i]
            for j in state_i.in_:
                state_j = self.ag.states[j]
                Z[i, j] = 1 / len(state_j.out)

        # In the paper, Mehta et al. indicated that a link with probability
        # 1-d should be added from each state toward the initial state.
        toward_initial_state = np.zeros((self.ag.N, self.ag.N))
        toward_initial_state[0, :] = 1 - self.d

        return toward_initial_state + self.d * Z

    def compute_page_rank_vector(self, Z, eps=1e-4):
        """
        Compute the PageRank vector R.

        :param ndarray Z: The normalized adjacency matrix.
        :param float eps: The precision for the convergence.
        :return ndarray R: The PageRank vector.
        """
        R = np.ones(self.ag.N) / self.ag.N
        distance = np.inf
        while distance > eps:
            new_R = Z.dot(R)
            distance = np.linalg.norm(R - new_R, ord=2)
            R = new_R
        R[0] = 0
        return R

    def apply(self):
        """
        Apply the method.
        The states of the provided attack graph will receive a new attribute
        called ranking_score.
        The attack graph will also receive two attributes namely ranking_min
        and ranking_max.
        """
        Z = self.compute_normalized_adjacency_matrix()
        R = self.compute_page_rank_vector(Z)

        self.update_ranking_scores(R)


class KuehlmannMethod(RankingMethod):
    """
    Implementation of the second method introduced by Mehta et al. in Ranking
    Attack Graphs (2006).
    It was inspired by a technique introduced by Kuehlmann et al.

    :param AttackGraph ag: The attack graph to consider.
    :param float eta: The eta used in Kuehlmann's technique.
    """
    def __init__(self, ag: AttackGraph, eta=0.85):
        RankingMethod.__init__(self, ag)
        self.eta = eta

    def compute_transition_probability_matrix(self):
        """
        Compute the transition probability matrix P.

        :return ndarray P: The transition probability matrix.
        """
        P = np.zeros((self.ag.N, self.ag.N))
        for i in range(self.ag.N):
            state_i = self.ag.states[i]
            for j in state_i.in_:
                state_j = self.ag.states[j]
                P[i, j] = 1 / len(state_j.out)
        return P

    def apply(self, max_m=100):
        """
        Apply the method.
        The states of the provided attack graph will receive a new attribute
        called ranking_score.
        The attack graph will also receive two attributes namely ranking_min
        and ranking_max.
        """
        P = self.compute_transition_probability_matrix()
        s = np.zeros(self.ag.N)
        s[0] = 1

        powers_P = np.zeros((max_m + 1, self.ag.N, self.ag.N))
        powers_eta = np.power(self.eta, np.arange(1, max_m + 1))

        r = np.zeros(self.ag.N)

        powers_P[0] = P
        for m in range(1, max_m + 1):
            powers_P[m] = P.dot(powers_P[m - 1])
            r += powers_eta[m - 1] * np.sum(powers_P[:m + 1], axis=0).dot(s)

        r *= (1 - self.eta) / self.eta

        self.update_ranking_scores(r)

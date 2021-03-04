from mulval import MulvalAttackGraph, MulvalVertexType
import numpy as np


class PageRankMethod:
    """
    Implementation of the first method introduced by Mehta et al. in Ranking
    Attack Graphs (2006).
    It was inspired by Google's PageRank.

    :param MulvalAttackGraph mag: The attack graph to consider.
    :param float d: The damping factor.
    """
    def __init__(self, mag: MulvalAttackGraph, d=0.85):
        self.mag = mag
        self.d = d

    def compute_normalized_adjacency_matrix(self):
        """
        Compute the normalized adjacency matrix Z.

        :return ndarray Z: The normalized adjacency matrix.
        """
        Z = np.zeros((self.mag.N, self.mag.N))
        for i in range(self.mag.N):
            vertex_i = self.mag.vertices[self.mag.ids[i]]
            for id_j in vertex_i.in_:
                vertex_j = self.mag.vertices[id_j]
                j = self.mag.ids.index(id_j)
                Z[i, j] = 1 / len(vertex_j.out)
        Z = (1 - self.d) / self.mag.N + self.d * Z
        return Z

    def compute_page_rank_vector(self, Z, eps=1e-4):
        """
        Compute the PageRank vector R.

        :param ndarray Z: The normalized adjacency matrix.
        :param float eps: The precision for the convergence.
        :return ndarray R: The PageRank vector.
        """
        R = np.ones(self.mag.N) / self.mag.N
        distance = np.inf
        while distance > eps:
            new_R = Z.dot(R)
            distance = np.linalg.norm(R - new_R, ord=2)
            R = new_R
        return R

    def apply(self):
        """
        Apply the method.
        The vertices of the provided attack graph will receive a new attribute
        called ranking_score.
        """
        Z = self.compute_normalized_adjacency_matrix()
        R = self.compute_page_rank_vector(Z)

        # Add a ranking score to each vertex in the graph
        for i in range(self.mag.N):
            vertex = self.mag.vertices[self.mag.ids[i]]
            vertex.ranking_score = R[i]


class KuehlmannMethod:
    """
    Implementation of the second method introduced by Mehta et al. in Ranking
    Attack Graphs (2006).
    It was inspired by a technique introduced by Kuehlmann et al.

    :param MulvalAttackGraph mag: The attack graph to consider.
    :param float eta: The eta used in Kuehlmann's technique.
    """
    def __init__(self, mag: MulvalAttackGraph, eta=0.85):
        self.mag = mag
        self.eta = eta

    def compute_transition_probability_matrix(self):
        """
        Compute the transition probability matrix P.

        :return ndarray P: The transition probability matrix.
        """
        P = np.zeros((self.mag.N, self.mag.N))
        for i in range(self.mag.N):
            vertex_i = self.mag.vertices[self.mag.ids[i]]
            for id_j in vertex_i.in_:
                vertex_j = self.mag.vertices[id_j]
                j = self.mag.ids.index(id_j)
                P[i, j] = 1 / len(vertex_j.out)
        return P

    def compute_start_vector(self):
        """
        Compute the start vector s.

        :return ndarray s: The start vector.
        """
        s = np.zeros(self.mag.N)
        for i in range(self.mag.N):
            vertex = self.mag.vertices[self.mag.ids[i]]
            if vertex.type_ == MulvalVertexType.LEAF:
                s[i] = 1
        return s

    def apply(self, max_m=100):
        """
        Apply the method.
        The vertices of the provided attack graph will receive a new attribute
        called ranking_score.
        """
        P = self.compute_transition_probability_matrix()
        s = self.compute_start_vector()

        powers_P = np.zeros((max_m + 1, self.mag.N, self.mag.N))
        powers_eta = np.power(self.eta, np.arange(1, max_m + 1))

        r = np.zeros(self.mag.N)

        powers_P[0] = P
        for m in range(1, max_m + 1):
            powers_P[m] = P.dot(powers_P[m - 1])
            r += powers_eta[m - 1] * np.sum(powers_P[:m + 1], axis=0).dot(s)

        r *= (1 - self.eta) / self.eta

        # Add a ranking score to each vertex in the graph
        for i in range(self.mag.N):
            vertex = self.mag.vertices[self.mag.ids[i]]
            vertex.ranking_score = r[i]

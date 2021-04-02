import numpy as np
import scipy.sparse as sps

from attack_graph import AttackGraph
from embedding import Embedding


class Hope(Embedding):
    """
    Class that runs the HOPE algorithm invented by Ou et al.

    :param ag AttackGraph: The attack graph to perform HOPE on.
    :param int dim_embedding: The dimension of the embedding.
    :param str measurement: A string corresponding to which measurement to use.
    It can either be cn (Common Neighbours), katz (Katz), pagerank
    (Personalized Pagerank) or aa (Adamic-Adar).
    """
    def __init__(self, ag: AttackGraph, dim_embedding: int, measurement: str):
        super().__init__(ag, dim_embedding)

        self.measurement = measurement

    def run(self):
        """
        Run the algorithm
        """
        self.createAdjacencyMatrix()
        self.createS(self.measurement)

        U, sigmas, Vt = sps.linalg.svds(self.S, k=int(self.dim_embedding / 2))
        sigmas = np.diagflat(np.sqrt(sigmas))
        left_embedding = np.dot(U, sigmas)
        right_embedding = np.dot(Vt.T, sigmas)

        self.embeddings = np.concatenate([left_embedding, right_embedding],
                                         axis=1)

    def createAdjacencyMatrix(self):
        """
        Create the directed adjacency matrix. If there is an edge between state
        i and state j, then A[i, j] = 1.
        """
        A = np.zeros((self.ag.N, self.ag.N))
        for state in self.ag.states:
            for id_neighbour in [*state.out]:
                A[state.id_, id_neighbour] = 1
        self.A = sps.csc_matrix(A)

    def createS(self, measurement: str):
        """
        Create the high-order proximity matrix S.

        :param str measurement: A string corresponding to which measurement to
        use. It can either be cn (Common Neighbours), katz (Katz), pagerank
        (Personalized Pagerank) or aa (Adamic-Adar).
        """
        if measurement == "cn":
            self.createSWithCommonNeighbours()
        elif measurement == "katz":
            self.createSWithKatz()
        elif measurement == "pagerank":
            self.createSWithPagerank()
        elif measurement == "aa":
            self.createSWithAdamicAdar()

    def createSWithCommonNeighbours(self):
        """
        Create the high-order proximity matrix by using Common Neighbours as a
        measurement.
        """
        self.S = self.A.dot(self.A)

    def createSWithKatz(self, beta=0.1):
        """
        Create the high-order proximity matrix by using Katz as a measurement.
        """
        Mg = sps.identity(self.ag.N) - beta * self.A
        Ml = beta * self.A

        self.S = sps.linalg.inv(Mg).dot(Ml)

    def createSWithPagerank(self, alpha=0.5):
        """
        Create the high-order proximity matrix by using Personalized Pagerank
        as a measurement.
        """
        sum_ = self.A.sum(axis=0)
        sum_ = np.where(sum_ == 0, 1, sum_)
        P = sps.csc_matrix(self.A / sum_)

        Mg = sps.identity(self.ag.N) - alpha * P

        self.S = (1 - alpha) * sps.linalg.inv(Mg)

    def createSWithAdamicAdar(self):
        """
        Create the high-order proximity matrix by using Adamic-Adar as a
        measurement.
        """
        D = np.zeros((self.ag.N, self.ag.N))
        for i in range(self.ag.N):
            D[i, i] = 1 / (self.A[i].sum() + self.A[:, i].sum())
        D = sps.csc_matrix(D)

        self.S = self.A.dot(D).dot(self.A)

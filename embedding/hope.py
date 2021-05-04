import numpy as np
import scipy.sparse as sps

from attack_graph import AttackGraph
from embedding.embedding import EmbeddingMethod


class Hope(EmbeddingMethod):
    def __init__(self,
                 ag: AttackGraph,
                 dim_embedding: int,
                 measurement: str = "cn"):
        super().__init__(ag, dim_embedding)

        self.measurement = measurement

    def embed(self):
        self.A = self.ag.compute_adjacency_matrix().astype("f")
        self.createS(self.measurement)

        U, sigmas, Vt = sps.linalg.svds(self.S, k=int(self.dim_embedding / 2))
        sigmas = np.diagflat(np.sqrt(sigmas))
        left_embedding = np.dot(U, sigmas)
        right_embedding = np.dot(Vt.T, sigmas)

        self.embedding = np.concatenate([left_embedding, right_embedding],
                                        axis=1)

    def createS(self, measurement: str):
        if measurement == "cn":
            self.createSWithCommonNeighbours()
        elif measurement == "katz":
            self.createSWithKatz()
        elif measurement == "pagerank":
            self.createSWithPagerank()
        elif measurement == "aa":
            self.createSWithAdamicAdar()

    def createSWithCommonNeighbours(self):
        self.S = self.A.dot(self.A)

    def createSWithKatz(self, beta=0.1):
        Mg = sps.identity(self.ag.number_of_nodes()) - beta * self.A
        Ml = beta * self.A

        self.S = sps.linalg.inv(Mg).dot(Ml)

    def createSWithPagerank(self, alpha=0.5):
        sum_ = self.A.sum(axis=0)
        sum_ = np.where(sum_ == 0, 1, sum_)
        P = sps.csc_matrix(self.A / sum_)

        Mg = sps.identity(self.ag.number_of_nodes()) - alpha * P

        self.S = (1 - alpha) * sps.linalg.inv(Mg)

    def createSWithAdamicAdar(self):
        D = np.zeros((self.ag.number_of_nodes(), self.ag.number_of_nodes()))
        for i in range(self.ag.number_of_nodes()):
            D[i, i] = 1 / (self.A[i].sum() + self.A[:, i].sum())
        D = sps.csc_matrix(D)

        self.S = self.A.dot(D).dot(self.A)

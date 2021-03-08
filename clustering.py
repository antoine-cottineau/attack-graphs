from attack_graph import AttackGraph
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans


class Spectral:
    """
    Base class for the clustering techniques developped in A Spectral
    Clustering Approach To Finding Communities in Graphs by White and Smyth in
    2005.

    :param AttackGraph ag: The attack graph on which clustering is applied.
    """
    def __init__(self, ag: AttackGraph):
        self.ag = ag

        self.W = np.zeros((self.ag.N, self.ag.N))
        self.D = np.zeros((self.ag.N, self.ag.N))
        self.inverse_D = np.zeros((self.ag.N, self.ag.N))

    def create_weight_matrix(self):
        """
        Create the weight matrix W.
        W is a sparse matrix as most of the nodes aren't linked.
        """
        W = np.zeros((self.ag.N, self.ag.N))

        for state in self.ag.states:
            for id_neighbour in state.in_:
                neighbour = self.ag.states[id_neighbour]
                W[state.id_, neighbour.id_] = 1
                W[neighbour.id_, state.id_] = 1

        self.W = csr_matrix(W)

    def create_transition_matrix(self):
        """
        Create the transition matrix and its inverse.
        """
        D = np.zeros((self.ag.N, self.ag.N))
        inverse_D = np.zeros((self.ag.N, self.ag.N))

        for i in range(self.ag.N):
            sum_ = np.sum(self.W[i])
            D[i, i] = sum_
            inverse_D[i, i] = 1 / sum_

        self.D = csr_matrix(D)
        self.inverse_D = csr_matrix(inverse_D)

    def compute_eigenvector_matrix(self, K: int):
        """
        Compute the top K-1 eigenvectors of the transition matrix M=D^-1*W.

        :param int K: The parameter K in the paper.
        :return ndarray eigenvectors: The top K-1 eigenvectors of the
        transition matrix.
        """
        transition_matrix = self.inverse_D.dot(self.W)
        eigenvectors = eigs(transition_matrix, k=K,
                            which="LR")[1].astype("float64")

        # Remove the trivial all-ones eigenvector
        for i in range(K):
            eigenvector = eigenvectors[:, i]
            if np.linalg.norm(eigenvector - eigenvector[0], ord=2) < 1e-4:
                break
        eigenvectors = np.delete(eigenvectors, i, axis=1)

        return eigenvectors[:, :K - 1]

    def compute_assignment_matrix(self, partition: list, k: int):
        """
        Compute the assignment matrix X.

        :param list partition: A list of integers giving the id of the cluster
        corresponding to each node.
        :param int k: The number of clusters.
        :return ndarray X: The assignment matrix X.
        """
        X = np.zeros((self.ag.N, k))
        for i in range(self.ag.N):
            X[i, partition[i]] = 1
        return X

    def compute_Q_function(self, partition: list, k: int):
        """
        Compute the Q function for a given partition.

        :param list partition: A list of integers giving the id of the cluster
        corresponding to each node.
        :param int k: The number of clusters.
        :return int: The value of the Q function.
        """
        X = self.compute_assignment_matrix(partition, k)
        vol_G = self.W.sum()

        # Compute the weighted degree of each vertex
        d = self.D.diagonal()
        D = np.outer(d, d)

        return np.trace(csr_matrix.dot(X.T, vol_G * self.W - D).dot(X))


class Spectral1(Spectral):
    """
    Implementation of the first algorithm introduced by White and Smyth.

    :param AttackGraph ag: The attack graph on which clustering is applied.
    """
    def __init__(self, ag: AttackGraph):
        Spectral.__init__(self, ag)

    def apply_for_k(self, eigenvectors: np.array, k: int):
        """
        Get the partition for a given k.

        :param ndarray eigenvectors: The top eigenvectors of U_K.
        :param int k: The number of desired clusters.
        :return list partition: A list of integers giving the id of the cluster
        corresponding to each node.
        """
        U_k = eigenvectors[:, :k - 1]

        # Normalize U_k
        norm = np.linalg.norm(U_k, axis=1, ord=2)
        U_k = (U_k.T / norm).T

        # Apply k-means on the rows of U_k
        k_means = KMeans(n_clusters=k)
        partition = k_means.fit_predict(U_k)

        return partition

    def apply(self, K: int):
        """
        Apply the algorithm and add a new attribute called id_cluster to every
        state.

        :param int K: The maximal number of clusters.
        """
        self.create_weight_matrix()
        self.create_transition_matrix()

        eigenvectors = self.compute_eigenvector_matrix(K)
        best_score = -np.inf
        best_partition = None

        for k in range(2, K + 1):
            partition = self.apply_for_k(eigenvectors, k)
            score = self.compute_Q_function(partition, k)
            if score > best_score:
                best_score = score
                best_partition = partition

        for i in range(self.ag.N):
            self.ag.states[i].id_cluster = best_partition[i]

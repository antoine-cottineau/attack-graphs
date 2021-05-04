from clustering.clustering import ClusteringMethod
import numpy as np

from attack_graph import AttackGraph
from clustering.space_metrics import score_with_Q_function
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans


class SpectralMethod(ClusteringMethod):
    def __init__(self, ag: AttackGraph):
        self.ag = ag

        self.W = ag.compute_adjacency_matrix(keep_directed=False)
        self.D = np.zeros((ag.number_of_nodes(), ag.number_of_nodes()))
        self.inverse_D = np.zeros((ag.number_of_nodes(), ag.number_of_nodes()))

        self.create_transition_matrix()

    def create_transition_matrix(self):
        D = np.zeros((self.ag.number_of_nodes(), self.ag.number_of_nodes()))
        inverse_D = np.zeros(
            (self.ag.number_of_nodes(), self.ag.number_of_nodes()))

        node_mapping = self.ag.get_node_mapping()

        for i in self.ag.nodes():
            pos = node_mapping[i]
            sum_ = np.sum(self.W[pos])
            D[pos, pos] = sum_
            inverse_D[pos, pos] = 1 / sum_

        self.D = csr_matrix(D)
        self.inverse_D = csr_matrix(inverse_D)

    def get_real_K(self, K: int):
        # We ideally want to compute the top K - 1 eigenvectors of the matrix
        # because one of these eigenvectors is the trivial all-ones
        # So, we need to compute K eigenvectors
        # However, if K >= N - 1, the function will crash
        # Thus, the real k we use in the eigs function is equal to
        # min(K, N - 2)
        return min(K, self.ag.number_of_nodes() - 2)

    def compute_eigenvector_matrix(self, K: int):
        transition_matrix = self.inverse_D.dot(self.W)

        eigenvectors = eigs(transition_matrix, k=K,
                            which="LR")[1].astype("float64")

        # Remove the trivial all-ones eigenvector
        for i in range(K):
            eigenvector = eigenvectors[:, i]
            if np.linalg.norm(eigenvector - eigenvector[0], ord=2) < 1e-4:
                break
        eigenvectors = np.delete(eigenvectors, i, axis=1)

        return eigenvectors

    def compute_Q_function(self, X: np.array, labels: list):
        return score_with_Q_function(X, labels, self.W, self.D)

    @staticmethod
    def extract_and_normalize_eigenvectors(eigenvectors: np.array, k: int):
        U_k = eigenvectors[:, :k - 1]

        # Normalize U_k
        norm = np.linalg.norm(U_k, axis=1, ord=2)
        U_k = (U_k.T / norm).T

        return U_k


class Spectral1(SpectralMethod):
    def __init__(self, ag: AttackGraph, K: int):
        super().__init__(ag)

        self.K = K

    @staticmethod
    def apply_for_k(eigenvectors: np.array, k: int):
        U_k = SpectralMethod.extract_and_normalize_eigenvectors(
            eigenvectors, k)

        # Apply k-means on the rows of U_k
        k_means = KMeans(n_clusters=k)
        labels = k_means.fit_predict(U_k)

        return U_k, labels

    def cluster(self):
        real_K = self.get_real_K(self.K)

        eigenvectors = self.compute_eigenvector_matrix(real_K)
        best_score = -np.inf
        best_labels = None

        for k in range(2, real_K + 1):
            U_k, labels = Spectral1.apply_for_k(eigenvectors, k)
            score = self.compute_Q_function(U_k, labels)
            if score > best_score:
                best_score = score
                best_labels = labels

        self.update_clusters(best_labels)


class Spectral2(SpectralMethod):
    def __init__(self, ag: AttackGraph, K: int, k_min: int = 2):
        super().__init__(ag)

        self.K = K
        self.k_min = k_min

    def apply_for_k(self, eigenvectors: np.array, k: int, P: list):
        P_new = P.copy()
        ids_clusters = set(P)
        node_assignments = {
            c: np.array(
                [i for i in range(self.ag.number_of_nodes()) if P[i] == c])
            for c in ids_clusters
        }
        has_updated = False

        for c in ids_clusters:
            U_k = SpectralMethod.extract_and_normalize_eigenvectors(
                eigenvectors, k)
            U_k_c = U_k[node_assignments[c]]

            sub_partition = KMeans(n_clusters=2).fit_predict(U_k_c)
            ids_states_in_new_cluster = [
                i for i in range(len(sub_partition)) if sub_partition[i] == 0
            ]

            P_prime = P.copy()
            new_id_cluster = P.max() + 1
            ids_states_to_update = node_assignments[c][
                ids_states_in_new_cluster]
            P_prime[ids_states_to_update] = new_id_cluster

            # Check if the modification improves the value of the Q function
            if self.compute_Q_function(U_k, P_prime) > self.compute_Q_function(
                    U_k, P):
                P_new = P_prime
                has_updated = True
                break

        return P_new, has_updated

    def cluster(self):
        real_K = self.get_real_K(self.K)

        eigenvectors = self.compute_eigenvector_matrix(real_K)

        k = self.k_min
        P = np.zeros(self.ag.number_of_nodes(), dtype=int)
        if k > 1:
            U_k = SpectralMethod.extract_and_normalize_eigenvectors(
                eigenvectors, k)
            P = KMeans(n_clusters=k).fit_predict(U_k)

        k += 1
        possible_splits = True
        while k <= real_K and possible_splits:
            P, has_updated = self.apply_for_k(eigenvectors, k, P)
            if has_updated:
                k += 1
            else:
                possible_splits = False

        self.update_clusters(P)

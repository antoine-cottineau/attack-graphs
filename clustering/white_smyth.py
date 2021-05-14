import numpy as np
from attack_graph import StateAttackGraph
from clustering.clustering import ClusteringMethod
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans


class SpectralMethod(ClusteringMethod):
    def __init__(self, ag: StateAttackGraph, K: int = 15):
        super().__init__(ag)

        self.K = min(K, self.ag.number_of_nodes() - 2)

    def cluster(self):
        self.create_W()
        self.create_D()
        self.create_M()
        self.compute_top_eigenvectors()

    def create_W(self):
        self.W = self.ag.compute_adjacency_matrix(keep_directed=False)

    def create_D(self):
        d = self.W.sum(axis=0).A1
        self.D: np.ndarray = np.diag(d)
        self.inverse_D: np.ndarray = np.diag(1 / d)

    def create_M(self):
        self.M: csr_matrix = csr_matrix(self.inverse_D).dot(self.W)

    def compute_top_eigenvectors(self):
        _, eigenvectors = eigs(self.M, k=self.K, which="LR")
        eigenvectors: np.ndarray = eigenvectors.astype("float")

        # Remove the all-ones eigenvector
        eigenvectors = eigenvectors[:, 1:]

        self.eigenvectors = eigenvectors

    def get_first_eigenvectors(self, k: int) -> np.ndarray:
        eigenvectors = self.eigenvectors[:, :k]
        norm = np.linalg.norm(eigenvectors, axis=1, ord=2)
        eigenvectors = (eigenvectors.T / norm).T
        return eigenvectors


class Spectral1(SpectralMethod):
    def cluster(self):
        super().cluster()

        best_node_assignment = None
        best_modularity = -np.inf

        for k in range(2, self.K + 1):
            eigenvectors = self.get_first_eigenvectors(k - 1)

            node_assignment = KMeans(n_clusters=k).fit_predict(eigenvectors)
            modularity = ClusteringMethod.modularity(self.ag, node_assignment)

            if modularity > best_modularity:
                best_node_assignment = node_assignment
                best_modularity = modularity

        self.update_clusters(best_node_assignment)


class Spectral2(SpectralMethod):
    def __init__(self, ag: StateAttackGraph, K: int = 15, k_min: int = 2):
        super().__init__(ag, K)

        self.k_min = k_min

    def cluster(self):
        super().cluster()

        k = self.k_min

        # Initialize the best node assignment
        if k == 1:
            best_node_assignment = np.zeros(self.ag.number_of_nodes(),
                                            dtype=int)
        else:
            eigenvectors = self.get_first_eigenvectors(k - 1)
            best_node_assignment = KMeans(
                n_clusters=k).fit_predict(eigenvectors)
        k += 1

        # Measure the current modularity
        modularity = ClusteringMethod.modularity(self.ag, best_node_assignment)

        # Main loop
        cant_improve = False
        while k <= self.K and not cant_improve:
            # Get the top k - 1 eigenvectors
            eigenvectors = self.get_first_eigenvectors(k - 1)

            # Try to split one of the clusters to improve the modularity
            has_improved = False
            clusters = np.unique(best_node_assignment)
            i_cluster = 0
            while i_cluster < len(clusters) and not has_improved:
                # Get the nodes in the cluster
                nodes_in_cluster = np.arange(self.ag.number_of_nodes())[
                    best_node_assignment == i_cluster]

                # If there is only one node in the cluster, we can't split it
                if len(nodes_in_cluster) == 1:
                    i_cluster += 1
                    continue

                # Perform K-means with 2 clusters on the involved rows of the
                # eigenvectors matrix
                cluster_eigenvectors = eigenvectors[nodes_in_cluster]
                sub_assignment = KMeans(
                    n_clusters=2).fit_predict(cluster_eigenvectors)

                # Assign a new cluster to the nodes of the first sub cluster
                nodes_in_sub_cluster_0 = nodes_in_cluster[sub_assignment == 0]
                new_node_assignment = np.copy(best_node_assignment)
                new_node_assignment[nodes_in_sub_cluster_0] = len(clusters)

                # Measure if there has been an improvement
                new_modularity = ClusteringMethod.modularity(
                    self.ag, new_node_assignment)

                if new_modularity > modularity:
                    # If there has been an improvement, we keep the changes
                    best_node_assignment = new_node_assignment
                    modularity = new_modularity
                    has_improved = True
                    k += 1
                else:
                    # Otherwise, we study the next cluster
                    i_cluster += 1

            # If there hasn't been any improvement during this iteration, we
            # stop the loop
            if not has_improved:
                cant_improve = True

        self.update_clusters(best_node_assignment)

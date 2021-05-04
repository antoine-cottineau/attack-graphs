import numpy as np
import clustering.space_metrics as space_metrics

from attack_graph import AttackGraph
from sklearn.cluster import KMeans
from typing import List


class ClusteringMethod:
    def __init__(self, ag: AttackGraph):
        self.ag = ag

        # By default, all the nodes are grouped into a unique cluster
        self.node_mapping = dict()
        for node in self.ag.nodes:
            self.node_mapping[node] = 0

    def cluster(self):
        pass

    def update_clusters(self, node_mapping: List[int]):
        self.node_mapping = dict()
        for i, node in enumerate(self.ag.nodes):
            self.node_mapping[node] = node_mapping[i]

    def evaluate_modularity(self) -> float:
        modularity = 0

        adjacency_matrix = self.ag.compute_adjacency_matrix()
        node_mapping = list(self.node_mapping.values())
        clusters = np.unique(node_mapping)

        sum_all_weights = adjacency_matrix.sum()

        for cluster in clusters:
            adj_matrix_cluster = adjacency_matrix[
                node_mapping == cluster][:, node_mapping == cluster]
            adj_matrix_out_cluster = adjacency_matrix[
                node_mapping != cluster][:, node_mapping != cluster]
            sum_within_cluster = adj_matrix_cluster.sum()
            sum_all_cluster = sum_all_weights - adj_matrix_out_cluster.sum()

            modularity += sum_within_cluster / sum_all_weights
            modularity -= (sum_all_cluster / sum_all_weights)**2

        return modularity

    @staticmethod
    def evaluate_space_clustering(X: np.array,
                                  k_min: int,
                                  k_max: int,
                                  metric: str = "silhouette") -> List[int]:
        best_score = -np.inf
        best_node_mapping = None

        for k in range(k_min, k_max + 1):
            # Apply k-means with k clusters
            node_mapping = KMeans(n_clusters=k).fit_predict(X)

            # Compute the score
            if metric == "silhouette":
                score = space_metrics.score_with_silhouette(X, node_mapping)
            elif metric == "ch":
                score = space_metrics.score_with_calinski_harabasz(
                    X, node_mapping)
            elif metric == "db":
                score = space_metrics.score_with_davies_bouldin(
                    X, node_mapping)
            else:
                raise Exception(
                    "The metric {} has not been implemented".format(metric))

            if score > best_score:
                best_score = score
                best_node_mapping = node_mapping

        return best_node_mapping

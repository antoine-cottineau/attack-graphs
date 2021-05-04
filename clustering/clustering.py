import numpy as np
import clustering.space_metrics as space_metrics

from attack_graph import AttackGraph
from sklearn.cluster import KMeans
from typing import List


class ClusteringMethod:
    def __init__(self, ag: AttackGraph):
        self.ag = ag

        # By default, all the nodes are grouped into a unique cluster
        self.clusters = dict()
        for node in self.ag.nodes:
            self.clusters[node] = 0

    def cluster(self):
        pass

    def update_clusters(self, node_mapping: List[int]):
        self.clusters = dict()
        for i, node in enumerate(self.ag.nodes):
            self.clusters[node] = node_mapping[i]

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

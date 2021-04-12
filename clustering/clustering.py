import numpy as np

from attack_graph import AttackGraph
from clustering.metric import Metric
from sklearn.cluster import KMeans


class Clustering:
    def __init__(self, ag: AttackGraph):
        self.ag = ag

    def find_optimal_number_of_clusters(self,
                                        X: np.array,
                                        k_min: int,
                                        k_max: int,
                                        metric: str = "silhouette"):
        best_score = -np.inf
        best_labels = None
        best_k = k_min

        for k in range(k_min, k_max + 1):
            # Apply k-means with k clusters
            labels = KMeans(n_clusters=k).fit_predict(X)

            # Compute the score
            if metric == "silhouette":
                score = Metric.score_with_silhouette(X, labels)
            elif metric == "ch":
                score = Metric.score_with_calinski_harabasz(X, labels)
            elif metric == "db":
                score = Metric.score_with_davies_bouldin(X, labels)
            else:
                raise Exception(
                    "The metric {} has not been implemented".format(metric))

            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k

        return best_labels, best_k, best_score

import numpy as np

from attack_graph import AttackGraph
from sklearn.cluster import KMeans


class Embedding:
    """
    Base class to represent methods that produce embeddings from an attack
    graph.

    :param AttackGraph ag: The attack graph.
    :param int dim_embedding: The dimension of the embedding of each node.
    """
    def __init__(self, ag: AttackGraph, dim_embedding: int):
        self.ag = ag
        self.dim_embedding = dim_embedding

        self.embeddings = np.zeros((self.ag.N, dim_embedding))

    def cluster_with_k_clusters(self, k: int):
        """
        Cluster the graph with k clusters by using K-means.

        :param int k: The number of clusters.
        """
        clustering = KMeans(n_clusters=k).fit_predict(self.embeddings)

        for i in range(self.ag.N):
            self.ag.states[i].id_cluster = clustering[i]

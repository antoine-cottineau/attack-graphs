import numpy as np
import utils

from attack_graph import AttackGraph
from clustering.clustering import Clustering


class Embedding:
    def __init__(self, ag: AttackGraph, dim_embedding: int):
        self.ag = ag
        self.dim_embedding = dim_embedding

        self.embedding = np.zeros((ag.number_of_nodes(), dim_embedding))

    def save_embedding_in_file(self, path: str):
        utils.create_parent_folders(path)
        np.save(path, self.embedding)

    def cluster(self):
        # Cluster the graph
        labels = Clustering(self.ag).find_optimal_number_of_clusters(
            X=self.embedding, k_min=2, k_max=10)[0]

        # Apply the clustering to the nodes
        for i, node in self.ag.nodes(data=True):
            node["id_cluster"] = labels[i]

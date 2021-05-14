import numpy as np
import utils

from attack_graph import StateAttackGraph
from clustering.clustering import ClusteringMethod


class EmbeddingMethod(ClusteringMethod):
    def __init__(self, ag: StateAttackGraph, dim_embedding: int = 16):
        super().__init__(ag)

        self.ag = ag
        self.dim_embedding = dim_embedding

        self.embedding = np.zeros((ag.number_of_nodes(), dim_embedding))

    def embed(self):
        pass

    def cluster(self, k_min: int = 2, k_max: int = 15):
        node_assignment = ClusteringMethod.evaluate_space_clustering(
            X=self.embedding, k_min=k_min, k_max=k_max)

        self.update_clusters(node_assignment)

    def save_embedding_in_file(self, path: str):
        utils.create_parent_folders(path)
        np.save(path, self.embedding)

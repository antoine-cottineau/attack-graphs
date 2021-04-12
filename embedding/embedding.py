import numpy as np

from attack_graph import AttackGraph
from clustering.clustering import Clustering
from pathlib import Path


class Embedding:
    def __init__(self, ag: AttackGraph, dim_embedding: int):
        self.ag = ag
        self.dim_embedding = dim_embedding

        self.embedding = np.zeros((ag.number_of_nodes(), dim_embedding))

    def save_embedding_in_file(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        np.save(path, self.embedding)
        print("Embedding saved in file \"{}\".".format(path))

    def cluster(self):
        # Cluster the graph
        labels = Clustering(self.ag).find_optimal_number_of_clusters(
            X=self.embedding, k_min=2, k_max=10)[0]

        # Apply the clustering to the nodes
        for i, node in self.ag.nodes(data=True):
            node["id_cluster"] = labels[i]

import click
import numpy as np

from attack_graph import AttackGraph
from sklearn.cluster import KMeans


class Embedding:
    def __init__(self, ag: AttackGraph, dim_embedding: int):
        self.ag = ag
        self.dim_embedding = dim_embedding

        self.embedding = np.zeros((ag.number_of_nodes(), dim_embedding))

    def save_embedding_in_file(self, path: str):
        np.save(path, self.embedding)
        click.echo("Embedding saved in file \"{}\".".format(path))

    def cluster_with_k_clusters(self, k: int):
        # Cluster the graph with k-means
        clustering = KMeans(n_clusters=k).fit_predict(self.embedding)

        # Apply the clustering to the nodes
        for i, node in self.ag.nodes(data=True):
            node["id_cluster"] = clustering[i]

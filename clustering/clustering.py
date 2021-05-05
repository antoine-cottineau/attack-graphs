import clustering.space_metrics as space_metrics
import networkx as nx
import numpy as np

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

    def get_clusters(self) -> np.array:
        node_mapping = list(self.node_mapping.values())
        return np.unique(node_mapping)

    def evaluate_modularity(self) -> float:
        modularity = 0

        adjacency_matrix = self.ag.compute_adjacency_matrix()
        node_mapping = list(self.node_mapping.values())
        clusters = self.get_clusters()

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

    def evaluate_mean_silhouette_index(self) -> float:
        n = self.ag.number_of_nodes()
        node_indices = self.ag.get_node_mapping()
        clusters = self.get_clusters()

        distance_matrix = np.zeros((n, n))

        # Fill distance_matrix
        distances = nx.shortest_path_length(
            nx.Graph(incoming_graph_data=self.ag))
        for result in enumerate(distances):
            i = node_indices[result[0]]
            for dst, distance in result[1][1].items():
                j = node_indices[dst]
                distance_matrix[i][j] = distance

        # Compute silhouette index node by node
        nodes_silhouette_index = np.zeros(n)
        clusters_content = [[
            node_indices[node] for node in self.ag.nodes
            if self.node_mapping[node] == cluster
        ] for cluster in clusters]
        for node in self.ag.nodes:
            i = node_indices[node]
            node_cluster = self.node_mapping[node]

            mean_cluster_distances = [
                distance_matrix[i][clusters_content[cluster]].mean()
                for cluster in clusters
            ]

            a = mean_cluster_distances[node_cluster]
            b = min([
                distance
                for cluster, distance in enumerate(mean_cluster_distances)
                if cluster != node_cluster
            ])

            nodes_silhouette_index[i] = (b - a) / max(a, b)

        mean_silhouette_index = nodes_silhouette_index.mean()

        return mean_silhouette_index

    def evaluate_external_conductance(self) -> float:
        adjacency_matrix = self.ag.compute_adjacency_matrix()
        clusters = self.get_clusters()
        node_positions = self.ag.get_node_mapping()

        # Compute the external conductance for each cluster
        cluster_conductances = np.zeros(len(clusters))
        for cluster in clusters:
            nodes = [
                node for node in self.ag.nodes
                if self.node_mapping[node] == cluster
            ]
            cluster_node_positions = [node_positions[node] for node in nodes]
            complement_node_positions = [
                node_positions[node] for node in self.ag.nodes
                if node not in nodes
            ]
            numerator = adjacency_matrix[
                cluster_node_positions][:, complement_node_positions].sum()

            a_cluster = adjacency_matrix[cluster_node_positions].sum()
            a_complement = adjacency_matrix[complement_node_positions].sum()
            denominator = min(a_cluster, a_complement)

            cluster_conductances[cluster] = numerator / denominator

        mean_cluster_conductance = cluster_conductances.mean()

        return mean_cluster_conductance

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

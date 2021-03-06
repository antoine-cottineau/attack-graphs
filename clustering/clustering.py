import clustering.space_metrics as space_metrics
import networkx as nx
import numpy as np
from attack_graph import BaseGraph
from sklearn.cluster import KMeans
from typing import Dict, List


class ClusteringMethod:
    def __init__(self, graph: BaseGraph):
        self.graph = graph

        # By default, all the nodes are grouped into a unique cluster
        self.node_assignment: Dict[int, int] = dict()
        for node in self.graph.nodes:
            self.node_assignment[node] = 0
        self.clusters: Dict[int, List[int]] = {"0": [list(self.graph.nodes)]}

    def cluster(self):
        pass

    def update_clusters(self, node_assignment: List[int]):
        self.node_assignment = {}
        self.clusters = {}
        for i, node in enumerate(self.graph.nodes):
            id_cluster = node_assignment[i]
            self.node_assignment[node] = id_cluster
            if id_cluster in self.clusters:
                self.clusters[id_cluster].append(node)
            else:
                self.clusters[id_cluster] = [node]

    def get_ids_clusters(self) -> np.array:
        node_assignment = list(self.node_assignment.values())
        return np.unique(node_assignment)

    def evaluate_modularity(self) -> float:
        node_assignment = list(self.node_assignment.values())
        return ClusteringMethod.modularity(self.graph, node_assignment)

    def evaluate_mean_silhouette_index(self) -> float:
        n = self.graph.number_of_nodes()
        node_ordering = self.graph.get_node_ordering()
        ids_clusters = self.get_ids_clusters()

        distance_matrix = np.zeros((n, n))

        # Fill distance_matrix
        distances = nx.shortest_path_length(self.graph.to_undirected())
        for result in distances:
            src = result[0]
            i = node_ordering[src]
            for dst, distance in result[1].items():
                j = node_ordering[dst]
                distance_matrix[i][j] = distance

        # Compute silhouette index node by node
        nodes_silhouette_index = np.zeros(n)
        clusters_content = [[
            node_ordering[node] for node in self.graph.nodes
            if self.node_assignment[node] == cluster
        ] for cluster in ids_clusters]
        for node in self.graph.nodes:
            i = node_ordering[node]
            node_cluster = self.node_assignment[node]

            mean_cluster_distances = [
                distance_matrix[i][clusters_content[cluster]].mean()
                for cluster in ids_clusters
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

    def evaluate_mean_conductance(self) -> float:
        adjacency_matrix = self.graph.compute_adjacency_matrix(directed=False)
        ids_clusters = self.get_ids_clusters()
        node_ordering = self.graph.get_node_ordering()

        # Compute the conductance for each cluster
        cluster_conductances = np.zeros(len(ids_clusters))
        for cluster in ids_clusters:
            nodes = [
                node for node in self.graph.nodes
                if self.node_assignment[node] == cluster
            ]
            cluster_node_positions = [node_ordering[node] for node in nodes]
            complement_node_positions = [
                node_ordering[node] for node in self.graph.nodes
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

    def evaluate_mean_coverage(self) -> float:
        adjacency_matrix = self.graph.compute_adjacency_matrix(directed=False)
        ids_clusters = self.get_ids_clusters()
        node_ordering = self.graph.get_node_ordering()

        # Compute the coverage for each cluster
        cluster_coverages = np.zeros(len(ids_clusters))
        for cluster in ids_clusters:
            nodes = [
                node for node in self.graph.nodes
                if self.node_assignment[node] == cluster
            ]
            cluster_node_positions = [node_ordering[node] for node in nodes]
            numerator = adjacency_matrix[
                cluster_node_positions][:, cluster_node_positions].sum()
            denominator = adjacency_matrix.sum()

            cluster_coverages[cluster] = numerator / denominator

        mean_cluster_coverage = cluster_coverages.mean()

        return mean_cluster_coverage

    @staticmethod
    def modularity(graph: BaseGraph, node_assignment: List[int]) -> float:
        modularity = 0

        adjacency_matrix = graph.compute_adjacency_matrix(directed=False)
        ids_clusters = np.unique(node_assignment)

        sum_all_weights = adjacency_matrix.sum()

        for cluster in ids_clusters:
            adj_matrix_cluster = adjacency_matrix[
                node_assignment == cluster][:, node_assignment == cluster]
            adj_matrix_out_cluster = adjacency_matrix[
                node_assignment != cluster][:, node_assignment != cluster]
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
        best_node_assignment = None

        for k in range(k_min, k_max + 1):
            # Apply k-means with k clusters
            node_assignment = KMeans(n_clusters=k).fit_predict(X)

            # Compute the score
            if metric == "silhouette":
                score = space_metrics.score_with_silhouette(X, node_assignment)
            elif metric == "ch":
                score = space_metrics.score_with_calinski_harabasz(
                    X, node_assignment)
            elif metric == "db":
                score = space_metrics.score_with_davies_bouldin(
                    X, node_assignment)
            else:
                raise Exception(
                    "The metric {} has not been implemented".format(metric))

            if score > best_score:
                best_score = score
                best_node_assignment = node_assignment

        return best_node_assignment

import torch

from attack_graph import AttackGraph
from attack_graph_generation import Generator
from ranking.mehta import PageRankMethod
from torch_geometric.data import Data
from typing import List, Tuple


class GraphSageRanking:
    def __init__(self, n_graphs: int, dim_embedding: int):
        self.n_graphs = n_graphs
        self.dim_embedding = dim_embedding

    def create_data(self) -> List[Data]:
        list_data = []
        graphs, list_rankings = self.generate_graphs()
        for i, graph in enumerate(graphs):
            rankings = list_rankings[i]

            features = GraphSageRanking.create_node_feature_matrix(graph)
            connectivity = GraphSageRanking.create_graph_connectivity(graph)
            targets = GraphSageRanking.create_targets(rankings)

            data = Data(x=features, edge_index=connectivity, y=targets)
            list_data.append(data)

        return list_data

    def generate_graphs(self) -> Tuple[List[AttackGraph], List[List[float]]]:
        generator = Generator()

        # Create the n_graphs that will be necessary to create the input data
        # and apply PageRank ranking on each one of them
        graphs = []
        list_rankings = []
        for _ in range(self.n_graphs):
            ag = generator.generate()
            graphs.append(ag)
            list_rankings.append(list(PageRankMethod(ag).apply().values()))

        return graphs, list_rankings

    @staticmethod
    def create_node_feature_matrix(graph: AttackGraph) -> torch.Tensor:
        feature_matrix = torch.zeros((graph.number_of_nodes(), 3),
                                     dtype=torch.float)

        # First feature: the layer of a node normalized by the number of layers
        nodes_layers = torch.tensor(list(graph.get_nodes_layers().values()),
                                    dtype=torch.float)
        nodes_layers = nodes_layers / max(nodes_layers)

        # Second feature: the number of incoming edges
        incoming_degrees = torch.tensor(
            [degree for _, degree in graph.in_degree()], dtype=torch.float)

        # Third feature: the number of outcoming edges
        outcoming_degrees = torch.tensor(
            [degree for _, degree in graph.out_degree()], dtype=torch.float)

        feature_matrix[:, 0] = nodes_layers
        feature_matrix[:, 1] = incoming_degrees
        feature_matrix[:, 2] = outcoming_degrees

        return feature_matrix

    @staticmethod
    def create_graph_connectivity(graph: AttackGraph) -> torch.Tensor:
        connectivity = torch.tensor(list(graph.edges), dtype=torch.long)
        connectivity = connectivity.t().contiguous()
        return connectivity

    @staticmethod
    def create_targets(rankings: List[float]) -> torch.Tensor:
        return torch.tensor(rankings, dtype=torch.float)

import torch
import torch.nn.functional as F
import utils

from attack_graph import AttackGraph
from attack_graph_generation import Generator
from ranking.mehta import PageRankMethod
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.data.sampler import NeighborSampler
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
from typing import Dict, List


class GraphSageRanking:

    path_weights = "methods_output/graphsage_ranking/weights.pth"
    path_graphs = "methods_input/graphsage_ranking"

    def create_model(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sage(3, 16, 1)
        self.model = self.model.to(self.device)

    def apply(self, graphs: List[AttackGraph]) -> List[Dict[int, float]]:
        list_data = self.create_data(graphs=graphs, with_targets=False)
        data = Batch.from_data_list(list_data)

        # Create a loader
        loader = NeighborSampler(data.edge_index,
                                 node_idx=None,
                                 sizes=[-1],
                                 batch_size=1024,
                                 num_workers=4)

        x = data.x.to(self.device)
        self.model.eval()
        out = self.model.infer(x, loader, self.device)

        list_rankings = []
        start = 0
        for graph in graphs:
            n = graph.number_of_nodes()
            rankings = out[start:start + n].squeeze()
            ids = list(graph.nodes)
            list_rankings.append(
                dict([(ids[i], float(rankings[i])) for i in range(n)]))
            start += n

        return list_rankings

    def train(self,
              n_epochs: int = 10,
              n_graphs: int = 10,
              load_graphs: bool = False,
              save_generated_graphs: bool = False):
        # Create the data
        list_data = self.create_data(
            n_graphs=n_graphs,
            load_graphs=load_graphs,
            save_generated_graphs=save_generated_graphs)
        data = Batch.from_data_list(list_data)

        # Create a loader
        self.train_loader = NeighborSampler(data.edge_index,
                                            sizes=[15, 5],
                                            batch_size=256,
                                            shuffle=True,
                                            num_workers=4)

        # Create the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        # Move the data to the device
        self.x: torch.Tensor = data.x.to(self.device)
        self.y: torch.Tensor = data.y.to(self.device)

        # Train the model
        for i_epoch in range(1, n_epochs + 1):
            loss = self.train_one_epoch(i_epoch)
            print("Loss at epoch {}: {:.2E}".format(i_epoch, loss))

    def train_one_epoch(self, i_epoch: int):
        self.model.train()

        pbar = tqdm(total=self.x.shape[0])
        pbar.set_description("Epoch {}".format(i_epoch))

        total_loss = 0

        for batch_size, n_id, adjs in self.train_loader:
            adjs = [adj.to(self.device) for adj in adjs]

            self.optimizer.zero_grad()
            out = self.model(self.x[n_id], adjs).squeeze()
            loss = F.mse_loss(out, self.y[n_id[:batch_size]])
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss)
            pbar.update(batch_size)

        pbar.close()

        return total_loss / len(self.train_loader)

    def evaluate(self, n_graphs: int = 5):
        list_data = self.create_data(n_graphs=n_graphs)
        data = Batch.from_data_list(list_data)

        # Create a loader
        loader = NeighborSampler(data.edge_index,
                                 node_idx=None,
                                 sizes=[-1],
                                 batch_size=1024,
                                 num_workers=4)

        x = data.x.to(self.device)
        self.model.eval()
        out = self.model.infer(x, loader, self.device).squeeze()

        loss = float(F.mse_loss(data.y, out))
        return loss / len(loader)

    def create_data(self,
                    n_graphs: int = None,
                    graphs: List[AttackGraph] = None,
                    load_graphs: bool = False,
                    save_generated_graphs: bool = False,
                    with_targets: bool = True) -> List[Data]:
        list_data = []
        if graphs is None:
            if load_graphs:
                GraphSageRanking.load_graphs(n_graphs)
            else:
                graphs = GraphSageRanking.generate_graphs(n_graphs)

        if save_generated_graphs:
            GraphSageRanking.save_graphs(graphs)

        if with_targets:
            list_rankings = GraphSageRanking.generate_rankings(graphs)

        for i, graph in enumerate(graphs):
            features = GraphSageRanking.create_node_feature_matrix(graph)
            connectivity = GraphSageRanking.create_graph_connectivity(graph)

            data = Data(x=features, edge_index=connectivity)
            if with_targets:
                rankings = list_rankings[i]
                targets = GraphSageRanking.create_targets(rankings)
                data.y = targets

            list_data.append(data)

        return list_data

    def save_model(self):
        utils.create_parent_folders(GraphSageRanking.path_weights)
        torch.save(self.model.state_dict(), GraphSageRanking.path_weights)

    def load_model(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sage(3, 16, 1)
        self.model.load_state_dict(torch.load(GraphSageRanking.path_weights))
        self.model = self.model.to(self.device)

    @staticmethod
    def save_graphs(graphs: List[AttackGraph]):
        utils.create_folders(GraphSageRanking.path_graphs)
        for i, graph in enumerate(graphs):
            graph.save("{}/{}.npy".format(GraphSageRanking.path_graphs, i))

    @staticmethod
    def load_graphs(n_graphs: int) -> List[AttackGraph]:
        graphs = []

        files = utils.list_files_in_directory(GraphSageRanking.path_graphs)
        i = 0
        while i < n_graphs and i < len(files):
            graph = AttackGraph()
            graphs.append(
                graph.load("{}/{}".format(GraphSageRanking.path_graphs, i)))
            i += 1

        return graphs

    @staticmethod
    def generate_graphs(n_graphs: int) -> List[AttackGraph]:
        generator = Generator(n_propositions=30, n_exploits=30)

        # Create the n_graphs that will be necessary to create the input data
        # and apply PageRank ranking on each one of them
        graphs = []
        for _ in range(n_graphs):
            graphs.append(generator.generate())

        return graphs

    @staticmethod
    def generate_rankings(graphs: List[AttackGraph]) -> List[List[float]]:
        list_rankings = []
        for graph in graphs:
            list_rankings.append(list(PageRankMethod(graph).apply().values()))
        return list_rankings

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


class Sage(torch.nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):
        super().__init__()

        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(SAGEConv(dim_in, dim_hidden))
        self.conv_layers.append(SAGEConv(dim_hidden, dim_out))

    def forward(self, x: torch.Tensor, adjs: list) -> torch.Tensor:
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.conv_layers[i]((x, x_target), edge_index)
            if i != len(self.conv_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def infer(self, x: torch.Tensor, loader: NeighborSampler,
              device: torch.device) -> torch.Tensor:
        for i in range(2):
            representations = []
            for _, n_id, adj in loader:
                edge_index, _, size = adj.to(device)
                current_x = x[n_id].to(device)
                current_x_target = current_x[:size[1]]
                current_x = self.conv_layers[i]((current_x, current_x_target),
                                                edge_index)
                if i != len(self.conv_layers) - 1:
                    current_x = F.relu(current_x)
                representations.append(current_x.cpu())

            x = torch.cat(representations, dim=0)

        return x

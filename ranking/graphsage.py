import torch
import torch.nn.functional as F
import utils

from attack_graph import AttackGraph
from attack_graph_generation import Generator
from ranking.mehta import PageRankMethod
from torch_geometric.data import Data, NeighborSampler
from torch_geometric.nn import SAGEConv
from typing import Dict, List


class GraphSageRanking:

    path_weights = "methods_output/graphsage_ranking/weights.pth"
    path_graphs = "methods_input/generated_graphs"

    def __init__(self, dim_hidden: int, verbose=False, with_ppce=False):
        self.dim_hidden = dim_hidden
        self.verbose = verbose
        self.with_ppce = with_ppce
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def create_model(self):
        if hasattr(self, "model"):
            return

        self.model = Sage(3, self.dim_hidden, 1)
        self.model = self.model.to(self.device)

    def train(self,
              n_epochs: int = 20,
              n_graphs: int = 30,
              load_graphs: bool = False,
              save_graphs: bool = False):
        # Create the input data
        if load_graphs:
            graphs = self.load_graphs(n_graphs)
        else:
            graphs = self.create_graphs(n_graphs)

        if save_graphs:
            self.save_graphs(graphs)

        rankings = self.create_rankings(graphs)

        self.show_message(
            "Generating the input data from the graphs and rankings")
        self.data_list: List[Data] = []
        for i, graph in enumerate(graphs):
            x = GraphSageRanking.create_node_feature_matrix(graph)
            edge_index = GraphSageRanking.create_graph_connectivity(graph)
            y = rankings[i]

            graph_data = Data(x=x, edge_index=edge_index, y=y)
            self.data_list.append(graph_data)

        # Create the model
        self.create_model()

        # Create the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        # Train the model
        self.show_message("Training the model")
        self.model.train()
        for i_epoch in range(1, n_epochs + 1):
            self.show_message("Epoch {}".format(i_epoch))
            loss = self.train_one_epoch()
            self.show_message("Loss at epoch {}: {:.2E}".format(i_epoch, loss))

    def train_one_epoch(self):
        total_loss = 0
        n_nodes = 0
        for data in self.data_list:
            # Create a sampler
            sampler = NeighborSampler(edge_index=data.edge_index,
                                      sizes=[15, 5],
                                      batch_size=32,
                                      shuffle=True,
                                      num_workers=4)

            # Move the data to the device
            x: torch.Tensor = data.x.to(self.device)
            y: torch.Tensor = data.y.to(self.device)

            for batch_size, n_id, adjs in sampler:
                moved_adjs = [adj.to(self.device) for adj in adjs]

                self.optimizer.zero_grad()
                out = self.model(x[n_id], moved_adjs).squeeze()
                targets = y[n_id[:batch_size]]
                loss = F.mse_loss(out, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss)

                if self.with_ppce and self.verbose and n_nodes % 200 == 0:
                    ppce = self.apply_ppce(targets, out)
                    self.show_message("PPCE: {:.0f}".format(ppce * 100))

            n_nodes += x.size(0)

        return total_loss / n_nodes

    def apply(self, graphs: List[AttackGraph]) -> List[Dict[int, float]]:
        rankings = []
        for graph in graphs:
            graph_rankings = self.apply_to_one_graph(graph)
            rankings.append(graph_rankings)
        return rankings

    def apply_to_one_graph(self, graph: AttackGraph) -> Dict[int, float]:
        # Generate the input data
        self.show_message("Generating the input data from the graphs")
        x = GraphSageRanking.create_node_feature_matrix(graph)
        edge_index = GraphSageRanking.create_graph_connectivity(graph)
        data = Data(x=x, edge_index=edge_index)

        # Create the model
        self.create_model()

        # Create a sampler
        sampler = NeighborSampler(data.edge_index,
                                  sizes=[-1],
                                  batch_size=1024,
                                  num_workers=4)

        # Apply the model
        self.show_message("Applying the model")
        self.model.eval()
        out = self.model.infer(data, sampler, self.device).squeeze().detach()

        # Create the rankings dictionary
        ids = list(graph.nodes)
        rankings = dict([(ids[i], out[i])
                         for i in range(graph.number_of_nodes())])
        return rankings

    def create_graphs(self, n_graphs: int) -> List[AttackGraph]:
        self.show_message("Creating {} graphs".format(n_graphs))
        graphs = []
        for i in range(n_graphs):
            complexity = 20 + int(10 * i / n_graphs)
            generator = Generator(n_propositions=complexity,
                                  n_exploits=complexity)
            self.show_message("Generating graph {} with complexity {}".format(
                i, complexity))
            graph = generator.generate()
            graphs.append(graph)
            self.show_message("Generated graph {} with {} nodes".format(
                i, graph.number_of_nodes()))
        return graphs

    def load_graphs(self, n_graphs: int) -> List[AttackGraph]:
        self.show_message("Loading {} graphs".format(n_graphs))

        graphs = []
        files = utils.list_files_in_directory(GraphSageRanking.path_graphs)
        i = 0
        while i < n_graphs and i < len(files):
            graph = AttackGraph()
            path = "{}/{}.json".format(GraphSageRanking.path_graphs, i)
            self.show_message("Loading graph {}".format(path))
            graph.load(path)
            self.show_message("Loaded graph {} with {} nodes".format(
                path, graph.number_of_nodes()))
            graphs.append(graph)
            i += 1

        return graphs

    def save_graphs(self, graphs: List[AttackGraph]):
        utils.create_folders(GraphSageRanking.path_graphs)
        for i, graph in enumerate(graphs):
            path = "{}/{}.json".format(GraphSageRanking.path_graphs, i)
            self.show_message("Saving graph {}".format(path))
            graph.save(path)

    def create_rankings(self, graphs: List[AttackGraph]) -> List[torch.Tensor]:
        self.show_message("Creating the rankings")
        rankings = []
        for i, graph in enumerate(graphs):
            self.show_message("Creating the rankings for graph {}".format(i))

            graph_rankings = list(PageRankMethod(graph).apply().values())
            graph_rankings = torch.tensor(graph_rankings, dtype=torch.float)
            graph_rankings /= torch.linalg.norm(graph_rankings)

            rankings.append(graph_rankings)
        return rankings

    @staticmethod
    def create_node_feature_matrix(graph: AttackGraph) -> torch.Tensor:
        feature_matrix = torch.zeros((graph.number_of_nodes(), 3),
                                     dtype=torch.float)

        # First feature: the layer of a node normalized by the number of layers
        nodes_layers = torch.tensor(list(graph.get_nodes_layers().values()),
                                    dtype=torch.float)
        nodes_layers = nodes_layers / torch.max(nodes_layers)

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
    def apply_ppce(true_rankings: torch.tensor,
                   output_rankings: torch.tensor) -> float:
        n = true_rankings.size(0)
        n_total = 0
        n_errors = 0

        true_positions = torch.argsort(true_rankings)
        output_positions = torch.argsort(output_rankings)
        for i in range(n):
            for j in range(n):
                if i != j:
                    n_total += 1
                    if true_positions[i] < true_positions[
                            j] and output_positions[i] > output_positions[j]:
                        n_errors += 1
                    if true_positions[i] > true_positions[
                            j] and output_positions[i] < output_positions[j]:
                        n_errors += 1
        return n_errors / n_total

    def save_model(self):
        utils.create_parent_folders(GraphSageRanking.path_weights)
        torch.save(self.model.state_dict(), GraphSageRanking.path_weights)

    def load_model(self):
        self.model = Sage(3, self.dim_hidden, 1)
        self.model.load_state_dict(torch.load(GraphSageRanking.path_weights))
        self.model = self.model.to(self.device)

    def show_message(self, message: str):
        if self.verbose:
            print(message)


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
        return torch.sigmoid(x)

    def infer(self, data: Data, sampler: NeighborSampler,
              device: torch.device) -> torch.Tensor:
        x_all = data.x
        for i in range(2):
            outputs = []
            for _, n_id, adj in sampler:
                edge_index, _, size = adj.to(device)
                x: torch.Tensor = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.conv_layers[i]((x, x_target), edge_index)
                if i != 1:
                    x = F.relu(x)
                else:
                    x = torch.sigmoid(x)
                outputs.append(x.cpu())

            x_all = torch.cat(outputs, dim=0)

        return x_all

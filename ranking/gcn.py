import torch
import torch.nn.functional as F
import utils
from attack_graph import AttackGraph
from attack_graph_generation import Generator
from ranking.mehta import PageRankMethod
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from typing import Dict, List


class GCNRanking:

    path_weights = "methods_output/gcn_ranking/weights.pth"
    path_graphs = "methods_input/gcn_ranking"

    def __init__(self, dim_hidden: int, verbose=False):
        self.dim_hidden = dim_hidden
        self.verbose = verbose
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def train(self,
              n_epochs: int = 5,
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
        self.data_list = []
        for i, graph in enumerate(graphs):
            x = GCNRanking.create_node_feature_matrix(graph)
            edge_index = GCNRanking.create_graph_connectivity(graph)
            y = rankings[i]

            graph_data = Data(x=x, edge_index=edge_index, y=y)
            self.data_list.append(graph_data)

        # Create the model
        self.model = GCN(3, self.dim_hidden, 1)

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
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index).squeeze()
            targets = data.y
            loss = F.mse_loss(out, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss)
            n_nodes += len(data.x)

        return total_loss / n_nodes

    def apply(self, graphs: List[AttackGraph]) -> List[Dict[int, float]]:
        rankings = self.create_rankings(graphs)

        self.show_message(
            "Generating the input data from the graphs and rankings")
        data_list = []
        for i, graph in enumerate(graphs):
            x = GCNRanking.create_node_feature_matrix(graph)
            edge_index = GCNRanking.create_graph_connectivity(graph)
            y = rankings[i]

            graph_data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(graph_data)

        self.data = DataLoader(data_list, batch_size=1024, shuffle=False)

        # Create the model
        self.model = GCN(3, 8, 1)

        # Apply the model
        self.model.eval()
        out = []
        for batch in self.data:
            batch_out = self.model(batch.x, batch.edge_index).squeeze()
            out.append(batch_out)
        out = torch.cat(out)

        # Create the output
        out_rankings = []
        start = 0
        for graph in graphs:
            n = graph.number_of_nodes()
            graph_rankings = out[start:start + n].squeeze()
            ids = list(graph.nodes)
            out_rankings.append(
                dict([(ids[i], float(graph_rankings[i])) for i in range(n)]))
            start += n

        return out_rankings

    def create_graphs(self, n_graphs: int) -> List[AttackGraph]:
        self.show_message("Creating {} graphs".format(n_graphs))
        graphs = []
        for i in range(n_graphs):
            complexity = 12 + int(23 * i / n_graphs)
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
        files = utils.list_files_in_directory(GCNRanking.path_graphs)
        i = 0
        while i < n_graphs and i < len(files):
            graph = AttackGraph()
            path = "{}/{}.json".format(GCNRanking.path_graphs, i)
            self.show_message("Loading graph {}".format(path))
            graph.load(path)
            self.show_message("Loaded graph {} with {} nodes".format(
                path, graph.number_of_nodes()))
            graphs.append(graph)
            i += 1

        return graphs

    def save_graphs(self, graphs: List[AttackGraph]):
        utils.create_folders(GCNRanking.path_graphs)
        for i, graph in enumerate(graphs):
            path = "{}/{}.json".format(GCNRanking.path_graphs, i)
            self.show_message("Saving graph {}".format(path))
            graph.save(path)

    def create_rankings(self, graphs: List[AttackGraph]) -> List[torch.Tensor]:
        self.show_message("Creating the rankings")
        rankings = []
        for i, graph in enumerate(graphs):
            self.show_message("Create the rankings for graph {}".format(i))

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

    def save_model(self):
        utils.create_parent_folders(GCNRanking.path_weights)
        torch.save(self.model.state_dict(), GCNRanking.path_weights)

    def load_model(self):
        self.model = GCN(3, self.dim_hidden, 1)
        self.model.load_state_dict(torch.load(GCNRanking.path_weights))

    def show_message(self, message: str):
        if self.verbose:
            print(message)


class GCN(torch.nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):
        super().__init__()

        self.conv_0 = GCNConv(dim_in, dim_hidden)
        self.conv_1 = GCNConv(dim_hidden, dim_out)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        new_x = self.conv_0(x, edge_index)
        new_x = F.relu(new_x)
        new_x = self.conv_1(new_x, edge_index)
        new_x = torch.sigmoid(new_x)
        return new_x

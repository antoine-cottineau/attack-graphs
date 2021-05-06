import torch
import torch.nn as nn
import torch.nn.functional as F
from attack_graph import AttackGraph
from embedding.embedding import EmbeddingMethod
from torch_cluster import random_walk
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, NeighborSampler as RawNeighborSampler


class GraphSage(EmbeddingMethod):
    def __init__(self,
                 ag: AttackGraph,
                 dim_embedding: int,
                 dim_hidden_layer: int = 16,
                 n_epochs: int = 50,
                 device: str = None,
                 verbose: bool = False):
        super().__init__(ag, dim_embedding)

        self.dim_hidden_layer = dim_hidden_layer
        self.n_epochs = n_epochs
        self.device = device
        self.verbose = verbose

        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

    def embed(self):
        # Train the model
        self.train()

        # Move the edge_index to the device
        edge_index = self.data.edge_index.to(self.device)

        # Apply the model
        self.model.eval()
        self.embedding = self.model.full_forward(
            self.x, edge_index).cpu().detach().numpy()

    def train(self):
        self.create_model_and_optimizer()
        self.create_data()
        self.create_neighbor_sampler()

        # Move the data to the device
        self.x = self.data.x.to(self.device)

        # Train the model
        self.model.train()
        for i_epoch in range(self.n_epochs):
            loss = self.train_one_epoch()
            self.show_message("Epoch {}, loss: {:.2f}".format(
                i_epoch + 1, loss))

    def train_one_epoch(self) -> float:
        total_loss = 0
        for _, n_id, adjs in self.neighbor_sampler:
            moved_adjs = [adj.to(self.device) for adj in adjs]
            self.optimizer.zero_grad()

            out: torch.Tensor = self.model(self.x[n_id], moved_adjs)
            out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

            pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
            neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
            loss = -pos_loss - neg_loss
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss) * out.size(0)

        return total_loss / self.data.num_nodes

    def create_model_and_optimizer(self):
        self.model = Sage(self.ag.number_of_nodes(), self.dim_hidden_layer,
                          self.dim_embedding)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def create_data(self):
        x = torch.eye(self.ag.number_of_nodes())
        edge_index = torch.tensor(list(self.ag.edges), dtype=torch.long)
        edge_index = edge_index.t().contiguous()

        self.data = Data(x=x, edge_index=edge_index)

    def create_neighbor_sampler(self):
        self.neighbor_sampler = NeighborSampler(self.data.edge_index,
                                                sizes=[10, 10],
                                                batch_size=256,
                                                shuffle=True,
                                                num_nodes=self.data.num_nodes)

    def show_message(self, message: str):
        if self.verbose:
            print(message)


class Sage(nn.Module):
    def __init__(self, dim_input: int, dim_hidden_1, dim_output):
        super().__init__()

        self.layer_1 = SAGEConv(dim_input, dim_hidden_1)
        self.layer_2 = SAGEConv(dim_hidden_1, dim_output)

    def forward(self, x, adjs):
        # First layer
        edge_index, _, size = adjs[0]
        x_target = x[:size[1]]
        out = self.layer_1((x, x_target), edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)

        # Second layer
        edge_index, _, size = adjs[1]
        x_target = out[:size[1]]
        out = self.layer_2((out, x_target), edge_index)

        return out

    def full_forward(self, x, edge_index):
        out = self.layer_1(x, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.layer_2(out, edge_index)

        return out


class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        new_batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        pos_batch = random_walk(row,
                                col,
                                new_batch,
                                walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0,
                                  self.adj_t.size(1), (new_batch.numel(), ),
                                  dtype=torch.long)

        new_batch = torch.cat([new_batch, pos_batch, neg_batch], dim=0)

        return super().sample(new_batch)

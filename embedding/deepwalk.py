import networkx as nx
import numpy as np

from attack_graph import AttackGraph
from embedding.embedding import EmbeddingMethod
from karateclub import DeepWalk as DW


class DeepWalk(EmbeddingMethod):
    def __init__(self,
                 ag: AttackGraph,
                 dim_embedding: int = 16,
                 walk_length: int = 80,
                 window_size: int = 5):
        super().__init__(ag, dim_embedding)

        self.walk_length = walk_length
        self.window_size = window_size

    def embed(self):
        seed = np.random.randint(1e6)
        model = DW(dimensions=self.dim_embedding,
                   walk_length=self.walk_length,
                   window_size=self.window_size,
                   seed=seed)
        model.fit(nx.Graph(incoming_graph_data=self.ag))
        self.embedding = model.get_embedding()

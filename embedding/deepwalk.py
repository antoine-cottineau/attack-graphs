import numpy as np
from attack_graph import BaseGraph
from embedding.embedding import EmbeddingMethod
from karateclub import DeepWalk as DW


class DeepWalk(EmbeddingMethod):
    def __init__(self,
                 graph: BaseGraph,
                 dim_embedding: int = 16,
                 walk_length: int = 80,
                 window_size: int = 5):
        super().__init__(graph, dim_embedding)

        self.walk_length = walk_length
        self.window_size = window_size

    def embed(self):
        seed = np.random.randint(1e6)
        model = DW(dimensions=self.dim_embedding,
                   walk_length=self.walk_length,
                   window_size=self.window_size,
                   seed=seed)
        model.fit(self.graph.to_undirected())
        self.embedding = model.get_embedding()

import networkx as nx

from attack_graph import AttackGraph
from embedding.embedding import Embedding
from karateclub import DeepWalk as DW


class DeepWalk(Embedding):
    def __init__(self, ag: AttackGraph, dim_embedding: int):
        super().__init__(ag, dim_embedding)

    def run(self):
        model = DW(dimensions=self.dim_embedding)
        model.fit(nx.Graph(incoming_graph_data=self.ag))
        self.embedding = model.get_embedding()
        self.save_embedding_in_file("methods_output/deepwalk/embedding.npy")

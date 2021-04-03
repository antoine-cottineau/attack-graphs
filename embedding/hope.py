import networkx as nx

from attack_graph import AttackGraph
from embedding.embedding import Embedding
from karateclub import HOPE


class Hope(Embedding):
    def __init__(self, ag: AttackGraph, dim_embedding: int):
        super().__init__(ag, dim_embedding)

    def run(self):
        model = HOPE(dimensions=self.dim_embedding)
        # TODO: Hope is supposed to work for directed graph
        model.fit(nx.Graph(incoming_graph_data=self.ag))
        self.embedding = model.get_embedding()
        self.save_embedding_in_file("methods_output/hope/embedding.npy")

import pathlib
import numpy as np
import utils
from attack_graph import AttackGraph
from attack_graph_generation import Generator
from time import time


class DatasetLoader:
    path_dataset = "methods_input/dataset"

    def __init__(self):
        self.files = [
            file
            for file in pathlib.Path(DatasetLoader.path_dataset).glob("*.json")
        ]
        self.sort_files()
        self.i = 0

    def sort_files(self):
        # Get the number of nodes of each graph
        self.number_of_nodes = np.zeros(len(self.files), dtype=int)
        for i, file in enumerate(self.files):
            ag = AttackGraph()
            ag.load(file)
            self.number_of_nodes[i] = ag.number_of_nodes()

        # Sort the attack graphs based on their number of nodes
        order = np.argsort(self.number_of_nodes)
        self.number_of_nodes = self.number_of_nodes[order]
        new_files = []
        for i in order:
            new_files.append(self.files[i])
        self.files = new_files

    def __iter__(self):
        return self

    def __next__(self) -> AttackGraph:
        self.i += 1
        if self.i < len(self.files):
            ag = AttackGraph()
            ag.load(self.files[self.i])
            return ag
        else:
            raise StopIteration


def create_dataset(start: int = 20, end: int = 50, quantity: int = 100):
    complexities = np.linspace(start, end, quantity, dtype=int)
    base_path = "methods_input/dataset"
    utils.create_folders(base_path)
    for i, complexity in enumerate(complexities):
        print("Generating graph with complexity {}".format(complexity))
        generator = Generator(n_propositions=complexity, n_exploits=complexity)
        start = time()
        ag = generator.generate()
        ag.save("{}/{}.json".format(base_path, i))
        print("Generated graph {} with complexity {} and {} nodes in {:.0f}"
              " seconds".format(i, complexity, ag.number_of_nodes(),
                                time() - start))

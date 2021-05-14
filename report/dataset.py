import numpy as np
import utils
from attack_graph import AttackGraph
from attack_graph_generation import Generator
from pathlib import Path
from typing import List


class Dataset:
    def __init__(self) -> None:
        self.set_sizes = [20, 20, 7]
        self.set_max_n_nodes = [1000, 10000, None]
        self.n_graphs = sum(self.set_sizes)
        self.base_path = "methods_input/dataset"
        self.paths = [
            Path(self.base_path, label)
            for label in ["small", "medium", "large"]
        ]

    def generate_and_save(self):
        min_complexity = 20
        max_complexity = 40

        sets: List[List[AttackGraph]] = []
        for _ in range(len(self.set_sizes)):
            sets.append([])

        # Generate the graphs
        are_sets_filled = False
        complexity = min_complexity
        while not are_sets_filled:
            # Generate a graph with the current complexity
            print("Generating graph with complexity {}".format(complexity))
            graph = Generator(
                n_propositions=complexity,
                n_exploits=complexity).generate_state_attack_graph()

            # Find the appropriate set for this graph
            n_nodes = graph.number_of_nodes()
            graph_set = None
            i_set = 0
            while i_set < len(self.set_sizes) and graph_set is None:
                set_max_n_nodes = self.set_max_n_nodes[i_set]
                if set_max_n_nodes is None or n_nodes < set_max_n_nodes:
                    graph_set = i_set
                i_set += 1

            print("With {} nodes, this graph belongs to set {}".format(
                n_nodes, graph_set))

            # Compute the number of existing graphs for each set
            current_set_sizes = [len(set) for set in sets]

            # Check whether we can add the graph or not
            if current_set_sizes[graph_set] < self.set_sizes[graph_set]:
                print("The graph has been added to the set")
                sets[graph_set].append(graph)
                current_set_sizes[graph_set] += 1
            else:
                print("The graph hasn't been added to the set")

            # Check whether all sets are filled or not
            are_sets_filled = current_set_sizes == self.set_sizes

            print("Current set sizes: {}\n".format(" ".join(
                [str(set_size) for set_size in current_set_sizes])))

            # Change the complexity for the next iteration
            if complexity == max_complexity:
                complexity = min_complexity
            else:
                complexity += 1

        # Save the graphs
        i_graph = 0
        for i_set, set in enumerate(sets):
            # Create the set folder
            path = self.paths[i_set]
            utils.create_folders(path)

            # Save the graphs in the order of their number of nodes
            graphs_n_nodes = [graph.number_of_nodes() for graph in set]
            order = np.argsort(graphs_n_nodes)
            for position in order:
                graph = set[position]
                graph.save("{}/{}.json".format(path, i_graph))
                i_graph += 1

    def load(self, i_graph: int) -> AttackGraph:
        path = [
            file
            for file in Path(self.base_path).glob("*/{}.json".format(i_graph))
        ][0]
        graph = AttackGraph()
        graph.load(path)
        return graph

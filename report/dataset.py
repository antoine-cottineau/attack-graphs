import json
import numpy as np
import utils
from attack_graph import DependencyAttackGraph, StateAttackGraph
from generation import Generator
from pathlib import Path
from time import time
from typing import Dict, List


class Dataset:
    set_sizes = [40, 40, 20]
    set_max_n_nodes = [1000, 10000, None]
    n_graphs = sum(set_sizes)
    base_path = "methods_input/dataset"
    summary_file_path = Path(base_path, "summary.json")
    min_n_exploits = 15
    max_n_exploits = 30

    @staticmethod
    def complete_dataset():
        # Create the necessary folders
        utils.create_folders(Dataset.base_path)

        # Start generating the graphs
        Dataset._add_one_pair_graphs(Dataset.min_n_exploits)

    @staticmethod
    def load_state_graph(i_graph: int) -> StateAttackGraph:
        graph_summary = Dataset._get_graph_pair_summary(i_graph)

        # Load the graph
        path = Path(graph_summary["state_path"])
        graph = StateAttackGraph()
        graph.load(path)
        return graph

    @staticmethod
    def load_dependency_graph(i_graph: int) -> DependencyAttackGraph:
        graph_summary = Dataset._get_graph_pair_summary(i_graph)

        # Load the graph
        path = Path(graph_summary["dependency_path"])
        graph = DependencyAttackGraph()
        graph.load(path)
        return graph

    @staticmethod
    def _add_one_pair_graphs(n_exploits: int):
        # Get current set populations
        set_populations = Dataset._get_current_set_populations()
        print("\nCurrent set populations: {}".format(" ".join(
            [str(i) for i in set_populations])))

        # If there is enough graphs, we stop generating
        if sum(set_populations) == Dataset.n_graphs:
            print("Generation done")
            return

        # Generate the graphs
        print("Generate graphs with {} exploits".format(n_exploits))
        generator = Generator(n_exploits=n_exploits)
        graphs = generator.generate_both_graphs()
        state_attack_graph, dependency_attack_graph = graphs

        # Get the appropriate set for these graphs
        n_nodes = state_attack_graph.number_of_nodes()
        appropriate_set = Dataset._find_appropriate_set(n_nodes)
        print("With {} state nodes, these graphs belong to set {}".format(
            n_nodes, appropriate_set))

        # Save the graphs if there is still room in the set
        if set_populations[appropriate_set] < Dataset.set_sizes[
                appropriate_set]:
            print("There is still room remaining in set {}".format(
                appropriate_set))

            print("Saving the graphs")
            Dataset._save_graphs(state_attack_graph, dependency_attack_graph,
                                 n_nodes, appropriate_set)

            # Print the updated set populations
            set_populations = Dataset._get_current_set_populations()
            print("Current set populations: {}".format(" ".join(
                [str(i) for i in set_populations])))
        else:
            print(
                "No room remaining in set {}, the graphs aren't saved".format(
                    appropriate_set))

        # Update the complexity for the creation of the next pair of graphs
        if n_exploits == Dataset.max_n_exploits:
            new_n_exploits = Dataset.min_n_exploits
        else:
            new_n_exploits = n_exploits + 1

        # Create a new graph with the new complexity
        Dataset._add_one_pair_graphs(new_n_exploits)

    @staticmethod
    def _get_current_set_populations() -> List[int]:
        # Get the current list of graphs that have been created
        graph_summaries = Dataset._get_summary_file_content()

        # Fill the set populations
        set_populations = [0] * len(Dataset.set_sizes)
        for graph_summary in graph_summaries:
            n_nodes = graph_summary["n_nodes"]
            set = Dataset._find_appropriate_set(n_nodes)
            set_populations[set] += 1

        return set_populations

    @staticmethod
    def _save_graphs(state_attack_graph: StateAttackGraph,
                     dependency_attack_graph: DependencyAttackGraph,
                     n_nodes: int, appropriate_set: int):
        # Create a base filename based on the current timestamp
        base_filename = str(time()).replace(".", "")[:13]

        # Save the state graph
        state_path = Path(Dataset.base_path, base_filename + "_state.json")
        state_attack_graph.save(state_path)

        # Save the dependency graph
        dependency_path = Path(Dataset.base_path,
                               base_filename + "_dependency.json")
        dependency_attack_graph.save(dependency_path)

        # Get the current list of graphs that have been created
        graph_summaries = Dataset._get_summary_file_content()
        new_summary_dict = dict(state_path=str(state_path),
                                dependency_path=str(dependency_path),
                                n_nodes=n_nodes,
                                set=appropriate_set)
        graph_summaries.append(new_summary_dict)

        # Save the updated list of graphs
        with open(Dataset.summary_file_path, "w") as f:
            json.dump(graph_summaries, f, indent=2)

    @staticmethod
    def _get_summary_file_content() -> List[dict]:
        if not Dataset.summary_file_path.exists():
            return []

        with open(Dataset.summary_file_path, "r") as f:
            result = json.load(f)

        return result

    @staticmethod
    def _find_appropriate_set(n_nodes: int) -> int:
        appropriate_set = None
        i_set = 0
        while i_set < len(Dataset.set_sizes) and appropriate_set is None:
            max_n_nodes = Dataset.set_max_n_nodes[i_set]
            if max_n_nodes is None or n_nodes < max_n_nodes:
                appropriate_set = i_set
            i_set += 1
        return appropriate_set

    @staticmethod
    def _get_graph_pair_summary(i_graph: int) -> dict:
        graph_summaries = Dataset._get_summary_file_content()

        # Fill a list of number of nodes
        list_n_nodes = np.zeros(len(graph_summaries))
        for i, graph_summary in enumerate(graph_summaries):
            n_nodes = graph_summary["n_nodes"]
            list_n_nodes[i] = n_nodes

        # Order the graph by number of nodes
        argsort = np.argsort(list_n_nodes)

        # Get the i-th graph summary in terms of number of nodes
        graph_summary = graph_summaries[argsort[i_graph]]
        return graph_summary


class HomerDataset:

    path = Path("methods_input/homer_dataset")
    n_graphs = 50

    def __init__(self):
        utils.create_folders(HomerDataset.path)

    def generate():
        for i_graph in range(HomerDataset.n_graphs):
            # Generate the graph
            print("Generating graph {}".format(i_graph))
            generator = Generator(
                exploits_prob_n_predecessors=HomerDataset._generate_probs(),
                propositions_prob_n_successors=HomerDataset._generate_probs())
            graph = generator.generate_dependency_attack_graph()

            print("The graph has {} branch nodes".format(
                len(graph.get_branch_nodes())))

            # Save the graph
            graph.save(Path(HomerDataset.path, "{}.json".format(i_graph)))

    def _generate_probs(size: int = 4) -> Dict[int, float]:
        probs = np.random.randint(10, size=size)
        probs = probs.astype(float)
        probs /= np.sum(probs)
        probs = dict([(i, probs[i]) for i in range(size)])
        return probs

    def load(i_graph: int) -> DependencyAttackGraph:
        path = Path(HomerDataset.path, "{}.json".format(i_graph))
        graph = DependencyAttackGraph()
        graph.load(path)
        return graph

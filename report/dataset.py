import json
import numpy as np
import utils
from attack_graph import DependencyAttackGraph, StateAttackGraph
from attack_graph_generation import Generator
from pathlib import Path
from time import time
from typing import List


class Dataset:
    set_sizes = [40, 40, 20]
    set_max_n_nodes = [1000, 10000, None]
    n_graphs = sum(set_sizes)
    base_path = "methods_input/dataset"
    summary_file_path = Path(base_path, "summary.json")

    def complete_dataset(self):
        # Create the necessary folders
        utils.create_folders(Dataset.base_path)

        # Start generating the graphs
        self._add_one_pair_graphs(20)

    def load_state_graph(self, i_graph: int) -> StateAttackGraph:
        graph_summary = self._get_graph_pair_summary(i_graph)

        # Load the graph
        path = Path(graph_summary["state_path"])
        graph = StateAttackGraph()
        graph.load(path)
        return graph

    def load_dependency_graph(self, i_graph: int) -> DependencyAttackGraph:
        graph_summary = self._get_graph_pair_summary(i_graph)

        # Load the graph
        path = Path(graph_summary["dependency_path"])
        graph = DependencyAttackGraph()
        graph.load(path)
        return graph

    def _add_one_pair_graphs(self, complexity: int):
        min_complexity = 20
        max_complexity = 40

        # Get current set populations
        set_populations = self._get_current_set_populations()
        print("\nCurrent set populations: {}".format(" ".join(
            [str(i) for i in set_populations])))

        # If there is enough graphs, we stop generating
        if sum(set_populations) == Dataset.n_graphs:
            print("Generation done")
            return

        # Generate a state attack graph
        print("Generate a state attack graph with complexity {}".format(
            complexity))
        generator = Generator(n_propositions=complexity, n_exploits=complexity)
        generator.generate_propositions_and_exploits()
        stateAttackGraph = generator.generate_state_attack_graph()

        # Get the appropriate set for these graphs
        n_nodes = stateAttackGraph.number_of_nodes()
        appropriate_set = self._find_appropriate_set(n_nodes)
        print("With {} state nodes, these graphs belong to set {}".format(
            n_nodes, appropriate_set))

        # Save the graphs if there is still room in the set
        if set_populations[appropriate_set] < Dataset.set_sizes[
                appropriate_set]:
            print("There is still room remaining in set {}, generating"
                  " dependency attack graph".format(appropriate_set))
            dependencyAttackGraph = generator.generate_dependency_attack_graph(
            )

            print("Saving the graphs")
            self._save_graphs(stateAttackGraph, dependencyAttackGraph, n_nodes,
                              appropriate_set)

            # Print the updated set populations
            set_populations = self._get_current_set_populations()
            print("Current set populations: {}".format(" ".join(
                [str(i) for i in set_populations])))
        else:
            print(
                "No room remaining in set {}, the graphs aren't saved".format(
                    appropriate_set))

        # Update the complexity for the creation of the next pair of graphs
        if complexity == max_complexity:
            new_complexity = min_complexity
        else:
            new_complexity = complexity + 1

        # Create a new graph with the new complexity
        self._add_one_pair_graphs(new_complexity)

    def _get_current_set_populations(self) -> List[int]:
        # Get the current list of graphs that have been created
        graph_summaries = self._get_summary_file_content()

        # Fill the set populations
        set_populations = [0] * len(Dataset.set_sizes)
        for graph_summary in graph_summaries:
            n_nodes = graph_summary["n_nodes"]
            set = self._find_appropriate_set(n_nodes)
            set_populations[set] += 1

        return set_populations

    def _save_graphs(self, stateAttackGraph: StateAttackGraph,
                     dependencyAttackGraph: DependencyAttackGraph,
                     n_nodes: int, appropriate_set: int):
        # Create a base filename based on the current timestamp
        base_filename = str(time()).replace(".", "")[:13]

        # Save the state graph
        state_path = Path(Dataset.base_path, base_filename + "_state.json")
        stateAttackGraph.save(state_path)

        # Save the dependency graph
        dependency_path = Path(Dataset.base_path,
                               base_filename + "_dependency.json")
        dependencyAttackGraph.save(dependency_path)

        # Get the current list of graphs that have been created
        graph_summaries = self._get_summary_file_content()
        new_summary_dict = dict(state_path=str(state_path),
                                dependency_path=str(dependency_path),
                                n_nodes=n_nodes,
                                set=appropriate_set)
        graph_summaries.append(new_summary_dict)

        # Save the updated list of graphs
        with open(Dataset.summary_file_path, "w") as f:
            json.dump(graph_summaries, f, indent=2)

    def _get_summary_file_content(self) -> List[dict]:
        if not Dataset.summary_file_path.exists():
            return []

        with open(Dataset.summary_file_path, "r") as f:
            result = json.load(f)

        return result

    def _find_appropriate_set(self, n_nodes: int) -> int:
        appropriate_set = None
        i_set = 0
        while i_set < len(Dataset.set_sizes) and appropriate_set is None:
            max_n_nodes = Dataset.set_max_n_nodes[i_set]
            if max_n_nodes is None or n_nodes < max_n_nodes:
                appropriate_set = i_set
            i_set += 1
        return appropriate_set

    def _get_graph_pair_summary(self, i_graph: int) -> dict:
        graph_summaries = self._get_summary_file_content()

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

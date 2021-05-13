import numpy as np
import utils
from attack_graph import AttackGraph
from clustering.clustering import ClusteringMethod
from clustering.white_smyth import Spectral1, Spectral2
from embedding.deepwalk import DeepWalk
from embedding.embedding import EmbeddingMethod
from embedding.graphsage import GraphSage
from embedding.hope import Hope
from matplotlib.pyplot import subplots
from pathlib import Path
from report.dataset import Dataset
from time import time
from typing import List, Tuple


class MethodsOptimizer:
    def __init__(self) -> None:
        self.dataset = Dataset()
        self.methods = ["DeepWalk", "GraphSAGE", "HOPE"]
        self.metrics = [
            "Modularity", "Mean silhouette index", "Mean conductance",
            "Mean coverage"
        ]

    def optimize(self, only_plot=False):
        parameters = [[("Embedding dimension", [8, 16, 32, 64, 128]),
                       ("Walk length", [20, 40, 80]),
                       ("Window size", [3, 5, 7])],
                      [("Embedding dimension", [8, 16, 32, 64, 128]),
                       ("Hidden layer dimension", [8, 16, 32, 64, 128])],
                      [("Embedding dimension", [8, 16, 32, 64, 128]),
                       ("Measurement", ["cn", "katz", "pagerank", "aa"])]]

        for i_method, method in enumerate(self.methods):
            for parameter in parameters[i_method]:
                print("Optimizing parameter {} of method {}".format(
                    parameter[0], method))
                if not only_plot:
                    self.optimize_embedding_method(method, parameter)
                self.plot_method_optimization(method, parameter)

    def optimize_embedding_method(self, method: str,
                                  parameter: Tuple[str, List[int]]):
        results = np.zeros(
            (self.dataset.n_graphs, len(self.metrics), len(parameter[1])))

        # For each graph, apply the method with each possible value of the
        # parameter and measure the resulting clustering with each metric
        for i_graph in range(self.dataset.n_graphs):
            graph = self.dataset.load(i_graph)

            # Apply the method
            print("Applying {} on graph {}/{} ({} nodes)".format(
                method, i_graph + 1, self.dataset.n_graphs,
                graph.number_of_nodes()))
            if method == "DeepWalk":
                method_results = self.apply_deepwalk(graph, parameter)
            elif method == "GraphSAGE":
                method_results = self.apply_graphsage(graph, parameter)
            elif method == "HOPE":
                method_results = self.apply_hope(graph, parameter)

            # Store the results
            results[i_graph] = method_results

        # Save the results
        path = Path(
            "report/data", "{}/{}_{}.npy".format(utils.sanitize(method),
                                                 utils.sanitize(method),
                                                 utils.sanitize(parameter[0])))
        utils.create_parent_folders(path)
        np.save(path, results)

    def plot_method_optimization(self, method: str,
                                 parameter: Tuple[str, List[int]]):
        path = Path(
            "report/data", "{}/{}_{}.npy".format(utils.sanitize(method),
                                                 utils.sanitize(method),
                                                 utils.sanitize(parameter[0])))
        results = np.load(path)

        files = [
            Path(
                "report/figures",
                "{}_{}_{}.png".format(utils.sanitize(method),
                                      utils.sanitize(parameter[0]),
                                      utils.sanitize(metric)))
            for metric in self.metrics
        ]

        # Create histograms comparing the performance of the method for
        # various values of the parameter
        bar_width = 0.2
        for i_metric, metric in enumerate(self.metrics):
            fig, ax = subplots()

            # Split the results by set
            n_bars = len(self.dataset.set_sizes)
            sets_results = [
                results[sum(self.dataset.set_sizes[:i]
                            ):sum(self.dataset.set_sizes[:i + 1]), i_metric, :]
                for i in range(n_bars)
            ]
            x = np.arange(n_bars)

            # Compute the position of the sub bars
            n_sub_bars = len(parameter[1])
            sub_bars_positions = np.arange(n_sub_bars,
                                           dtype=float) - n_sub_bars // 2
            sub_bars_positions *= bar_width
            if n_sub_bars % 2 == 0:
                sub_bars_positions += bar_width / 2

            for i_value, value in enumerate(parameter[1]):
                # Create the histogram thanks to the results
                y = [
                    set_result[:, i_value].mean()
                    for set_result in sets_results
                ]
                errors = [
                    set_result[:, i_value].std() for set_result in sets_results
                ]

                ax.bar(x + sub_bars_positions[i_value],
                       y,
                       bar_width,
                       yerr=errors,
                       label=str(value))

            # Add various information on the figure
            ax.set_xlabel("Group")
            ax.set_ylabel(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(["Small", "Medium", "Large"])
            ax.legend(title=parameter[0])

            # Save the figure
            utils.create_parent_folders(files[i_metric])
            fig.tight_layout()
            fig.savefig(files[i_metric])

    def apply_deepwalk(self, graph: AttackGraph,
                       parameter: Tuple[str, List[int]]) -> np.ndarray:
        results = np.zeros((len(self.metrics), len(parameter[1])))

        for i_value, value in enumerate(parameter[1]):
            if parameter[0] == "Embedding dimension":
                deepwalk = DeepWalk(graph, dim_embedding=value)
            elif parameter[0] == "Walk length":
                deepwalk = DeepWalk(graph, walk_length=value)
            elif parameter[0] == "Window size":
                deepwalk = DeepWalk(graph, window_size=value)

            print("Applying DeepWalk with parameter {} set to {}".format(
                parameter[0], value))
            results[:, i_value] = self.compute_all_metrics(deepwalk)

        return results

    def apply_graphsage(self, graph: AttackGraph,
                        parameter: Tuple[str, List[int]]) -> np.ndarray:
        results = np.zeros((len(self.metrics), len(parameter[1])))

        for i_value, value in enumerate(parameter[1]):
            if parameter[0] == "Embedding dimension":
                graphsage = GraphSage(graph, dim_embedding=value)
            elif parameter[0] == "Hidden layer dimension":
                graphsage = GraphSage(graph, dim_hidden_layer=value)

            print("Applying GraphSage with parameter {} set to {}".format(
                parameter[0], value))
            results[:, i_value] = self.compute_all_metrics(graphsage)

        return results

    def apply_hope(self, graph: AttackGraph,
                   parameter: Tuple[str, List[int]]) -> np.ndarray:
        results = np.zeros((len(self.metrics), len(parameter[1])))

        for i_value, value in enumerate(parameter[1]):
            if parameter[0] == "Embedding dimension":
                hope = Hope(graph, dim_embedding=value)
            elif parameter[0] == "Hidden layer dimension":
                hope = Hope(graph, dim_hidden_layer=value)

            print("Applying HOPE with parameter {} set to {}".format(
                parameter[0], value))
            results[:, i_value] = self.compute_all_metrics(hope)

        return results

    def compute_all_metrics(self, method: EmbeddingMethod) -> np.ndarray:
        results = np.zeros(len(self.metrics))

        method.embed()
        method.cluster()

        for i_metric in range(len(self.metrics)):
            if i_metric == 0:
                result = method.evaluate_modularity()
            elif i_metric == 1:
                result = method.evaluate_mean_silhouette_index()
            elif i_metric == 2:
                result = method.evaluate_mean_conductance()
            elif i_metric == 3:
                result = method.evaluate_mean_coverage()
            results[i_metric] = result

        return results


class MethodsComparator:
    def __init__(self):
        self.dataset = Dataset()
        self.methods = [
            "Spectral1", "Spectral2", "DeepWalk", "GraphSAGE", "HOPE"
        ]
        self.metrics = [
            "Modularity", "Mean silhouette index", "Mean conductance",
            "Mean coverage"
        ]

    def compare(self, only_plot=False):
        if not only_plot:
            self.apply_comparison()
        self.plot_comparison()

    def apply_comparison(self):
        results = np.zeros(
            (self.dataset.n_graphs, len(self.methods), len(self.metrics)))

        for i_graph in range(self.dataset.n_graphs):
            graph = self.dataset.load(i_graph)
            print("Comparing methods on graph {}/{} ({} nodes)".format(
                i_graph + 1, self.dataset.n_graphs, graph.number_of_nodes()))

            # Create an instance of each method with default parameters
            methods: List[ClusteringMethod] = [
                Spectral1(graph),
                Spectral2(graph),
                DeepWalk(graph),
                GraphSage(graph),
                Hope(graph)
            ]

            # Apply clustering with each method
            for i_method, method in enumerate(methods):
                print("Applying {}".format(self.methods[i_method]))
                if isinstance(method, EmbeddingMethod):
                    method.embed()

                method.cluster()

                # Apply each metric
                for i_metric in range(len(self.metrics)):
                    if i_metric == 0:
                        result = method.evaluate_modularity()
                    elif i_metric == 1:
                        result = method.evaluate_mean_silhouette_index()
                    elif i_metric == 2:
                        result = method.evaluate_mean_conductance()
                    elif i_metric == 3:
                        result = method.evaluate_mean_coverage()
                    results[i_graph, i_method, i_metric] = result

        # Save the results
        path = Path("report/data/comparison_methods.npy")
        utils.create_parent_folders(path)
        np.save(path, results)

    def plot_comparison(self):
        path = Path("report/data/comparison_methods.npy")

        results = np.load(path)

        files = [
            Path("report/figures",
                 "comparison_{}.png".format(utils.sanitize(metric)))
            for metric in self.metrics
        ]

        # Create histograms comparing the performance of the methods
        bar_width = 0.2
        for i_metric, metric in enumerate(self.metrics):
            fig, ax = subplots()

            # Split the results by set
            n_bars = len(self.dataset.set_sizes)
            sets_results = [
                results[sum(self.dataset.set_sizes[:i]
                            ):sum(self.dataset.set_sizes[:i + 1]), :, i_metric]
                for i in range(n_bars)
            ]
            x = np.arange(n_bars)

            # Compute the position of the sub bars
            n_sub_bars = len(self.methods)
            sub_bars_positions = np.arange(n_sub_bars,
                                           dtype=float) - n_sub_bars // 2
            sub_bars_positions *= bar_width
            if n_sub_bars % 2 == 0:
                sub_bars_positions += bar_width / 2

            for i_method, method in enumerate(self.methods):
                # Create the histogram thanks to the results
                y = [
                    set_result[:, i_method].mean()
                    for set_result in sets_results
                ]
                errors = [
                    set_result[:, i_method].std()
                    for set_result in sets_results
                ]

                ax.bar(x + sub_bars_positions[i_method],
                       y,
                       bar_width,
                       yerr=errors,
                       label=str(method))

            # Add various information on the figure
            ax.set_xlabel("Group")
            ax.set_ylabel(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(["Small", "Medium", "Large"])
            ax.legend(title="Method")

            # Save the figure
            utils.create_parent_folders(files[i_metric])
            fig.tight_layout()
            fig.savefig(files[i_metric])

    def compare_execution_times(self, only_plot=False):
        if not only_plot:
            self.apply_execution_times_comparison()
        self.plot_execution_time_comparison()

    def apply_execution_times_comparison(self):
        results = np.zeros((self.dataset.n_graphs, len(self.methods)))

        # for i_graph in range(self.dataset.n_graphs):
        for i_graph in range(15):
            graph = self.dataset.load(i_graph)
            print("Comparing execution times on graph {}/{} ({} nodes)".format(
                i_graph + 1, self.dataset.n_graphs, graph.number_of_nodes()))

            # Create an instance of each method with default parameters
            methods: List[ClusteringMethod] = [
                Spectral1(graph),
                Spectral2(graph),
                DeepWalk(graph),
                GraphSage(graph),
                Hope(graph)
            ]

            # Apply clustering with each method
            for i_method, method in enumerate(methods):
                print("Applying {}".format(self.methods[i_method]))
                starting_time = time()
                if isinstance(method, EmbeddingMethod):
                    method.embed()

                method.cluster()
                results[i_graph, i_method] = time() - starting_time

        # Save the results
        path = Path("report/data/comparison_execution_times.npy")
        utils.create_parent_folders(path)
        np.save(path, results)

    def plot_execution_time_comparison(self):
        path = Path("report/data/comparison_execution_times.npy")

        results = np.load(path)

        # Create a histogram comparing the execution time of the methods
        bar_width = 0.2
        fig, ax = subplots()

        # Split the results by set
        n_bars = len(self.dataset.set_sizes)
        sets_results = [
            results[sum(self.dataset.set_sizes[:i]):sum(self.dataset.
                                                        set_sizes[:i + 1]), :]
            for i in range(n_bars)
        ]
        x = np.arange(n_bars)

        # Compute the position of the sub bars
        n_sub_bars = len(self.methods)
        sub_bars_positions = np.arange(n_sub_bars,
                                       dtype=float) - n_sub_bars // 2
        sub_bars_positions *= bar_width
        if n_sub_bars % 2 == 0:
            sub_bars_positions += bar_width / 2

        for i_method, method in enumerate(self.methods):
            # Create the histogram thanks to the results
            y = [set_result[:, i_method].sum() for set_result in sets_results]

            ax.bar(x + sub_bars_positions[i_method],
                   y,
                   bar_width,
                   label=str(method))

        # Add various information on the figure
        ax.set_xlabel("Group")
        ax.set_ylabel("Total execution time (s)")
        ax.set_xticks(x)
        ax.set_xticklabels(["Small", "Medium", "Large"])
        ax.legend(title="Method")

        # Save the figure
        output_path = Path("report/figures/comparison_execution_times.png")
        utils.create_parent_folders(output_path)
        fig.tight_layout()
        fig.savefig(output_path)

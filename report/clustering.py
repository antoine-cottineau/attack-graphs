import numpy as np
import utils
from attack_graph import StateAttackGraph
from clustering.white_smyth import Spectral1, Spectral2
from embedding.deepwalk import DeepWalk
from embedding.embedding import EmbeddingMethod
from embedding.graphsage import GraphSage
from embedding.hope import Hope
from matplotlib.pyplot import subplots
from pathlib import Path
from report.dataset import Dataset
from time import time
from typing import Dict, List


class ClusteringFigureCreator:
    PATH_RESULTS_DATA = Path("report/data")
    PATH_RESULTS_FIGURES = Path("report/figures")

    METHODS: Dict[str, Dict[str, list]] = {
        "Spectral 1": None,
        "Spectral 2": None,
        "DeepWalk": {
            "Embedding dimension": [8, 16, 32, 64, 128],
            "Walk length": [20, 40, 80],
            "Window size": [3, 5, 7]
        },
        "GraphSAGE": {
            "Embedding dimension": [8, 16, 32, 64, 128],
            "Hidden layer dimension": [8, 16, 32, 64, 128]
        },
        "HOPE": {
            "Embedding dimension": [8, 16, 32, 64, 128],
            "Measurement": ["cn", "katz", "pagerank", "aa"]
        }
    }

    METRICS = [
        "Modularity", "Mean silhouette index", "Mean conductance",
        "Mean coverage"
    ]

    @staticmethod
    def _apply_method(graph: StateAttackGraph, method: str, parameter: str,
                      values: list, metrics: list) -> np.ndarray:
        results = np.zeros((len(values), len(metrics)))
        for i_value, value in enumerate(values):
            print("Applying {} with {} set to {}".format(
                method, parameter, value))
            # Apply the method
            if method == "Spectral 1":
                clustering = Spectral1(graph)
            elif method == "Spectral 2":
                clustering = Spectral2(graph)
            elif method == "DeepWalk":
                if parameter == "Embedding dimension":
                    clustering = DeepWalk(graph, dim_embedding=value)
                elif parameter == "Walk length":
                    clustering = DeepWalk(graph, walk_length=value)
                elif parameter == "Window size":
                    clustering = DeepWalk(graph, window_size=value)
                else:
                    clustering = DeepWalk(graph)
            elif method == "GraphSAGE":
                if parameter == "Embedding dimension":
                    clustering = GraphSage(graph, dim_embedding=value)
                elif parameter == "Hidden layer dimension":
                    clustering = GraphSage(graph, dim_hidden_layer=value)
                else:
                    clustering = GraphSage(graph)
            elif method == "HOPE":
                if parameter == "Embedding dimension":
                    clustering = Hope(graph, dim_embedding=value)
                elif parameter == "Measurement":
                    clustering = Hope(graph, measurement=value)
                else:
                    clustering = Hope(graph)

            # Embed if the method is an embedding method
            try:
                if isinstance(clustering, EmbeddingMethod):
                    clustering.embed()

                clustering.cluster()

                # Evaluate the clustering with the various metrics
                for i_metric, metric in enumerate(metrics):
                    if metric == "Modularity":
                        result = clustering.evaluate_modularity()
                    elif metric == "Mean silhouette index":
                        result = clustering.evaluate_mean_silhouette_index()
                    elif metric == "Mean conductance":
                        result = clustering.evaluate_mean_conductance()
                    elif metric == "Mean coverage":
                        result = clustering.evaluate_mean_coverage()
                    results[i_value, i_metric] = result

            except Exception:
                print("Couldn't apply {}".format(method))

        return results

    @staticmethod
    def _plot_histogram(results: np.ndarray, legend_title: str,
                        bar_labels: List[int], y_label: str, filename: str):
        fig, ax = subplots()

        n_bars = results.shape[1]

        # Compute the relative position of the bars
        bar_width = 1 / (n_bars + 1)
        bar_positions = np.arange(n_bars, dtype=float) - n_bars // 2
        bar_positions *= bar_width
        if n_bars % 2 == 0:
            bar_positions += bar_width / 2

        for i_bar in range(n_bars):
            x = np.arange(len(Dataset.set_sizes)) + bar_positions[i_bar]
            y = results[:, i_bar]

            ax.bar(x, y, bar_width, label=bar_labels[i_bar])

        # Add other information to the figure
        ax.set_xlabel("Group")
        ax.set_xticks(np.arange(len(Dataset.set_sizes)))
        ax.set_xticklabels(["Small", "Medium", "Large"])
        ax.set_ylabel(y_label)
        ax.legend(title=legend_title)

        # Save the figure
        utils.create_folders(ClusteringFigureCreator.PATH_RESULTS_FIGURES)
        path = Path(ClusteringFigureCreator.PATH_RESULTS_FIGURES,
                    filename + ".png")
        fig.tight_layout()
        fig.savefig(path)
        fig.clf()


class EmbeddingMethodOptimizer(ClusteringFigureCreator):
    @staticmethod
    def apply(n_graphs: int = None):
        for method, parameters in EmbeddingMethodOptimizer.METHODS.items():
            if parameters is None:
                continue

            for parameter in parameters:
                print("Optimizing parameter {} of method {}".format(
                    parameter, method))
                EmbeddingMethodOptimizer._optimize(method, parameter, n_graphs)
                EmbeddingMethodOptimizer._plot(method, parameter)

    @staticmethod
    def _optimize(method: str, parameter: str, n_graphs: int):
        # Obtain the existing results if they exist
        data_file_path = EmbeddingMethodOptimizer._get_data_file_path(
            method, parameter)
        if data_file_path.exists():
            results: np.ndarray = np.load(data_file_path)
            i_graph = results.shape[0]
            if i_graph >= Dataset.n_graphs or (n_graphs is not None
                                               and i_graph >= n_graphs):
                # All the graphs have been taken care of
                return
        else:
            results = None
            i_graph = 0

        # Load the graph
        graph = Dataset.load_state_graph(i_graph)

        # Perform the optimization
        print("Optimizing graph {}/{} with {} nodes".format(
            i_graph + 1, Dataset.n_graphs, graph.number_of_nodes()))
        values = EmbeddingMethodOptimizer.METHODS[method][parameter]
        metrics = EmbeddingMethodOptimizer.METRICS
        graph_results = EmbeddingMethodOptimizer._apply_method(
            graph, method, parameter, values, metrics)

        # Add the graph results to the results array and save it
        graph_results = np.expand_dims(graph_results, axis=0)
        if results is None:
            results = graph_results
        else:
            results = np.concatenate((results, graph_results))
        utils.create_folders(EmbeddingMethodOptimizer.PATH_RESULTS_DATA)
        np.save(data_file_path, results)

        # Call the function recursively with the same parameters
        EmbeddingMethodOptimizer._optimize(method, parameter, n_graphs)

    @staticmethod
    def _plot(method: str, parameter: str):
        data_file_path = EmbeddingMethodOptimizer._get_data_file_path(
            method, parameter)
        results: np.ndarray = np.load(data_file_path)

        # Create one histogram for each metric
        for i_metric, metric in enumerate(EmbeddingMethodOptimizer.METRICS):
            if results.shape[0] < Dataset.n_graphs:
                # Complete the results array with zeros
                results = np.concatenate(
                    (results,
                     np.zeros((Dataset.n_graphs - results.shape[0],
                               results.shape[1], results.shape[2]))))
            metric_results = np.array([
                np.mean(results[
                    sum(Dataset.set_sizes[:i]):sum(Dataset.set_sizes[:i +
                                                                     1]), :,
                    i_metric],
                        axis=0) for i in range(len(Dataset.set_sizes))
            ])
            values = EmbeddingMethodOptimizer.METHODS[method][parameter]
            filename = "{}_{}_{}".format(utils.sanitize(method),
                                         utils.sanitize(parameter),
                                         utils.sanitize(metric))

            EmbeddingMethodOptimizer._plot_histogram(metric_results, parameter,
                                                     values, metric, filename)

    @staticmethod
    def _get_data_file_path(method: str, parameter: str) -> Path:
        return Path(
            EmbeddingMethodOptimizer.PATH_RESULTS_DATA,
            "{}_{}.npy".format(utils.sanitize(method),
                               utils.sanitize(parameter)))


class MethodComparator(ClusteringFigureCreator):
    @staticmethod
    def apply(n_graphs: int = None):
        MethodComparator._compare(n_graphs)
        MethodComparator._plot()

    @staticmethod
    def _compare(n_graphs: int):
        # Obtain the existing results if they exist
        data_file_path = MethodComparator._get_data_file_path()
        if data_file_path.exists():
            results: np.ndarray = np.load(data_file_path)
            i_graph = results.shape[0]
            if i_graph >= Dataset.n_graphs or (n_graphs is not None
                                               and i_graph >= n_graphs):
                # All the graphs have been taken care of
                return
        else:
            results = None
            i_graph = 0

        # Load the graph
        graph = Dataset.load_state_graph(i_graph)

        # Perform the comparison
        print("Comparing methods on graph {}/{} with {} nodes".format(
            i_graph + 1, Dataset.n_graphs, graph.number_of_nodes()))
        metrics = MethodComparator.METRICS
        graph_results = np.zeros((len(MethodComparator.METHODS), len(metrics)))
        for i_method, method in enumerate(MethodComparator.METHODS):
            method_results = MethodComparator._apply_method(
                graph, method, None, [None], metrics)[0]
            graph_results[i_method] = method_results

        # Add the graph results to the results array and save it
        graph_results = np.expand_dims(graph_results, axis=0)
        if results is None:
            results = graph_results
        else:
            results = np.concatenate((results, graph_results))
        utils.create_folders(MethodComparator.PATH_RESULTS_DATA)
        np.save(data_file_path, results)

        # Call the function recursively with the same parameters
        MethodComparator._compare(n_graphs)

    @staticmethod
    def _plot():
        data_file_path = MethodComparator._get_data_file_path()
        results: np.ndarray = np.load(data_file_path)

        # Create one histogram for each metric
        for i_metric, metric in enumerate(EmbeddingMethodOptimizer.METRICS):
            if results.shape[0] < Dataset.n_graphs:
                # Complete the results array with zeros
                results = np.concatenate(
                    (results,
                     np.zeros((Dataset.n_graphs - results.shape[0],
                               results.shape[1], results.shape[2]))))
            metric_results = np.array([
                np.mean(results[
                    sum(Dataset.set_sizes[:i]):sum(Dataset.set_sizes[:i +
                                                                     1]), :,
                    i_metric],
                        axis=0) for i in range(len(Dataset.set_sizes))
            ])
            bar_labels = list(MethodComparator.METHODS.keys())
            filename = "comparison_{}".format(utils.sanitize(metric))

            MethodComparator._plot_histogram(metric_results, "Methods",
                                             bar_labels, metric, filename)

    @staticmethod
    def _get_data_file_path() -> Path:
        return Path(MethodComparator.PATH_RESULTS_DATA, "comparison.npy")


class TimeComparator(ClusteringFigureCreator):
    @staticmethod
    def apply(n_graphs: int = None):
        TimeComparator._compare(n_graphs)
        TimeComparator._plot()

    @staticmethod
    def _compare(n_graphs: int):
        # Obtain the existing results if they exist
        data_file_path = TimeComparator._get_data_file_path()
        if data_file_path.exists():
            results: np.ndarray = np.load(data_file_path)
            i_graph = results.shape[0]
            if i_graph >= Dataset.n_graphs or (n_graphs is not None
                                               and i_graph >= n_graphs):
                # All the graphs have been taken care of
                return
        else:
            results = None
            i_graph = 0

        # Load the graph
        graph = Dataset.load_state_graph(i_graph)

        # Perform the comparison
        print("Comparing time durations on graph {}/{} with {} nodes".format(
            i_graph + 1, Dataset.n_graphs, graph.number_of_nodes()))
        graph_results = np.zeros(len(TimeComparator.METHODS))
        for i_method, method in enumerate(TimeComparator.METHODS):
            start_time = time()
            TimeComparator._apply_method(graph, method, None, [None], [])[0]
            duration = time() - start_time
            graph_results[i_method] = duration

        # Add the graph results to the results array and save it
        graph_results = np.expand_dims(graph_results, axis=0)
        if results is None:
            results = graph_results
        else:
            results = np.concatenate((results, graph_results))
        utils.create_folders(TimeComparator.PATH_RESULTS_DATA)
        np.save(data_file_path, results)

        # Call the function recursively with the same parameters
        TimeComparator._compare(n_graphs)

    @staticmethod
    def _plot():
        data_file_path = TimeComparator._get_data_file_path()
        results: np.ndarray = np.load(data_file_path)

        # Create the histogram
        if results.shape[0] < Dataset.n_graphs:
            # Complete the results array with zeros
            results = np.concatenate(
                (results,
                 np.zeros(
                     (Dataset.n_graphs - results.shape[0], results.shape[1]))))
        metric_results = np.array([
            np.mean(results[
                sum(Dataset.set_sizes[:i]):sum(Dataset.set_sizes[:i + 1]), :],
                    axis=0) for i in range(len(Dataset.set_sizes))
        ])
        bar_labels = list(TimeComparator.METHODS.keys())
        filename = "time_comparison"

        TimeComparator._plot_histogram(metric_results, "Methods", bar_labels,
                                       "Time (s)", filename)

    @staticmethod
    def _get_data_file_path() -> Path:
        return Path(TimeComparator.PATH_RESULTS_DATA, "time_comparison.npy")

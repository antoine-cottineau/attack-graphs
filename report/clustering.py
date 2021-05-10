import matplotlib.pyplot as plt
import numpy as np
import utils
from attack_graph import AttackGraph
from clustering.clustering import ClusteringMethod
from clustering.white_smyth import Spectral1, Spectral2
from embedding.deepwalk import DeepWalk
from embedding.embedding import EmbeddingMethod
from embedding.graphsage import GraphSage
from embedding.hope import Hope
from pathlib import Path
from report.dataset import DatasetLoader
from time import time
from typing import List

METRICS = [
    "Modularity", "Mean silhouette index", "Mean conductance", "Mean coverage"
]


def compare_methods_by_metric(dl: DatasetLoader):
    methods = ["Spectral 1", "Spectral 2", "DeepWalk", "GraphSAGE", "HOPE"]

    n_graphs = len(dl.files)
    n_methods = len(methods)
    n_metrics = len(METRICS)

    # Create the array containing all the results
    results = np.zeros((n_graphs, n_methods, n_metrics))

    # For each graph, apply each method and apply each metric on the resulting
    # clustering
    for i_graph, graph in enumerate(dl):
        # Create an instance of each method
        methods: List[ClusteringMethod] = [
            Spectral1(graph),
            Spectral2(graph),
            DeepWalk(graph),
            GraphSage(graph),
            Hope(graph)
        ]

        # For each method which is an embedding method, call the embed function
        for method in methods:
            if isinstance(method, EmbeddingMethod):
                method.embed()

        # For each method, apply clustering and measure it with all the metrics
        for i_method, method in enumerate(methods):
            method.cluster()

            for i_metric, metric in enumerate(METRICS):
                results[i_graph, i_method,
                        i_metric] = applyMetric(method, metric)

    files = ["{}.png".format(sanitize(metric)) for metric in METRICS]

    # Create 4 histograms: one for each metric
    for i_metric, metric in enumerate(METRICS):
        fig, ax = plt.subplots()

        # Create the histogram thanks to the results
        x = np.arange(len(methods))

        metric_results: np.ndarray = results[:, :, i_metric]
        y = metric_results.mean(axis=0)
        deviations = metric_results.std(axis=0)

        ax.barh(x, y, tick_label=methods, xerr=deviations)

        # Add various information on the figure
        ax.set_xlabel(metric)
        ax.invert_yaxis()

        # Save the figure
        path = Path("report/figures", files[i_metric])
        utils.create_parent_folders(path)
        fig.tight_layout()
        fig.savefig(path)


def optimize_methods_parameters(max_n_files: int = None):
    dl = DatasetLoader(max_n_files=max_n_files)
    parameters = dict(deepwalk=dict(embedding_dimension=[8, 16, 32, 64, 128],
                                    walk_length=[20, 40, 80],
                                    window_size=[3, 5, 7]),
                      graphsage=dict(dim_embedding=[8, 16, 32, 64, 128],
                                     dim_hidden_layer=[8, 16, 32, 64, 128]),
                      hope=dict(dim_embedding=[8, 16, 32, 64, 128],
                                measurement=["cn", "katz", "pagerank", "aa"]))

    for method, method_parameters in parameters.items():
        for method_parameter, values in method_parameters.items():
            print("-" * 100)
            print("Optimizing parameter {} of method {}".format(
                method_parameter, method))
            print("-" * 100)
            dl.reinitialize()
            path = optimize_embedding_method(dl, method, method_parameter,
                                             values)
            plot_embedding_method_optimization(dl, method, method_parameter,
                                               values, path)


def optimize_embedding_method(dl: DatasetLoader, method: str, parameter: str,
                              values: list) -> Path:
    n_graphs = len(dl.files)
    n_metrics = len(METRICS)
    n_values = len(values)

    results = np.zeros((n_graphs, n_metrics, n_values))

    for i_graph, graph in enumerate(dl):
        print("Applying {} on graph {}/{} ({} nodes)".format(
            method, i_graph + 1, n_graphs, graph.number_of_nodes()))

        # Call the right method
        if method == "deepwalk":
            method_results = applyDeepWalk(graph, parameter, values)
        elif method == "graphsage":
            method_results = applyGraphSage(graph, parameter, values)
        else:
            method_results = applyHope(graph, parameter, values)

        # Store the results
        results[i_graph] = method_results

    path = Path("report/data/{}/{}_{}.npy".format(method, method, parameter))
    utils.create_parent_folders(path)
    np.save(path, results)
    return path


def plot_embedding_method_optimization(dl: DatasetLoader, method: str,
                                       parameter: str, values: list,
                                       path: Path):
    results = np.load(path)

    files = [
        "{}_{}_{}.png".format(method, sanitize(parameter), sanitize(metric))
        for metric in METRICS
    ]

    for i_metric, metric in enumerate(METRICS):
        fig, ax = plt.subplots()

        x = dl.number_of_nodes
        for i_value, value in enumerate(values):
            y: np.ndarray = results[:, i_metric, i_value]
            ax.plot(x, y, label=str(value))

        ax.set_xlabel("Number of nodes")
        ax.set_ylabel(metric)
        ax.legend(title=parameter)

        path = Path("report/figures/{}".format(method), files[i_metric])
        utils.create_parent_folders(path)
        fig.tight_layout()
        fig.savefig(path)


def applyDeepWalk(graph: AttackGraph, parameter: str,
                  values: list) -> np.ndarray:
    results = np.zeros((len(METRICS), len(values)))

    for i_value, value in enumerate(values):
        if parameter == "embedding_dimension":
            deepwalk = DeepWalk(graph, dim_embedding=value)
        elif parameter == "walk_length":
            deepwalk = DeepWalk(graph, walk_length=value)
        else:
            deepwalk = DeepWalk(graph, window_size=value)

        print("Applying DeepWalk with parameter {} set to {}".format(
            parameter, value))
        deepwalk.embed()
        deepwalk.cluster()

        for i_metric, metric in enumerate(METRICS):
            results[i_metric, i_value] = applyMetric(deepwalk,
                                                     sanitize(metric))

    return results


def applyGraphSage(graph: AttackGraph, parameter: str,
                   values: list) -> np.ndarray:
    results = np.zeros((len(METRICS), len(values)))

    for i_value, value in enumerate(values):
        if parameter == "dim_embedding":
            graphsage = GraphSage(graph, dim_embedding=value, device="cpu")
        else:
            graphsage = GraphSage(graph, dim_hidden_layer=value, device="cpu")

        print("Applying GraphSAGE with parameter {} set to {}".format(
            parameter, value))
        graphsage.embed()
        graphsage.cluster()

        for i_metric, metric in enumerate(METRICS):
            results[i_metric, i_value] = applyMetric(graphsage,
                                                     sanitize(metric))

    return results


def applyHope(graph: AttackGraph, parameter: str, values: list) -> np.ndarray:
    results = np.zeros((len(METRICS), len(values)))

    for i_value, value in enumerate(values):
        if parameter == "dim_embedding":
            hope = Hope(graph, dim_embedding=value)
        else:
            hope = Hope(graph, measurement=value)

        print("Applying HOPE with parameter {} set to {}".format(
            parameter, value))
        hope.embed()
        hope.cluster()

        for i_metric, metric in enumerate(METRICS):
            results[i_metric, i_value] = applyMetric(hope, sanitize(metric))

    return results


def applyMetric(method: ClusteringMethod, metric: str) -> float:
    if metric == "modularity":
        return method.evaluate_modularity()
    elif metric == "mean_silhouette_index":
        return method.evaluate_mean_silhouette_index()
    elif metric == "mean_conductance":
        return method.evaluate_mean_conductance()
    else:
        return method.evaluate_mean_coverage()


def sanitize(string: str):
    return string.replace(" ", "_").lower()


def measure_execution_times(dl: DatasetLoader):
    method_labels = [
        "Spectral 1", "Spectral 2", "DeepWalk", "GraphSAGE", "HOPE"
    ]

    i_graphs = [0, len(dl.files) // 2, len(dl.files)]
    graphs = [dl.get(i) for i in i_graphs]

    execution_times = np.zeros((len(method_labels), len(graphs)))

    for i_method, method_label in enumerate(method_labels):
        for i_graph, graph in graphs:
            if method_label == "Spectral 1":
                method = Spectral1(graph)
            elif method_label == "Spectral 2":
                method = Spectral2(graph)
            elif method_label == "DeepWalk":
                method = DeepWalk(graph)
            elif method_label == "GraphSAGE":
                method = GraphSage(graph)
            elif method_label == "HOPE":
                method = Hope(graph)

            start_time = time()

            if isinstance(method, EmbeddingMethod):
                method.embed()
            method.cluster()

            execution_times[i_method, i_graph] = time() - start_time

    path = Path("report/data/execution_times.npy")
    utils.create_parent_folders(path)
    np.save(path, execution_times)

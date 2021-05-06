from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import utils
from attack_graph_generation import Generator
from clustering.clustering import ClusteringMethod
from clustering.white_smyth import Spectral1, Spectral2
from embedding.deepwalk import DeepWalk
from embedding.embedding import EmbeddingMethod
from embedding.graphsage import GraphSage
from embedding.hope import Hope
from typing import List


def compare_methods_by_metric():
    n_graphs = 2
    n_methods = 5
    n_metrics = 4

    # Start by generating n_graphs graphs
    generator = Generator()
    graphs = [generator.generate() for _ in range(n_graphs)]

    # Create the array containing all the results
    results = np.zeros((n_graphs, n_methods, n_metrics))

    # For each graph, apply each method and apply each metric on the resulting
    # clustering
    for i_graph, graph in enumerate(graphs):
        # Create an instance of each method
        methods: List[ClusteringMethod] = [
            Spectral1(graph, K=10),
            Spectral2(graph, K=10),
            DeepWalk(graph, dim_embedding=16),
            GraphSage(graph,
                      dim_embedding=16,
                      dim_hidden_layer=16,
                      n_epochs=50),
            Hope(graph, dim_embedding=16, measurement="katz")
        ]

        # For each method which is an embedding method, call the embed function
        for method in methods:
            if isinstance(method, EmbeddingMethod):
                method.embed()

        # For each method, apply clustering and measure it with all the metrics
        for i_method, method in enumerate(methods):
            method.cluster()

            metrics = [
                method.evaluate_modularity,
                method.evaluate_mean_silhouette_index,
                method.evaluate_mean_conductance, method.evaluate_mean_coverage
            ]

            for i_metric, metric in enumerate(metrics):
                results[i_graph, i_method, i_metric] = metric()

    metrics = [
        "Modularity", "Mean silhouette index", "Mean conductance",
        "Mean coverage"
    ]
    methods = ["Spectral 1", "Spectral 2", "DeepWalk", "GraphSAGE", "HOPE"]
    files = [
        "modularity.png", "mean_silhouette_index.png", "mean_conductance.png",
        "mean_coverage.png"
    ]

    # Create 4 histograms: one for each metric
    for i_metric, metric in enumerate(metrics):
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

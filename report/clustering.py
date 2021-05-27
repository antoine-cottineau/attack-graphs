from clustering.clustering import ClusteringMethod
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

# CONSTANTS
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
    "Modularity", "Mean silhouette index", "Mean conductance", "Mean coverage"
]


class MethodApplicator:
    def __init__(self,
                 graph: StateAttackGraph,
                 method: str,
                 parameter: str,
                 values: list,
                 metrics: List[str],
                 use_gpu: bool = True):
        self.graph = graph
        self.method = method
        self.parameter = parameter
        self.values = values
        self.metrics = metrics
        self.use_gpu = use_gpu

    def apply_method(self) -> np.ndarray:
        if self.parameter is None:
            print("Applying {}".format(self.method))
            return self._apply_method_for_value(None)

        # Go through each value and apply the method
        results = np.zeros((len(self.values), len(self.metrics)))
        for i_value, value in enumerate(self.values):
            print("Applying {} with {} set to {}".format(
                self.method, self.parameter, value))
            results[i_value] = self._apply_method_for_value(value)
        return results

    def _apply_method_for_value(self, value) -> np.ndarray:
        instance = self._instantiate_method(value)

        # Apply the method
        has_crashed = False
        if isinstance(instance, EmbeddingMethod):
            # Sometimes the embedding crashes because the graph is too large.
            # In this case, return an array of zeros
            try:
                instance.embed()
            except Exception:
                has_crashed = True
        instance.cluster()

        # Evaluate the method
        if self.metrics is None:
            return None
        else:
            if has_crashed:
                print("Error when applying {}".format(self.method))
                return np.zeros(len(self.metrics))
            else:
                return self._apply_metrics(instance)

    def _instantiate_method(self, value) -> ClusteringMethod:
        if self.method == "Spectral 1":
            return Spectral1(self.graph)
        elif self.method == "Spectral 2":
            return Spectral2(self.graph)
        elif self.method == "DeepWalk":
            return self._instantiate_deepwalk(value)
        elif self.method == "GraphSAGE":
            return self._instantiate_graphsage(value)
        elif self.method == "HOPE":
            return self._instantiate_hope(value)

    def _apply_metrics(self, instance: ClusteringMethod) -> np.ndarray:
        results = np.zeros(len(self.metrics))
        for i_metric, metric in enumerate(self.metrics):
            if metric == "Modularity":
                results[i_metric] = instance.evaluate_modularity()
            elif metric == "Mean silhouette index":
                results[i_metric] = instance.evaluate_mean_silhouette_index()
            elif metric == "Mean conductance":
                results[i_metric] = instance.evaluate_mean_conductance()
            elif metric == "Mean coverage":
                results[i_metric] = instance.evaluate_mean_coverage()
        return results

    def _instantiate_deepwalk(self, value) -> DeepWalk:
        if self.parameter == "Embedding dimension":
            return DeepWalk(self.graph, dim_embedding=value)
        elif self.parameter == "Walk length":
            return DeepWalk(self.graph, walk_length=value)
        elif self.parameter == "Window size":
            return DeepWalk(self.graph, window_size=value)
        else:
            return DeepWalk(self.graph)

    def _instantiate_graphsage(self, value) -> GraphSage:
        device = None if self.use_gpu else "cpu"
        if self.parameter == "Embedding dimension":
            return GraphSage(self.graph, dim_embedding=value, device=device)
        elif self.parameter == "Hidden layer dimension":
            return GraphSage(self.graph, dim_hidden_layer=value, device=device)
        else:
            return GraphSage(self.graph, device=device)

    def _instantiate_hope(self, value) -> Hope:
        if self.parameter == "Embedding dimension":
            return Hope(self.graph, dim_embedding=value)
        elif self.parameter == "Measurement":
            return Hope(self.graph, measurement=value)
        else:
            return Hope(self.graph)


class Histogram:
    def __init__(self, results: np.ndarray, legend_title: str,
                 bar_labels: list, y_labels: list, filenames: List[str]):
        self.results = results
        self.legend_title = legend_title
        self.bar_labels = bar_labels
        self.y_labels = y_labels
        self.filenames = filenames

        self.n_graphs, self.n_bars = results.shape[:2]
        if len(results.shape) == 3:
            self.n_figures = results.shape[2]
        else:
            self.n_figures = 1

        utils.create_folders(PATH_RESULTS_FIGURES)

    def create(self):
        self._split_results_into_groups()
        self._compute_bar_width_and_positions()

        if self.n_figures == 1:
            self._create_figure(self.splitted_results, self.filenames[0],
                                self.y_labels[0])
        else:
            for i_figure in range(self.n_figures):
                results = [r[:, :, i_figure] for r in self.splitted_results]
                self._create_figure(results, self.filenames[i_figure],
                                    self.y_labels[i_figure])

    def _create_figure(self, results: List[np.ndarray], filename: str,
                       y_label: str):
        fig, ax = subplots()

        # Add the bars
        for i_bar in range(self.n_bars):
            x = np.arange(len(results)) + self.bar_positions[i_bar]
            y = []
            for split in results:
                bar_split: np.ndarray = split[:, i_bar]
                if bar_split.shape[0] == 0:
                    y.append(0)
                else:
                    y.append(np.mean(bar_split))

            ax.bar(x, y, self.bar_width, label=self.bar_labels[i_bar])

        # Add other information to the figure
        ax.set_xlabel("Group")
        ax.set_xticks(np.arange(len(results)))
        ax.set_xticklabels(["Small", "Medium", "Large"])
        ax.set_ylabel(y_label)
        ax.legend(title=self.legend_title)

        # Save the figure
        path = Path(PATH_RESULTS_FIGURES, filename + ".png")
        fig.tight_layout()
        fig.savefig(path)
        fig.clf()

    def _split_results_into_groups(self):
        splitted_results = []
        for i in range(len(Dataset.set_sizes)):
            start = sum(Dataset.set_sizes[:i])
            end = sum(Dataset.set_sizes[:i + 1])
            splitted_results.append(self.results[start:end])
        self.splitted_results = splitted_results

    def _compute_bar_width_and_positions(self):
        self.bar_width = 1 / (self.n_bars + 1)
        bar_positions = np.arange(self.n_bars, dtype=float) - self.n_bars // 2
        bar_positions *= self.bar_width
        if self.n_bars % 2 == 0:
            bar_positions += self.bar_width / 2
        self.bar_positions = bar_positions


class ClusteringFigureCreator:
    def __init__(self,
                 data_file_name: str,
                 n_graphs: int = None,
                 continuous_plotting: bool = True):
        self.data_file_path = Path(PATH_RESULTS_DATA, data_file_name + ".npy")
        self.n_graphs = n_graphs
        self.continuous_plotting = continuous_plotting

        utils.create_parent_folders(self.data_file_path)
        self._get_max_number_of_graphs()

    def apply(self):
        # Fetch the existing results
        existing_results = self._get_existing_results()
        if existing_results is None:
            i_graph = 0
        else:
            i_graph = existing_results.shape[0]
        if i_graph >= self.max_n_graphs:
            return

        # Load the next graph
        graph = Dataset.load_state_graph(i_graph)
        print("Graph {}/{}: {} nodes".format(i_graph + 1, Dataset.n_graphs,
                                             graph.number_of_nodes()))

        # Create and add the new results for this graph
        new_results = self._apply_for_graph(graph)
        self._append_and_save_new_results(existing_results, new_results)

        # Plot if required
        if self.continuous_plotting:
            self.plot()

        # Call the function recursively
        self.apply()

    def plot(self):
        pass

    def _apply_for_graph(self, graph: StateAttackGraph) -> np.ndarray:
        pass

    def _append_and_save_new_results(self, existing_results: np.ndarray,
                                     new_results: np.ndarray):
        results = np.expand_dims(new_results, axis=0)
        if existing_results is not None:
            results = np.concatenate((existing_results, results))
        np.save(self.data_file_path, results)

    def _get_existing_results(self) -> np.ndarray:
        if self.data_file_path.exists():
            return np.load(self.data_file_path)
        else:
            return None

    def _get_max_number_of_graphs(self):
        if self.n_graphs is None:
            self.max_n_graphs = Dataset.n_graphs
        else:
            self.max_n_graphs = min(self.n_graphs, Dataset.n_graphs)


class MethodOptimizer(ClusteringFigureCreator):
    def __init__(self,
                 method: str,
                 parameter: str,
                 n_graphs: int = None,
                 continuous_plotting: bool = True):
        data_file_name = "{}_{}".format(utils.sanitize(method),
                                        utils.sanitize(parameter))
        super().__init__(data_file_name, n_graphs, continuous_plotting)

        self.method = method
        self.parameter = parameter

    def _apply_for_graph(self, graph: StateAttackGraph) -> np.ndarray:
        return MethodApplicator(graph, self.method, self.parameter,
                                METHODS[self.method][self.parameter],
                                METRICS).apply_method()

    def plot(self):
        results = self._get_existing_results()
        if results is None:
            return

        filenames = [
            "{}_{}_{}".format(utils.sanitize(self.method),
                              utils.sanitize(self.parameter),
                              utils.sanitize(metric)) for metric in METRICS
        ]

        Histogram(results, self.parameter,
                  METHODS[self.method][self.parameter], METRICS,
                  filenames).create()


class MethodComparator(ClusteringFigureCreator):
    def __init__(self, n_graphs: int = None, continuous_plotting: bool = True):
        data_file_name = "comparison"
        super().__init__(data_file_name, n_graphs, continuous_plotting)

    def _apply_for_graph(self, graph: StateAttackGraph) -> np.ndarray:
        new_results = np.zeros((len(METHODS), len(METRICS)))
        for i_method, method in enumerate(METHODS):
            new_results[i_method] = MethodApplicator(graph, method, None, None,
                                                     METRICS).apply_method()
        return new_results

    def plot(self):
        results = self._get_existing_results()
        if results is None:
            return

        filenames = [
            "comparison_{}".format(utils.sanitize(metric))
            for metric in METRICS
        ]
        Histogram(results, "Methods", list(METHODS.keys()), METRICS,
                  filenames).create()


class TimeComparator(ClusteringFigureCreator):
    def __init__(self, n_graphs: int = None, continuous_plotting: bool = True):
        data_file_name = "time_comparison"
        super().__init__(data_file_name, n_graphs, continuous_plotting)

    def _apply_for_graph(self, graph: StateAttackGraph) -> np.ndarray:
        new_results = np.zeros(len(METHODS))
        for i_method, method in enumerate(METHODS):
            start = time()
            MethodApplicator(graph, method, None, None, None,
                             use_gpu=False).apply_method()
            new_results[i_method] = time() - start
        return new_results

    def plot(self):
        results = self._get_existing_results()
        if results is None:
            return

        Histogram(results, "Methods", list(METHODS.keys()),
                  ["Mean execution time (s)"], ["time_comparison"]).create()


def run_embedding_methods_optimization(n_graphs: int = None,
                                       continuous_plotting: bool = True):
    for method, parameters in METHODS.items():
        if parameters is None:
            continue

        for parameter in parameters:
            print("Optimizing parameter {} of method {}".format(
                parameter, method))
            mo = MethodOptimizer(method, parameter, n_graphs,
                                 continuous_plotting)
            mo.apply()
            mo.plot()


def run_method_comparison(n_graphs: int = None,
                          continuous_plotting: bool = True):
    mc = MethodComparator(n_graphs, continuous_plotting)
    mc.apply()
    mc.plot()


def run_method_time_comparison(n_graphs: int = None,
                               continuous_plotting: bool = True):
    tc = TimeComparator(n_graphs, continuous_plotting)
    tc.apply()
    tc.plot()

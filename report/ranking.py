import math
import multiprocessing
import numpy as np
import time
import utils
from attack_graph import DependencyAttackGraph, StateAttackGraph
from matplotlib.pyplot import subplots
from pathlib import Path
from ranking.abraham import ProbabilisticPath
from ranking.homer import RiskQuantifier
from ranking.mehta import PageRankMethod, KuehlmannMethod
from ranking.random import RandomRankingMethod
from ranking.ranking import RankingMethod
from ranking.sheyner import ValueIteration
from report.dataset import Dataset, HomerDataset
from report.report import Histogram, PATH_FIGURES
from typing import Dict, List, Set, Tuple

# CONSTANTS
PATH_DATA_FILE = Path("report/data/ranking.npy")
PATH_DATA_FILE_TOP_EXPLOITS = Path("report/data/ranking_top_exploits.npy")
PATH_DATA_FILE_TIME = Path("report/data/ranking_time.npy")

METHODS: List[str] = ["PR", "KUE", "VI(S)", "VI(D)", "HOM", "PP", "RAN"]


class RankingMethodsComparator:
    def __init__(self, n_graphs: int = None, continuous_plotting: bool = True):
        self.n_graphs = n_graphs
        self.continuous_plotting = continuous_plotting

    def create(self):
        # Create the necessary folders
        utils.create_parent_folders(PATH_DATA_FILE)
        utils.create_folders(PATH_FIGURES)

        self._compute_max_n_graphs()

        # Load the existing results and the new graphs
        existing_results = RankingMethodsComparator._load_existing_results()
        cur_results, cur_top, cur_times = existing_results
        new_graphs = self._load_next_graphs(cur_results)
        if new_graphs is None:
            return

        i_graph, state_graph, dependency_graph = new_graphs
        print("Graph {}/{}: {} nodes, {} exploits".format(
            i_graph + 1, Dataset.n_graphs, state_graph.number_of_nodes(),
            len(state_graph.exploits)))

        # Apply the methods on the new graphs
        exploits_rankings, times = RankingMethodsComparator._apply_methods(
            state_graph, dependency_graph)

        # Compute the ppce matrix
        ppce_matrix = RankingMethodsComparator._compute_ppce_matrix(
            exploits_rankings)

        # Get the top exploits
        top_exploits = RankingMethodsComparator._count_common_top_exploits(
            exploits_rankings)

        # Save the new results
        ppce_matrix = np.expand_dims(ppce_matrix, axis=0)
        top_exploits = np.expand_dims(top_exploits, axis=0)
        times = np.expand_dims(times, axis=0)
        if cur_results is not None:
            ppce_matrix = np.concatenate((cur_results, ppce_matrix))
            top_exploits = np.concatenate((cur_top, top_exploits))
            times = np.concatenate((cur_times, times))
        np.save(PATH_DATA_FILE, ppce_matrix)
        np.save(PATH_DATA_FILE_TOP_EXPLOITS, top_exploits)
        np.save(PATH_DATA_FILE_TIME, times)

        # Draw the matrix if asked by the user
        if self.continuous_plotting:
            RankingMethodsComparator.draw_ppce_matrix()
            RankingMethodsComparator.draw_top_exploits_matrix()
            RankingMethodsComparator.draw_time_histogram()

        self.create()

    @staticmethod
    def draw_matrix(matrix: np.ndarray, filename: str):
        fig, ax = subplots()
        plot = ax.matshow(matrix)
        fig.colorbar(plot)

        for (i, j), z in np.ndenumerate(matrix):
            ax.text(j,
                    i,
                    "{:0.2f}".format(z),
                    ha='center',
                    va='center',
                    color="white")

        ax.set_xticks(np.arange(len(METHODS)))
        ax.set_yticks(np.arange(len(METHODS)))
        ax.set_xticklabels(METHODS)
        ax.set_yticklabels(METHODS)

        path = Path(PATH_FIGURES, "{}.png".format(filename))
        fig.tight_layout()
        fig.savefig(path)
        fig.clf()

    @staticmethod
    def draw_ppce_matrix():
        results = RankingMethodsComparator._load_existing_results()[0]

        # Create one figure per set of graphs
        for i in range(len(Dataset.set_sizes)):
            start = sum(Dataset.set_sizes[:i])
            end = sum(Dataset.set_sizes[:i + 1])

            set_results = results[start:end]
            if set_results.shape[0] == 0:
                continue

            set_results = np.mean(set_results, axis=0)
            filename = "ranking_{}".format(i)
            RankingMethodsComparator.draw_matrix(set_results, filename)

    @staticmethod
    def draw_top_exploits_matrix():
        results = RankingMethodsComparator._load_existing_results()[1]

        # Create one figure per set of graphs
        for i in range(len(Dataset.set_sizes)):
            start = sum(Dataset.set_sizes[:i])
            end = sum(Dataset.set_sizes[:i + 1])

            set_results = results[start:end]
            if set_results.shape[0] == 0:
                continue

            set_results = np.mean(set_results, axis=0)
            filename = "ranking_top_exploits_{}".format(i)
            RankingMethodsComparator.draw_matrix(set_results, filename)

    @staticmethod
    def draw_time_histogram():
        times = RankingMethodsComparator._load_existing_results()[2]

        Histogram(times, "Methods", METHODS, ["Time execution (s)"],
                  ["ranking_time"]).create()

    def _compute_max_n_graphs(self):
        if self.n_graphs is None:
            self.max_n_graphs = Dataset.n_graphs
        else:
            self.max_n_graphs = min(self.n_graphs, Dataset.n_graphs)

    @staticmethod
    def _load_existing_results() -> Tuple[np.ndarray, np.ndarray]:
        if PATH_DATA_FILE.exists():
            return np.load(PATH_DATA_FILE), np.load(
                PATH_DATA_FILE_TOP_EXPLOITS), np.load(PATH_DATA_FILE_TIME)
        else:
            return None, None, None

    def _load_next_graphs(
        self, existing_results: np.ndarray
    ) -> Tuple[int, StateAttackGraph, DependencyAttackGraph]:
        if existing_results is None:
            i_graph = 0
        else:
            i_graph = existing_results.shape[0]

        if i_graph == self.max_n_graphs:
            return None

        state_graph = Dataset.load_state_graph(i_graph)
        dependency_graph = Dataset.load_dependency_graph(i_graph)

        return i_graph, state_graph, dependency_graph

    @staticmethod
    def _apply_methods(
        state_graph: StateAttackGraph, dependency_graph: DependencyAttackGraph
    ) -> Tuple[List[Dict[int, int]], List[float]]:
        instances: List[RankingMethod] = [
            PageRankMethod(state_graph),
            KuehlmannMethod(state_graph),
            ValueIteration(state_graph),
            ValueIteration(dependency_graph),
            RiskQuantifier(dependency_graph),
            ProbabilisticPath(state_graph),
            RandomRankingMethod(state_graph)
        ]

        exploit_rankings = []
        times = []
        for i, instance in enumerate(instances):
            print("Applying {}".format(METHODS[i]))
            start = time.time()
            exploit_rankings.append(instance.rank_exploits())
            times.append((time.time() - start) / len(state_graph.exploits))

        return exploit_rankings, times

    @staticmethod
    def _compute_ppce_matrix(
            exploit_rankings: List[Dict[int, int]]) -> np.ndarray:
        ppce_matrix = np.zeros((len(exploit_rankings), len(exploit_rankings)))
        for i, ranking_a in enumerate(exploit_rankings):
            for j, ranking_b in enumerate(exploit_rankings):
                if i >= j:
                    continue

                ppce = RankingMethodsComparator._compute_ppce(
                    ranking_a, ranking_b)
                ppce_matrix[i, j] = ppce
                ppce_matrix[j, i] = ppce
        return ppce_matrix

    @staticmethod
    def _compute_ppce(ranking_a: Dict[int, int],
                      ranking_b: Dict[int, int]) -> float:
        ppce = 0
        exploits = list(ranking_a)

        for i in range(len(exploits)):
            for j in range(len(exploits)):
                if i >= j:
                    continue

                exploit_0 = exploits[i]
                exploit_1 = exploits[j]

                value_a_0 = ranking_a[exploit_0]
                value_a_1 = ranking_a[exploit_1]
                value_b_0 = ranking_b[exploit_0]
                value_b_1 = ranking_b[exploit_1]

                # Check that the exploit 0 and 1 are sorted in the same order
                # in ranking_a and ranking_b
                difference_a = value_a_0 - value_a_1
                difference_b = value_b_0 - value_b_1
                if difference_a * difference_b < 0:
                    ppce += 1

        n_exploits = len(exploits)
        ppce /= (n_exploits * (n_exploits - 1)) / 2
        return ppce

    @staticmethod
    def _count_common_top_exploits(exploit_rankings: List[Dict[int, int]],
                                   top: float = 1 / 3) -> np.ndarray:
        list_top_exploits: List[Set[int]] = []
        n_top_exploits = math.ceil(len(exploit_rankings[0]) * top)

        # Fill the list of top exploits
        for rankings in exploit_rankings:
            top_exploits: Set[int] = set()
            for id_exploit, ranking in rankings.items():
                if ranking < n_top_exploits:
                    top_exploits.add(id_exploit)
            list_top_exploits.append(top_exploits)

        # Create a matrix comparing the sets of top exploits
        common_top_exploits = np.identity((len(METHODS)))
        for i in range(len(METHODS)):
            for j in range(len(METHODS)):
                if i >= j:
                    continue

                common_exploits = list_top_exploits[i] & list_top_exploits[j]
                common_top_exploits[i,
                                    j] = len(common_exploits) / n_top_exploits
                common_top_exploits[j,
                                    i] = len(common_exploits) / n_top_exploits

        return common_top_exploits


class HomerBranchNodes:
    def __init__(self, n_graphs: int = None):
        self.path_results = Path("report/data/homer.npy")
        if n_graphs is None:
            self.n_graphs = HomerDataset.n_graphs
        else:
            self.n_graphs = min(n_graphs, HomerDataset.n_graphs)

    def plot_execution_time_for_homer(self):
        utils.create_parent_folders(self.path_results)

        if not self.path_results.exists():
            np.save(self.path_results, np.zeros(self.n_graphs))

        for i_graph in range(self.n_graphs):
            print("Applying Homer on graph {}/{}".format(
                i_graph + 1, self.n_graphs))

            results = np.load(self.path_results)
            if results[i_graph] != 0:
                continue

            results[i_graph] = np.inf
            np.save(self.path_results, results)

            # Execute Homer
            process = multiprocessing.Process(target=self._run_homer,
                                              name="Homer",
                                              args=(i_graph, ))
            process.start()

            # If the execution isn't finished after one hour, stop the
            # execution
            for _ in range(360):
                time.sleep(10)
                if not process.is_alive():
                    break
            if process.is_alive():
                process.terminate()

            self._draw_scatter_plot_for_homer()

        self._draw_scatter_plot_for_homer()

    def _run_homer(self, i_graph: int):
        # Load the graph
        graph = HomerDataset.load(i_graph)

        # Apply Homer
        start = time.time()
        RiskQuantifier(graph).apply()
        total_time = time.time() - start

        # Save the time
        results = np.load(self.path_results)
        results[i_graph] = total_time
        np.save(self.path_results, results)

    def _draw_scatter_plot_for_homer(self):
        # Count the number of branch nodes
        n_branch_nodes = np.zeros(self.n_graphs)
        for i_graph in range(self.n_graphs):
            graph = HomerDataset.load(i_graph)
            n_branch_nodes[i_graph] = len(graph.get_branch_nodes())

        # Draw the graph
        results = np.load(self.path_results)
        results[results == np.inf] = 3600

        argsort = np.argsort(n_branch_nodes)
        n_branch_nodes = n_branch_nodes[argsort]
        results = results[argsort]

        fig, ax = subplots()
        ax.scatter(n_branch_nodes, results)
        ax.set_xlabel("Number of branch nodes")
        ax.set_ylabel("Logarithm of the execution time (log(s))")
        ax.set_yscale("log")

        path = Path(PATH_FIGURES, "homer.png")
        fig.tight_layout()
        fig.savefig(path)
        fig.clf()

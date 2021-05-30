from time import time
import numpy as np
import utils
from attack_graph import DependencyAttackGraph, StateAttackGraph
from matplotlib.pyplot import subplots
from pathlib import Path
from ranking.abraham import ExpectedPathLength
from ranking.homer import RiskQuantifier
from ranking.mehta import PageRankMethod, KuehlmannMethod
from ranking.ranking import RankingMethod
from ranking.sheyner import ValueIteration
from report.dataset import Dataset
from report.report import Histogram, PATH_FIGURES
from typing import Dict, List, Tuple

# CONSTANTS
PATH_DATA_FILE = Path("report/data/ranking.npy")
PATH_DATA_FILE_TIME = Path("report/data/ranking_time.npy")

METHODS: List[str] = ["PR", "KUE", "VI", "HOM", "EPL"]


class PpceMatrixCreator:
    def __init__(self, n_graphs: int = None, continuous_plotting: bool = True):
        self.n_graphs = n_graphs
        self.continuous_plotting = continuous_plotting

    def create(self):
        # Create the necessary folders
        utils.create_parent_folders(PATH_DATA_FILE)
        utils.create_folders(PATH_FIGURES)

        self._compute_max_n_graphs()

        # Load the existing results and the new graphs
        existing_results, existing_times = self._load_existing_results()
        new_graphs = self._load_next_graphs(existing_results)
        if new_graphs is None:
            return

        state_graph, dependency_graph = new_graphs
        print("{} nodes".format(state_graph.number_of_nodes()))

        # Apply the methods on the new graphs
        exploits_rankings, new_times = self._apply_methods(
            state_graph, dependency_graph)

        # Compute the ppce matrix
        ppce_matrix = self._compute_ppce_matrix(exploits_rankings)

        # Save the new results
        results = np.expand_dims(ppce_matrix, axis=0)
        times = np.expand_dims(new_times, axis=0)
        if existing_results is not None:
            results = np.concatenate((existing_results, results))
            times = np.concatenate((existing_times, times))
        np.save(PATH_DATA_FILE, results)
        np.save(PATH_DATA_FILE_TIME, times)

        # Draw the matrix if asked by the user
        if self.continuous_plotting:
            self.draw_ppce_matrix()
            self.draw_time_histogram()

        self.create()

    def draw_ppce_matrix(self):
        results = self._load_existing_results()[0]

        # Create one figure per set of graphs
        for i in range(len(Dataset.set_sizes)):
            start = sum(Dataset.set_sizes[:i])
            end = sum(Dataset.set_sizes[:i + 1])

            set_results = results[start:end]
            if set_results.shape[0] == 0:
                continue

            set_results = np.mean(set_results, axis=0)

            fig, ax = subplots()
            plot = ax.matshow(set_results)
            fig.colorbar(plot)

            ax.set_xticks(np.arange(len(METHODS)))
            ax.set_yticks(np.arange(len(METHODS)))
            ax.set_xticklabels(METHODS)
            ax.set_yticklabels(METHODS)

            path = Path(PATH_FIGURES, "ranking_{}.png".format(i))
            fig.tight_layout()
            fig.savefig(path)
            fig.clf()

    def draw_time_histogram(self):
        times = self._load_existing_results()[1]

        Histogram(times, "Methods", METHODS, ["Time execution (s)"],
                  ["ranking_time"]).create()

    def _compute_max_n_graphs(self):
        if self.n_graphs is None:
            self.max_n_graphs = Dataset.n_graphs
        else:
            self.max_n_graphs = min(self.n_graphs, Dataset.n_graphs)

    def _load_existing_results(self) -> Tuple[np.ndarray, np.ndarray]:
        if PATH_DATA_FILE.exists():
            return np.load(PATH_DATA_FILE), np.load(PATH_DATA_FILE_TIME)
        else:
            return None, None

    def _load_next_graphs(
        self, existing_results: np.ndarray
    ) -> Tuple[StateAttackGraph, DependencyAttackGraph]:
        if existing_results is None:
            i_graph = 0
        else:
            i_graph = existing_results.shape[0]

        if i_graph == self.max_n_graphs:
            return None

        state_graph = Dataset.load_state_graph(i_graph)
        dependency_graph = Dataset.load_dependency_graph(i_graph)

        return state_graph, dependency_graph

    def _apply_methods(
        self, state_graph: StateAttackGraph,
        dependency_graph: DependencyAttackGraph
    ) -> Tuple[List[Dict[int, float]], List[float]]:
        instances: List[RankingMethod] = [
            PageRankMethod(state_graph),
            KuehlmannMethod(state_graph),
            ValueIteration(state_graph),
            RiskQuantifier(dependency_graph),
            ExpectedPathLength(state_graph)
        ]

        exploit_rankings = []
        times = []
        for i, instance in enumerate(instances):
            print("Applying {}".format(METHODS[i]))
            start = time()
            exploit_rankings.append(instance.rank_exploits())
            times.append(time() - start)

        return exploit_rankings, times

    def _compute_ppce_matrix(
            self, exploit_rankings: List[Dict[int, float]]) -> np.ndarray:
        ppce_matrix = np.zeros((len(exploit_rankings), len(exploit_rankings)))
        for i, ranking_a in enumerate(exploit_rankings):
            for j, ranking_b in enumerate(exploit_rankings):
                if i >= j:
                    continue

                ppce = self._compute_ppce_between_two_orderings(
                    ranking_a, ranking_b)
                ppce_matrix[i, j] = ppce
                ppce_matrix[j, i] = ppce
        return ppce_matrix

    def _compute_ppce_between_two_orderings(
            self, ranking_a: Dict[int, float],
            ranking_b: Dict[int, float]) -> float:
        ppce = 0
        exploits = list(ranking_a)

        for exploit_0 in exploits:
            for exploit_1 in exploits:
                if exploit_0 == exploit_1:
                    continue

                value_a_0 = ranking_a[exploit_0]
                value_a_1 = ranking_a[exploit_1]
                value_b_0 = ranking_b[exploit_0]
                value_b_1 = ranking_b[exploit_1]

                # Check that the exploit 0 and 1 are sorted in the same order
                # in ranking_a and ranking_b
                difference_a = value_a_0 - value_a_1
                difference_b = value_b_0 - value_b_1
                if difference_a == 0 and difference_b != 0:
                    ppce += 1
                elif difference_a != 0 and difference_b == 0:
                    ppce += 1
                elif difference_a * difference_b < 0:
                    ppce += 1

        n_exploits = len(exploits)
        return ppce / (n_exploits * (n_exploits - 1))

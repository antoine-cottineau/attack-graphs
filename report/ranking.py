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
from typing import Dict, List, Tuple

# CONSTANTS
PATH_DATA_FILE = Path("report/data/ranking.npy")
PATH_FIGURES = Path("report/figures")

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
        existing_results = self._load_existing_results()
        new_graphs = self._load_next_graphs(existing_results)
        if new_graphs is None:
            return

        state_graph, dependency_graph = new_graphs
        print("{} nodes".format(state_graph.number_of_nodes()))

        # Apply the methods on the new graphs
        exploits_rankings = self._apply_methods(state_graph, dependency_graph)

        # Compute the ppce matrix
        ppce_matrix = self._compute_ppce_matrix(exploits_rankings)

        # Save the new results
        results = np.expand_dims(ppce_matrix, axis=0)
        if existing_results is not None:
            results = np.concatenate((existing_results, results))
        np.save(PATH_DATA_FILE, results)

        # Draw the matrix if asked by the user
        if self.continuous_plotting:
            self.draw_ppce_matrix()

        self.create()

    def draw_ppce_matrix(self):
        results = self._load_existing_results()

        # Fill the matrix with zeros if not all the graphs have been dealt with
        # if results.shape[0] < Dataset.n_graphs:
        #     results = np.concatenate(
        #         (results,
        #          np.zeros((Dataset.n_graphs - results.shape[0], len(METHODS),
        #                    len(METHODS)))))

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

    def _compute_max_n_graphs(self):
        if self.n_graphs is None:
            self.max_n_graphs = Dataset.n_graphs
        else:
            self.max_n_graphs = min(self.n_graphs, Dataset.n_graphs)

    def _load_existing_results(self) -> np.ndarray:
        if PATH_DATA_FILE.exists():
            return np.load(PATH_DATA_FILE)
        else:
            return None

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
            dependency_graph: DependencyAttackGraph) -> List[Dict[int, float]]:
        instances: List[RankingMethod] = [
            PageRankMethod(state_graph),
            KuehlmannMethod(state_graph),
            ValueIteration(state_graph),
            RiskQuantifier(dependency_graph),
            ExpectedPathLength(state_graph)
        ]

        exploit_rankings = [instance.rank_exploits() for instance in instances]

        return exploit_rankings

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

import numpy as np
import utils
from matplotlib.pyplot import subplots
from pathlib import Path
from report.dataset import Dataset
from typing import List

PATH_FIGURES = Path("report/figures")


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

        utils.create_folders(PATH_FIGURES)

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

                if bar_split.shape[0] == 0 or np.all(np.isnan(bar_split)):
                    y.append(0)
                else:
                    y.append(np.mean(bar_split[~np.isnan(bar_split)]))

            ax.bar(x, y, self.bar_width, label=self.bar_labels[i_bar])

        # Add other information to the figure
        ax.set_xlabel("Group")
        ax.set_xticks(np.arange(len(results)))
        ax.set_xticklabels(["Small", "Medium", "Large"])
        ax.set_ylabel(y_label)
        ax.legend(title=self.legend_title)

        # Save the figure
        path = Path(PATH_FIGURES, filename + ".png")
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

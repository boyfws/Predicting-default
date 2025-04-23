from optbinning import OptimalBinning
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


class BinningTest:
    @staticmethod
    def plot_graph_per_feature(
            binner: OptimalBinning,
            ax: plt.Axes
    ):
        table = binner.binning_table.build()
        mask = (table["Bin"] != "Missing") & (table["Bin"] != "Special") & (table["Bin"] != "")
        filtered_data = table[mask]

        bars = sns.barplot(x=filtered_data["Bin"],
                           y=filtered_data["Count (%)"],
                           color='magenta',
                           alpha=0.35,
                           ax=ax)

        ax.set_xticklabels(filtered_data["Bin"],
                           rotation=45,
                           ha='right',
                           fontsize=10)

        ax2 = ax.twinx()
        line = sns.lineplot(x=np.arange(len(filtered_data)),
                            y=filtered_data["Event rate"],
                            color='black',
                            marker='o',
                            markersize=8,
                            linewidth=2.5,
                            ax=ax2)

        ax.set_xlabel("Bin", fontsize=12, labelpad=15)
        ax.set_ylabel("Count (%)", fontsize=12, color='magenta')
        ax2.set_ylabel("Event Rate", fontsize=12, color='black')

        ax.tick_params(axis='y', colors='magenta')
        ax2.tick_params(axis='y', colors='black')

        ax.grid(True, linestyle='--', alpha=0.7)
        ax2.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

    @staticmethod
    def plot_graph(
            num_columns: list[str],
            binners: dict[str, OptimalBinning],
            figsize: tuple[int, int],
    ):
        grid_w = 4
        grid_h = len(num_columns) // grid_w + int((len(num_columns) % grid_w) != 0)
        fig, axs = plt.subplots(grid_h, grid_w, figsize=figsize)

        for i in range(grid_h):
            for j in range(grid_w):
                if 4 * i + j >= len(num_columns):
                    break
                axs[i, j].set_title(num_columns[4 * i + j])
                BinningTest.plot_graph_per_feature(binners[num_columns[4 * i + j]], axs[i, j])

        plt.tight_layout()
        plt.show()

    def __call__(self,
                 binners: dict[str, OptimalBinning],
                 plot: bool = False,
                 figsize: tuple[int, int] = (10, 10),
                 ) -> dict[
        str,
        dict[str, np.ndarray]
    ]:
        num_columns = []
        for el in binners:
            if binners[el].dtype == "numerical":
                num_columns.append(el)

        result_dict = {}

        for el in num_columns:
            table = binners[el].binning_table.build()
            mask = (table["Bin"] != "Missing") & (table["Bin"] != "Special") & (table["Bin"] != "")

            result_dict[el] = {
                "event_rate": table["Event rate"][mask].to_numpy(),
                "count": table["Count (%)"][mask].to_numpy(),
            }

        if plot:
            self.plot_graph(num_columns, binners, figsize)

        return result_dict




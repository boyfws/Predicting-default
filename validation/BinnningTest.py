import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from optbinning import OptimalBinning


class BinningTest:
    @staticmethod
    def plot_graph_per_feature(binner: OptimalBinning, ax: plt.Axes):
        table = binner.binning_table.build()
        mask = (
            (table["Bin"] != "Missing")
            & (table["Bin"] != "Special")
            & (table["Bin"] != "")
        )
        filtered_data = table[mask]

        sns.barplot(
            x=filtered_data["Bin"],
            y=filtered_data["Count (%)"],
            color="magenta",
            alpha=0.35,
            ax=ax,
        )

        ax.set_xticklabels(filtered_data["Bin"], rotation=45, size=7, ha="right")

        ax2 = ax.twinx()
        sns.lineplot(
            x=np.arange(len(filtered_data)),
            y=filtered_data["Event rate"],
            color="black",
            marker="o",
            markersize=7,
            linewidth=0.5,
            ax=ax2,
        )

        ax.set_xlabel("Bin")
        ax.set_ylabel("Count (%)", color="magenta")
        ax2.set_ylabel("Event Rate", color="black")

        ax.tick_params(axis="y", colors="magenta")
        ax2.tick_params(axis="y", colors="black")

        ax.grid(True, linestyle="--", alpha=0.7)
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
        grid_h = len(num_columns) // grid_w
        if grid_h * grid_w < len(num_columns):
            grid_h += 1

        fig, axs = plt.subplots(grid_h, grid_w, figsize=figsize)

        for i in range(grid_h):
            for j in range(grid_w):
                if grid_w * i + j >= len(num_columns):
                    break
                axs[i, j].set_title(num_columns[grid_w * i + j])
                BinningTest.plot_graph_per_feature(
                    binners[num_columns[grid_w * i + j]], axs[i, j]
                )

        plt.tight_layout()
        plt.show()

    def __call__(
        self,
        binners: dict[str, OptimalBinning],
        plot: bool = False,
        figsize: tuple[int, int] = (10, 10),
    ) -> dict[str, dict[str, np.ndarray]]:
        figsize = tuple(int(el * 1.5) for el in figsize)
        num_columns = []
        for key, value in binners.items():
            if value.dtype == "numerical":
                num_columns.append(key)

        result_dict = {}

        for el in num_columns:
            table = binners[el].binning_table.build()
            mask = (
                (table["Bin"] != "Missing")
                & (table["Bin"] != "Special")
                & (table["Bin"] != "")
            )

            result_dict[el] = {
                "event_rate": table["Event rate"][mask].to_numpy(),
                "count": table["Count (%)"][mask].to_numpy(),
            }

        if plot:
            self.plot_graph(num_columns, binners, figsize)

        return result_dict

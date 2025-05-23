import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .FeatureGini import FeatureGini


class FeatureGiniChangeTest(FeatureGini):
    @staticmethod
    def plot_ginis_change(
        result_train: dict[str, float],
        result_test: dict[str, float],
        figsize: tuple[int, int],
    ) -> None:

        keys = list(result_train.keys())
        values1 = list(result_train.values())
        values2 = list(result_test.values())

        plt.figure(figsize=figsize)
        bar_width = 0.35
        x = np.arange(len(keys))

        plt.bar(
            x - bar_width / 2,
            values1,
            width=bar_width,
            label="Train dataset",
            color="skyblue",
        )
        plt.bar(
            x + bar_width / 2,
            values2,
            width=bar_width,
            label="Test dataset",
            color="orange",
        )

        plt.xlabel("Features")
        plt.ylabel("Gini coefficient")
        # plt.title(title)
        plt.xticks(x, keys, rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def __call__(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        plot: bool = False,
        figsize: tuple[int, int] = (10, 10),
        n_jobs: int = -1,
    ):
        columns = X_train.columns

        #        result_train = {}
        #        result_test = {}
        #        for el in columns:
        #            result_train[el] = self.compute_gini_per_feature(
        #                X_train[el].to_numpy(),
        #                y_train,
        #            )

        #            result_test[el] = self.compute_gini_per_feature(
        #                X_test[el].to_numpy(),
        #                y_test,
        #            )

        train_results = Parallel(n_jobs=n_jobs)(
            delayed(self.compute_gini_per_feature)(
                col, X_train[col].to_numpy(), y_train
            )
            for col in columns
        )

        test_results = Parallel(n_jobs=n_jobs)(
            delayed(self.compute_gini_per_feature)(col, X_test[col].to_numpy(), y_test)
            for col in columns
        )

        result_train = {col: gini for col, gini in train_results}
        result_test = {col: gini for col, gini in test_results}

        if plot:
            self.plot_ginis_change(result_train, result_test, figsize)

        return result_train, result_test

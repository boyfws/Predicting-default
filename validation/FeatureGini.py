import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class FeatureGini:
    @staticmethod
    def compute_gini_per_feature(
            X_feat: np.ndarray,
            y_true: np.ndarray
    ) -> float:
        sorted_idx = np.argsort(X_feat)
        sorted_y_true = y_true[sorted_idx]

        y = np.cumsum(sorted_y_true) / np.sum(y_true)
        x = np.arange(1, len(y) + 1) / len(y)

        return np.abs(2 * np.trapz(y, x) - 1)

    @staticmethod
    def plot_ginis(
            ginis: dict[str, float],
            figsize: tuple[int, int]
        ) -> None:
        names = np.array(list(ginis.keys()))
        coefs = np.array(list(ginis.values()))

        sorted_idx = np.argsort(coefs)[::-1]

        sorted_names = names[sorted_idx]
        sorted_coefs = coefs[sorted_idx]

        plt.figure(figsize=figsize)
        sns.barplot(orient="h",
                    x=sorted_coefs,
                    y=sorted_names
                    )

    def __call__(self,
                 X: pd.DataFrame,
                 y_true: np.ndarray,
                 plot: bool = False,
                 figsize: tuple[int, int] = (10, 10),
                 ):
        columns = X.columns

        result = {}
        for el in columns:
            result[el] = self.compute_gini_per_feature(
                X[el].to_numpy(),
                y_true
            )

        if plot:
            self.plot_ginis(result, figsize)

        return result

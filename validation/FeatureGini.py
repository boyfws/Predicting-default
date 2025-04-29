import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import Parallel, delayed


class FeatureGini:
    @staticmethod
    def compute_gini_per_feature(
            col: str,
            X_feat: np.ndarray,
            y_true: np.ndarray
    ) -> tuple[str, float]:
        sorted_idx = np.argsort(X_feat)
        sorted_y_true = y_true[sorted_idx]

        y = np.cumsum(sorted_y_true) / np.sum(y_true)
        x = np.arange(1, len(y) + 1) / len(y)

        return col, np.abs(2 * np.trapz(y, x) - 1)

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
        plt.show()

    def __call__(self,
                 X: pd.DataFrame,
                 y_true: np.ndarray,
                 plot: bool = False,
                 figsize: tuple[int, int] = (10, 10),
                 n_jobs: int = -1
                 ):
        columns = X.columns

#        result = {}
#        for el in columns:
#            result[el] = self.compute_gini_per_feature(
#                X[el].to_numpy(),
#                y_true
#            )

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.compute_gini_per_feature)(col, X[col].to_numpy(), y_true)
            for col in columns
        )

        result = {col: gini for col, gini in results}

        if plot:
            self.plot_ginis(result, figsize)

        return result

from sklearn.base import TransformerMixin
from optbinning import OptimalBinning

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


class Binner(TransformerMixin):
    def __init__(
            self,
            params: dict[str, dict]
    ):
        self.params = params

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series
    ):

        X = X.copy()

        self.optb_ = {
            el: OptimalBinning(
                **self.params.get(el, {}),
                solver="cp"
            )
            for el in X.columns
        }

        self.iv_ = {}

        for el in X.columns:
            self.optb_[el].fit(X[el], y)

            self.optb_[el].binning_table.build()

            self.iv_[el] = round(self.optb_[el].binning_table.iv, 4)

    def transform(
            self,
            X: pd.DataFrame
    ):
        X = X.copy()
        for el in self.optb_:
            X[el] = self.optb_[el].transform(X[el])

        return X.astype(np.float32)

    def fit_transform(
            self,
            X: pd.DataFrame,
            y: pd.Series
    ):
        self.fit(X, y)
        return self.transform(X)

    def plot_iv(self):
        plt.figure(figsize=(10, 10))
        x_y = sorted(self.iv_.items(), key=lambda x: -x[1])
        x = [el[0] for el in x_y]
        y = [el[1] for el in x_y]

        sns.barplot(x=y, y=x, orient="h")
        plt.show()
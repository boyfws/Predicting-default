from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from optbinning import OptimalBinning
from sklearn.base import TransformerMixin


class Binner(TransformerMixin):
    def __init__(
        self,
        params: dict[str, dict],
        n_jobs: int = -1,
    ):
        self.params = params
        self.n_jobs = n_jobs

    @staticmethod
    def fit_feature(col: str, feature: pd.Series, y: np.ndarray, param):
        optb = OptimalBinning(**param, solver="cp")
        optb.fit(feature, y)
        optb.binning_table.build()
        iv = round(optb.binning_table.iv, 4)
        return col, optb, iv

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> None:
        X = X.copy()
        if not isinstance(y, np.ndarray):
            y_np = y.values
        else:
            y_np = y

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_feature)(el, X[el], y_np, self.params.get(el, {}))
            for el in X.columns
        )

        self.optb_ = {el: optb for el, optb, iv in results}
        self.iv_ = {el: iv for el, optb, iv in results}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for el in self.optb_:
            X[el] = self.optb_[el].transform(X[el])

        return X.astype(np.float32)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def plot_iv(self) -> None:
        plt.figure(figsize=(10, 10))
        x_y = sorted(self.iv_.items(), key=lambda x: -x[1])
        x = [el[0] for el in x_y]
        y = [el[1] for el in x_y]

        sns.barplot(x=y, y=x, orient="h")
        plt.show()

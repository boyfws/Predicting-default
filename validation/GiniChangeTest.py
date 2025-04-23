import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .GeneralGini import GeneralGini


class GiniChangeTest(GeneralGini):
    @staticmethod
    def plot_ginis(
            gini_train: float,
            gini_test: float,
            figsize: tuple[int, int],
    ):
        plt.figure(figsize=figsize)
        ax = sns.barplot(
            x=["Train", "Test"],
            y=[gini_train, gini_test],
        )
        ax.bar_label(ax.containers[0], fontsize=10);

        plt.show()

    def __call__(
            self,
            y_train: np.ndarray,
            y_test: np.ndarray,
            y_pred_train: np.ndarray,
            y_pred_test: np.ndarray,
            plot: bool = False,
            figsize: tuple[int, int] = (10, 10),
    ) -> tuple[float, float]:

        CAP_train = self.CAP_curve(y_train, y_pred_train)
        gini_train = self.gini_value(*CAP_train[::-1])

        CAP_test = self.CAP_curve(y_test, y_pred_test)
        gini_test = self.gini_value(*CAP_test[::-1])

        abs_change = (gini_train - gini_test) * 100
        rel_change = 100 * (gini_train - gini_test) / gini_train

        if plot:
            self.plot_ginis(
                gini_train,
                gini_test,
                figsize=figsize
            )

        return abs_change, rel_change

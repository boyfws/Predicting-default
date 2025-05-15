import matplotlib.pyplot as plt
import numpy as np


class GeneralGini:
    """ "
    Calculates the CAP curve
    (Cumulative Accuracy Profile)
    and calculates the gini coefficient from it
    """

    @staticmethod
    def CAP_curve(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        sorted_idx = np.argsort(y_pred)[::-1]
        y_true = y_true[sorted_idx]

        y = np.cumsum(y_true) / sum(y_true)
        x = np.arange(1, len(y) + 1) / len(y)

        return x, y

    @staticmethod
    def gini_value(y: np.ndarray, x: np.ndarray) -> float:
        return 2 * np.trapz(y, x) - 1

    @staticmethod
    def plot_CAP(
        y: np.ndarray,
        x: np.ndarray,
        figsize: tuple[int, int],
    ) -> None:
        plt.figure(figsize=figsize)
        plt.title("Cumulative Accuracy Profile for model")
        plt.plot(x, y, label="CAP curve")
        plt.plot(x, x, label="Random model")
        plt.fill_between(x, y, x, color="gray", alpha=0.3)
        plt.legend()
        plt.show()

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        plot: bool = False,
        figsize: tuple[int, int] = (10, 10),
    ) -> float:
        x, y = self.CAP_curve(y_true, y_pred)
        coef = self.gini_value(y, x)

        if plot:
            self.plot_CAP(y, x, figsize=figsize)

        return coef

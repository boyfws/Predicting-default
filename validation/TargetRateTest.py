import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class TargetRateTest:
    @staticmethod
    def compute_metrics(
        predicted_probs: np.ndarray, actual_events: np.ndarray
    ) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
        mean_prob = np.mean(predicted_probs)
        actual_rate = np.mean(actual_events)

        n = len(predicted_probs)
        std_err = np.std(predicted_probs, ddof=1) / np.sqrt(n)

        ci_95 = stats.norm.interval(0.95, loc=mean_prob, scale=std_err)
        ci_99 = stats.norm.interval(0.99, loc=mean_prob, scale=std_err)

        return mean_prob, actual_rate, ci_95, ci_99

    @staticmethod
    def plot_validation(
        mean_prob: float,
        actual_rate: float,
        ci_95: tuple[float, float],
        ci_99: tuple[float, float],
        figsize: tuple = (8, 6),
    ) -> None:
        plt.figure(figsize=figsize)

        mean_prob_pct = mean_prob * 100
        actual_rate_pct = actual_rate * 100
        ci_95_pct = (ci_95[0] * 100, ci_95[1] * 100)
        ci_99_pct = (ci_99[0] * 100, ci_99[1] * 100)

        ext_params = dict(capsize=10, capthick=2, elinewidth=1)

        plt.errorbar(
            [1],
            [mean_prob_pct],
            yerr=[[mean_prob_pct - ci_95_pct[0]], [ci_95_pct[1] - mean_prob_pct]],
            fmt="o",
            label="0.95 confidence interval",
            color="green",
            **ext_params,
        )
        plt.errorbar(
            [1],
            [mean_prob_pct],
            yerr=[[mean_prob_pct - ci_99_pct[0]], [ci_99_pct[1] - mean_prob_pct]],
            fmt="o",
            label="Predicted PD + 0.99 confidence interval ",
            color="blue",
            **ext_params,
        )
        plt.plot([1], [actual_rate_pct], "o", color="red", label="Observed target rate")
        plt.xticks([])

        plt.legend()
        plt.tight_layout()
        plt.show()

    def __call__(
        self,
        predicted_probs: np.ndarray,
        actual_events: np.ndarray,
        plot: bool = True,
        figsize: tuple = (10, 6),
    ) -> dict:
        mean_prob, actual_rate, ci_95, ci_99 = self.compute_metrics(
            predicted_probs, actual_events
        )

        results = {
            "mean_predicted_probability": mean_prob,
            "actual_event_rate": actual_rate,
            "95_ci_lower": ci_95[0],
            "95_ci_upper": ci_95[1],
            "99_ci_lower": ci_99[0],
            "99_ci_upper": ci_99[1],
        }

        if plot:
            self.plot_validation(
                mean_prob=mean_prob,
                actual_rate=actual_rate,
                ci_95=ci_95,
                ci_99=ci_99,
                figsize=figsize,
            )

        return results

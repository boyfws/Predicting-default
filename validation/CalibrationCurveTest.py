from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


class CalibrationCurveTest:
    def __call__(
        self,
        predicted_probs: np.ndarray,
        actual_events: np.ndarray,
        n_buckets: int = 5,
        strategy: str = "quantile",
        plot: bool = True,
        figsize: Tuple[int, int] = (12, 6),
    ) -> pd.DataFrame:
        df = self._create_buckets(predicted_probs, actual_events, n_buckets, strategy)

        results = self._compute_metrics(df)

        if plot:
            self._plot_validation(results, figsize)

        return results

    @staticmethod
    def _create_buckets(
        probs: np.ndarray, events: np.ndarray, n_buckets: int, strategy: str
    ) -> pd.DataFrame:
        if strategy == "quantile":
            df = pd.DataFrame({"prob": probs, "event": events})
            df["bucket"] = pd.qcut(probs, q=n_buckets, duplicates="drop")
        elif strategy == "uniform":
            df = pd.DataFrame({"prob": probs, "event": events})
            df["bucket"] = pd.cut(probs, bins=n_buckets)
        else:
            raise ValueError("Strategy must be 'quantile' or 'uniform'")

        return df

    @staticmethod
    def _compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
        results = (
            df.groupby("bucket", observed=True)
            .agg(
                mean_prob=("prob", "mean"),
                actual_rate=("event", "mean"),
                n_obs=("event", "count"),
            )
            .reset_index()
        )

        results["ci_95_lower"], results["ci_95_upper"] = zip(
            *results.apply(CalibrationCurveTest._calc_ci, axis=1, ci_level=0.95)
        )
        results["ci_99_lower"], results["ci_99_upper"] = zip(
            *results.apply(CalibrationCurveTest._calc_ci, axis=1, ci_level=0.99)
        )

        total = results["n_obs"].sum()
        results["bucket_share"] = results["n_obs"] / total

        return results

    @staticmethod
    def _calc_ci(row: pd.Series, ci_level: float) -> Tuple[float, float]:
        n = row["n_obs"]
        p = row["mean_prob"]
        std_err = np.sqrt(p * (1 - p) / n)
        return stats.norm.interval(ci_level, loc=p, scale=std_err)

    @staticmethod
    def _plot_validation(
        results: pd.DataFrame,
        figsize: Tuple[int, int],
    ) -> None:
        params = dict(
            # join=False,
            scale=0.6,
            linestyles="--",  # "-", "--", "-.", ":"
            # linewidth=1.5
        )
        fig, ax = plt.subplots(figsize=figsize)
        sns.pointplot(
            x=results["bucket"].astype(str),
            y=results["actual_rate"],
            label="Actual rate",
            color="Blue",
            ax=ax,
            **params,
        )
        sns.pointplot(
            x=results["bucket"].astype(str),
            y=results["mean_prob"],
            label="Mean prob.",
            color="Red",
            ax=ax,
            **params,
        )

        plt.xticks(rotation=45)
        ax2 = ax.twinx()
        sns.barplot(
            data=results,
            x="bucket",
            y="bucket_share",
            alpha=0.2,
            color="gray",
            ax=ax2,
            label="Bucket Share",
        )

        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            handles1 + handles2,
            labels1 + labels2,
            # loc='upper right'
        )
        plt.show()

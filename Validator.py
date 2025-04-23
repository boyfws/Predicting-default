from validation import *
from binning import Binner
from interfaces import Model

import pandas as pd
import numpy as np

from rich.console import Console
from rich.panel import Panel


class Validator:
    console: Console

    def __init__(
            self,
            plot_graphs: bool = True,
            external_tests: bool = True,
            figsize: tuple[int, int] = (10, 10),
    ) -> None:
        self.general_gini = GeneralGini()
        self.feature_gini = FeatureGini()
        self.binner_test = BinningTest()
        self.target_rate_test = TargetRateTest()

        self.plot_graphs = plot_graphs
        self.external_tests = external_tests

        self.figsize = figsize

        self.console = Console()

    def _validate_gini(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray
                       ) -> int:
        gen_gini = self.general_gini(
            y_true,
            y_pred,
            plot=self.plot_graphs,
            figsize=self.figsize,
        )
        title = "General Gini test"
        if gen_gini <= 0.25:
            score = -1
            self.console.print(Panel(f"[red]❌ Test failed with gini {gen_gini:.2f}[/red]", title=title))

        elif gen_gini <= 0.35:
            score = 0
            self.console.print(Panel(f"[yellow]⚠️ Intermediate result with gini {gen_gini:.2f}[/yellow]", title=title))
        else:
            score = 1
            self.console.print(Panel(f"[green]✅ Test passed with gini {gen_gini:.2f}[/green]", title=title))

        return score

    def _validate_features_gini(self,
                               X_orig: pd.DataFrame,
                               y_true: np.ndarray
                               ) -> int:
        coefs = self.feature_gini(
            X_orig,
            y_true,
            plot=self.plot_graphs,
            figsize=self.figsize,
        )

        names = np.array(list(coefs.keys()))
        coefs = np.array(list(coefs.values()))

        r, y, g = 0, 0, 0
        for el in coefs:
            if el < 0.05:
                r += 1
            elif el < 0.1:
                y += 1
            else:
                g += 1

        ryg_sum = r + y + g

        title = "Feature Gini test"
        if (r / ryg_sum) > 0.2:
            score = -1
            self.console.print(
                Panel(
                    f"[red]❌ Test failed with red, yellow, green shares {(r / ryg_sum):.2f}, {(y / ryg_sum):.2f}, {(g / ryg_sum):.2f}[/red]",
                    title=title)
            )
        elif (y / ryg_sum) > 0.1:
            score = 0
            self.console.print(
                Panel(
                    f"[yellow]⚠️ Intermediate result with red, yellow, green shares {(r / ryg_sum):.2f}, {(y / ryg_sum):.2f}, {(g / ryg_sum):.2f}[/yellow]",
                    title=title
                )
            )
        else:
            score = 1
            self.console.print(
                Panel(
                    f"[green]✅ Test passed with red, yellow, green shares {(r / ryg_sum):.2f}, {(y / ryg_sum):.2f}, {(g / ryg_sum):.2f}[/green]",
                    title=title
                )
            )

        return score

    def _validate_binning(
            self,
            binner: Binner
    ) -> int:
        results = self.binner_test(
            binner.optb_,
            plot=self.plot_graphs,
            figsize=self.figsize,
        )

        r, y, g = 0, 0, 0
        for key in results:
            ev_rate = results[key]["event_rate"]
            count = results[key]["count"]

            if ev_rate.shape[0] == 1:
                continue
            elif ev_rate.shape[0] == 2:
                g += 1
                continue

            ev_rate_diff = np.abs(np.diff(ev_rate))
            rel_ev_rate_diff = ev_rate_diff / (ev_rate[1:] + 1e-8)

            count_mask = count[1:] > 0.1
            diff_mask = rel_ev_rate_diff >= 0.1

            if np.any(count_mask * diff_mask):
                r += 1
            elif np.any(
                count_mask != diff_mask
            ):
                y += 1
            else:
                g += 1

        ryg_sum = r + y + g

        title = "Binning test"
        if (r / ryg_sum) > 0.2:
            score = -1
            self.console.print(
                Panel(
                    f"[red]❌ Test failed with red share {(r / ryg_sum):.2f}[/red]",
                    title=title
                )
            )

        elif (y / ryg_sum) > 0.1:
            score = 0
            self.console.print(
                Panel(
                    f"[yellow]⚠️ Intermediate result with red, yellow, green shares {(r / ryg_sum):.2f}, {(y / ryg_sum):.2f}, {(g / ryg_sum):.2f}[/yellow]",
                    title=title
                )
            )
        else:
            score = 1
            self.console.print(Panel(f"[green]✅ Test passed with red, yellow, green shares {(r / ryg_sum):.2f}, {(y / ryg_sum):.2f}, {(g / ryg_sum):.2f}[/green]", title=title))

        return score

    def _validate_target_rate(
            self,
            pred_p: np.ndarray,
            y_true: np.ndarray
    ):
        results = self.target_rate_test(
            pred_p,
            y_true,
            plot=self.plot_graphs,
            figsize=self.figsize
        )

        coef = np.abs(results["actual_event_rate"] - results["mean_predicted_probability"]) / results["actual_event_rate"]

        title = "Target Rate test"
        if coef > 0.2:
            score = -1
            self.console.print(
                Panel(
                    f"[red]❌ Test failed with share {(coef * 100):.2f}%[/red]",
                    title=title
                )
            )

        elif coef > 0.1:
            score = 0
            self.console.print(
                Panel(
                    f"[yellow]⚠️ Intermediate result with share {(coef * 100):.2f}%[/yellow]",
                    title=title
                )
            )
        else:
            score = 1
            self.console.print(Panel(f"[green]✅ Test passed with share {(coef * 100):.2f}%[/green]", title=title))

        return score


    def validate(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            binner: Binner,
            model: Model
    ):
        X_transformed: pd.DataFrame = binner.transform(X)

        pred_prob: np.ndarray = model.predict_proba(X_transformed)
        y_np = y.to_numpy()

        score1 = self._validate_gini(y_np, pred_prob)
        score2 = self._validate_features_gini(X.select_dtypes(include="number"), y_np)
        score3 = self._validate_binning(binner)
        score4 = self._validate_target_rate(pred_prob, y_np)

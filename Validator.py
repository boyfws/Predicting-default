from typing import Optional, Union

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from binning import Binner
from utils import Model
from validation import (
    BinningTest,
    CalibrationCurveTest,
    FeatureGini,
    FeatureGiniChangeTest,
    GeneralGini,
    GiniChangeTest,
    TargetRateTest,
)


class Validator:
    console: Console

    def __init__(
        self,
        plot_graphs: bool = True,
        figsize: tuple[int, int] = (10, 10),
        n_jobs: int = -1,
    ) -> None:
        self.general_gini = GeneralGini()
        self.feature_gini = FeatureGini()
        self.binner_test = BinningTest()
        self.target_rate_test = TargetRateTest()
        self.calibration_curve_test = CalibrationCurveTest()
        self.gini_change_test = GiniChangeTest()
        self.feature_gini_change_test = FeatureGiniChangeTest()

        self.plot_graphs = plot_graphs

        self.figsize = figsize
        self.n_jobs = n_jobs

        self.console = Console()

    def _validate_gini(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        gen_gini = self.general_gini(
            y_true,
            y_pred,
            plot=self.plot_graphs,
            figsize=self.figsize,
        )
        title = "General Gini test"
        if gen_gini <= 0.25:
            score = -1
            self.console.print(
                Panel(
                    f"[red]❌ Test failed with gini {gen_gini:.2f}[/red]", title=title
                )
            )

        elif gen_gini <= 0.35:
            score = 0
            self.console.print(
                Panel(
                    f"[yellow]⚠️ Intermediate result with gini {gen_gini:.2f}[/yellow]",
                    title=title,
                )
            )
        else:
            score = 1
            self.console.print(
                Panel(
                    f"[green]✅ Test passed with gini {gen_gini:.2f}[/green]",
                    title=title,
                )
            )

        return score

    def _validate_features_gini(self, X_orig: pd.DataFrame, y_true: np.ndarray) -> int:
        coefs = self.feature_gini(
            X_orig,
            y_true,
            plot=self.plot_graphs,
            figsize=self.figsize,
            n_jobs=self.n_jobs,
        )

        # names = np.array(list(coefs.keys()))
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
                    "[red]❌ Test failed with red, yellow, green shares "
                    + f"{(r / ryg_sum):.2f}, "
                    + f"{(y / ryg_sum):.2f}, {(g / ryg_sum):.2f}[/red]",
                    title=title,
                )
            )
        elif (y / ryg_sum) > 0.1:
            score = 0
            self.console.print(
                Panel(
                    "[yellow]⚠️ Intermediate result with red, yellow, green shares "
                    + f"{(r / ryg_sum):.2f}, "
                    + f"{(y / ryg_sum):.2f}, {(g / ryg_sum):.2f}[/yellow]",
                    title=title,
                )
            )
        else:
            score = 1
            self.console.print(
                Panel(
                    "[green]✅ Test passed with red, yellow, green shares "
                    + f"{(r / ryg_sum):.2f}, {(y / ryg_sum):.2f}, "
                    + f"{(g / ryg_sum):.2f}[/green]",
                    title=title,
                )
            )

        return score

    def _validate_binning(self, binner: Binner) -> int:
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
            elif np.any(count_mask != diff_mask):
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
                    title=title,
                )
            )

        elif (y / ryg_sum) > 0.1:
            score = 0
            self.console.print(
                Panel(
                    "[yellow]⚠️ Intermediate result "
                    + f"with red, yellow, green shares {(r / ryg_sum):.2f}, "
                    + f"{(y / ryg_sum):.2f}, {(g / ryg_sum):.2f}[/yellow]",
                    title=title,
                )
            )
        else:
            score = 1
            self.console.print(
                Panel(
                    "[green]✅ Test passed with red, "
                    + f"yellow, green shares {(r / ryg_sum):.2f}, "
                    + f"{(y / ryg_sum):.2f}, {(g / ryg_sum):.2f}[/green]",
                    title=title,
                )
            )

        return score

    def _validate_target_rate(self, pred_p: np.ndarray, y_true: np.ndarray) -> int:
        results = self.target_rate_test(
            pred_p, y_true, plot=self.plot_graphs, figsize=self.figsize
        )

        coef = (
            np.abs(results["actual_event_rate"] - results["mean_predicted_probability"])
            / results["actual_event_rate"]
        )

        title = "Target Rate test"
        if coef > 0.2:
            score = -1
            self.console.print(
                Panel(
                    f"[red]❌ Test failed with share {(coef * 100):.2f}%[/red]",
                    title=title,
                )
            )

        elif coef > 0.1:
            score = 0
            self.console.print(
                Panel(
                    "[yellow]⚠️ Intermediate result "
                    + f"with share {(coef * 100):.2f}%[/yellow]",
                    title=title,
                )
            )
        else:
            score = 1
            self.console.print(
                Panel(
                    f"[green]✅ Test passed with share {(coef * 100):.2f}%[/green]",
                    title=title,
                )
            )

        return score

    def _validate_curve_test(self, pred_p: np.ndarray, y_true: np.ndarray) -> int:

        result_df = self.calibration_curve_test(
            pred_p,
            y_true,
            n_buckets=10,
            plot=self.plot_graphs,
            figsize=self.figsize,
            strategy="quantile",
        )

        fact = result_df["actual_rate"].to_numpy()
        pred = result_df["mean_prob"].to_numpy()

        coef = np.abs(fact - pred) / fact
        share = np.mean(coef > 0.2)

        title = "Calibration Curve test"
        if share > 0.3:
            score = -1
            self.console.print(Panel("[red]❌ Test failed[/red]", title=title))

        elif share > 0.15:
            score = 0
            self.console.print(
                Panel("[yellow]⚠️ Intermediate result[/yellow]", title=title)
            )
        else:
            score = 1
            self.console.print(Panel("[green]✅ Test passed[/green]", title=title))

        return score

    def _validate_gini_change(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_pred_train: np.ndarray,
        y_pred_test: np.ndarray,
    ):
        abs_diff, rel_diff = self.gini_change_test(
            y_train,
            y_test,
            y_pred_train,
            y_pred_test,
            plot=self.plot_graphs,
            figsize=self.figsize,
        )

        title = "Gini Change test"
        if abs_diff > 5 and rel_diff > 20:
            score = -1
            self.console.print(
                Panel(
                    "[red]❌ Test failed with "
                    + "absolute and relative diffs: "
                    + f"{-abs_diff:.2f} p.p and {-rel_diff:.2f}%[/red]",
                    title=title,
                )
            )

        elif abs_diff > 3 and rel_diff > 15:
            score = 0
            self.console.print(
                Panel(
                    "[yellow]⚠️ Intermediate result "
                    + "with absolute and relative diffs: "
                    + f"{-abs_diff:.2f} p.p and {-rel_diff:.2f}%[/yellow]",
                    title=title,
                )
            )
        else:
            score = 1
            self.console.print(
                Panel(
                    "[green]✅ Test passed with absolute "
                    + "and relative diffs: "
                    + f"{-abs_diff:.2f} p.p and {-rel_diff:.2f}%[/green]",
                    title=title,
                )
            )

        return score

    def _validate_features_gini_change_test(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> int:
        result_train, result_test = self.feature_gini_change_test(
            X_train,
            y_train,
            X_test,
            y_test,
            plot=self.plot_graphs,
            figsize=self.figsize,
            n_jobs=self.n_jobs,
        )
        keys = list(result_train.keys())
        train_gini = np.array([result_train[el] for el in keys])
        test_gini = np.array([result_test[el] for el in keys])

        abs_diff = (train_gini - test_gini) * 100
        rel_diff = 100 * (train_gini - test_gini) / train_gini

        scores = np.empty(len(keys), dtype=int)

        mask1 = (abs_diff > 5) * (rel_diff > 20)
        mask2 = (abs_diff > 3) * (rel_diff > 15)

        scores[mask1] = -1
        scores[mask2] = 0
        scores[~(mask1 + mask2)] = 1

        r_share = (scores == -1).mean()
        y_share = (scores == 0).mean()
        g_share = 1 - r_share - y_share

        title = "Features Gini Change test"
        if r_share > 0.0:
            score = -1
            self.console.print(
                Panel(
                    "[red]❌ Test failed with "
                    + f"{r_share:.2f} , {y_share:.2f} , {g_share:.2f} "
                    + "shares for red, yellow, green scores[/red]",
                    title=title,
                )
            )

        elif y_share > 0.1:
            score = 0
            self.console.print(
                Panel(
                    "[yellow]⚠️ Intermediate result with "
                    + f"{r_share:.2f}, {y_share:.2f}, {g_share:.2f} "
                    + "shares for red, yellow, green scores[/yellow]",
                    title=title,
                )
            )
        else:
            score = 1
            self.console.print(
                Panel(
                    "[green]✅ Test passed with "
                    + f"{r_share:.2f} , {y_share:.2f} , {g_share:.2f} "
                    + "shares for red, yellow, green scores[/green]",
                    title=title,
                )
            )

        return score

    def _final_validation(self, scores: list[int]) -> None:
        title = "[bold]Final result[/bold]"
        if -1 in scores or scores.count(0) > 3:
            self.console.print(Panel("[red]❌ Validation failed[/red]", title=title))

        else:
            self.console.print(
                Panel("[green]✅ Validation passed[/green]", title=title)
            )

    def validate(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        model: Model,
        train_data: tuple[pd.DataFrame, Union[pd.Series, np.ndarray]],
        binner: Optional[Binner] = None,
    ) -> None:
        if binner is None:
            y_pred_test = model.predict_proba(X)
            y_pred_train = model.predict_proba(train_data[0])
        else:
            X_transformed_test = binner.transform(X)
            X_transformed_train = binner.transform(train_data[0])

            y_pred_test = model.predict_proba(X_transformed_test)
            y_pred_train = model.predict_proba(X_transformed_train)

        if y_pred_test.ndim == 2:
            y_pred_test = y_pred_test[:, 1]

        if y_pred_train.ndim == 2:
            y_pred_train = y_pred_train[:, 1]

        if isinstance(y, np.ndarray):
            y_test_np = y.copy()
        else:
            y_test_np = y.to_numpy()

        if isinstance(train_data[1], np.ndarray):
            y_train_np = train_data[1].copy()
        else:
            y_train_np = train_data[1].to_numpy()

        scores = []

        score1 = self._validate_gini(y_test_np, y_pred_test)
        scores.append(score1)
        score2 = self._validate_features_gini(
            X.select_dtypes(include="number"), y_test_np
        )
        scores.append(score2)

        if binner is not None:
            score3 = self._validate_binning(binner)
            scores.append(score3)

        score4 = self._validate_target_rate(y_pred_test, y_test_np)
        scores.append(score4)
        score5 = self._validate_curve_test(y_pred_test, y_test_np)
        scores.append(score5)
        score6 = self._validate_gini_change(
            y_train_np, y_test_np, y_pred_train, y_pred_test
        )
        scores.append(score6)
        score7 = self._validate_features_gini_change_test(
            train_data[0].select_dtypes(include="number"),
            y_train_np,
            X.select_dtypes(include="number"),
            y_test_np,
        )
        scores.append(score7)

        self._final_validation(scores)

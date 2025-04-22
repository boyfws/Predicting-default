from validation import GeneralGini
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
    ):
        self.general_gini = GeneralGini()

        self.plot_graphs = plot_graphs
        self.external_tests = external_tests

        self.figsize = figsize

        self.console = Console()

    def _validate_gini(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray
                       ):
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

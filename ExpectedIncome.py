import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


class IncomePredictor:
    def __init__(
        self,
        amount_col: str,
        term_month_col: str,
        recovery_rate: float,
        interest_rate: float,
    ) -> None:
        """
        Annuity payments

        :param amount_col:
        :param term_month_col:
        :param recovery_rate:
        :param interest_rate: Annual interest rate in
        fractional format, e.g., 13% → 0.13
        """
        assert 0 <= interest_rate <= 1
        assert 0 <= recovery_rate <= 1, "Recovery rate must be between 0 and 1"

        self.amount_col = amount_col
        self.term_month_col = term_month_col
        self.recovery_rate = recovery_rate
        self.interest_rate = interest_rate

    def _profit_calc(
        self,
        amnt: np.ndarray,
        term_months: np.ndarray,
    ):
        r = self.interest_rate / 12
        A = amnt * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)
        total_paid = A * term_months

        return total_paid - amnt

    def calculate_income(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        probs: np.ndarray,
        t: float,
    ):
        pred = probs >= t

        amnt = X[self.amount_col].to_numpy().astype(np.float32)
        term = X[self.term_month_col].to_numpy().astype(np.float32)

        profit = self._profit_calc(amnt, term)

        profit_giv = profit[~pred]
        def_giv = y[~pred].astype(bool)
        amnt_giv = amnt[~pred]

        no_default_profit = profit_giv[~def_giv].sum()
        default_loss = amnt_giv[def_giv].sum() * (1 - self.recovery_rate)

        return {
            "income": no_default_profit - default_loss,
            "funded_sum": amnt_giv.sum(),
        }

    def _find_optimal_t(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        probs: np.ndarray,
        n_jobs: int = -1,
    ) -> list[tuple[float, float, float]]:
        amnt = X[self.amount_col].to_numpy().astype(np.float32)
        term = X[self.term_month_col].to_numpy().astype(np.float32)

        profit = self._profit_calc(amnt, term)

        t_list = np.linspace(probs.min() - 1e-3, probs.max() + 1e-3, 200)

        def _calc_income(
            t: float,
            y: np.ndarray,
            probs: np.ndarray,
            profit: np.ndarray,
            amnt: np.ndarray,
        ) -> tuple[float, float, float]:
            pred = probs >= t

            profit_giv = profit[~pred]
            def_giv = y[~pred].astype(bool)
            amnt_giv = amnt[~pred]

            income = profit_giv[~def_giv].sum() - amnt_giv[def_giv].sum() * (
                1 - self.recovery_rate
            )
            return t, income, amnt_giv.sum()

        results = Parallel(n_jobs=n_jobs)(
            delayed(_calc_income)(t, y, probs, profit, amnt) for t in t_list
        )
        return results

    def find_optimal_t(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        probs: np.ndarray,
        n_jobs: int = -1,
        plot: bool = True,
        figsize: tuple[int, int] = (12, 6),
    ):
        results = self._find_optimal_t(X, y, probs, n_jobs=n_jobs)

        t = [el[0] for el in results]
        income = [el[1] for el in results]
        funded_sum = [el[2] for el in results]
        fund_yield = [
            inc * 100 / (fund_s + 1e-8) for inc, fund_s in zip(income, funded_sum)
        ]

        if plot:
            plt.figure(figsize=figsize)
            plt.title("Optimal Income based on threshold")
            plt.plot(t, income)
            plt.xlabel("Threshold")
            plt.ylabel("Income")
            plt.show()

            plt.figure(figsize=figsize)
            plt.title("Optimal percentage yield  based on threshold")
            plt.plot(t, fund_yield)
            plt.xlabel("Threshold")
            plt.ylabel("Yield %")
            plt.show()

        inc_arg_max = np.argmax(income)
        yield_arg_max = np.argmax(fund_yield)

        return {
            "Optimal Income": {
                "t": t[inc_arg_max],
                "income": income[inc_arg_max],
                "funded_sum": funded_sum[inc_arg_max],
                "yield": fund_yield[inc_arg_max],
            },
            "Optimal Yield": {
                "t": t[yield_arg_max],
                "income": income[yield_arg_max],
                "funded_sum": funded_sum[yield_arg_max],
                "yield": fund_yield[yield_arg_max],
            },
        }

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict

MISSING_VALUE = "__MISSING__"


class DataTransformer:
    def __init__(self):
        self._final_scaler = MinMaxScaler(feature_range=(-1, 1))

        self._columns_data = {}
        self._initial_columns_order = []
        self._fitted = False
        self._final_size = None

    @staticmethod
    def _check_df(df: pd.DataFrame) -> None:
        if df.empty:
            raise RuntimeError("DataFrame is empty")

        if df.select_dtypes(include="number").map(np.isinf).any(axis=None):
            raise RuntimeError("DataFrame is infinite")

    def _prepare_indices(self) -> None:
        self._softmax_idx = {
            tuple(self._columns_data[el]["one_hot"]["one_hot_idx"]): el for el in self._columns_data
            if len(self._columns_data[el]["one_hot"]["one_hot_idx"]) != 0
        }

        self._masked_idx = {
            (
                self._columns_data[el]["col_idx"],
                self._columns_data[el]["mask_idx"]
            ): el for el in self._columns_data
            if self._columns_data[el]["mask_idx"] is not None
        }

        self._ordinal_idx = {
            self._columns_data[el]["col_idx"]: el
            for el in self._columns_data
            if self._columns_data[el]["mask_idx"] is None
               and self._columns_data[el]["col_idx"] is not None
        }
        self._ordinal_idx_list = list(self._ordinal_idx.keys())

    def fit(self, df: pd.DataFrame) -> None:
        self._check_df(df)
        self._initial_columns_order = df.columns.tolist()

        new_df = OrderedDict()
        i = 0

        for el in df.columns:
            self._columns_data[el] = {
                "dtype": df[el].dtype,
                "apply_round": False,
                "col_idx": None,  # Columns id
                "mask_idx": None,  # Mask id
                "one_hot": {
                    "categories": [],  # List of categories
                    "one_hot_idx": [],  # Start and end idx
                }

            }

            if is_numeric_dtype(df[el]) or is_bool_dtype(df[el]):
                mask = df[el].isna()

                if mask.any():
                    new_df[el] = df[el].fillna(0, inplace=False)
                    new_df[f"{el}_mask"] = mask
                    self._columns_data[el]["col_idx"] = i
                    self._columns_data[el]["mask_idx"] = i + 1
                    i += 2
                else:
                    new_df[el] = df[el]
                    self._columns_data[el]["col_idx"] = i
                    i += 1

                if (df[el][~mask].round() == df[el][~mask]).fillna(False).all():
                    self._columns_data[el]["apply_round"] = True

            else:
                cat_df = pd.get_dummies(df[el], dummy_na=True)
                cat_df.columns = cat_df.columns.fillna(MISSING_VALUE)
                self._columns_data[el]["one_hot"]["categories"] = cat_df.columns.to_list()
                self._columns_data[el]["one_hot"]["one_hot_idx"] = [i, i + cat_df.shape[1] - 1]
                i += cat_df.shape[1]

                for col in cat_df.columns:
                    new_df[f"{el}_{col}"] = cat_df[col]

        self._final_scaler.fit(pd.DataFrame(new_df).to_numpy())

        self._final_size = len(new_df)
        self._prepare_indices()

        self._fitted = True

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError(
                "fit method must be called before transform"
            )

        self._check_df(df)

        new_df = OrderedDict()

        for el in self._initial_columns_order:
            if self._columns_data[el]["col_idx"] is not None:
                new_df[el] = df[el]

                if self._columns_data[el]["mask_idx"] is not None:
                    new_df[el] = df[el].fillna(0, inplace=False)
                    new_df[f"{el}_mask"] = df[el].isna()
                else:
                    new_df[el] = df[el]

            else:
                cat_df = pd.get_dummies(df[el], dummy_na=True)
                cat_df.columns = cat_df.columns.fillna(MISSING_VALUE)
                cat_df_cols_set = set(cat_df.columns)

                extra = cat_df_cols_set - set(self._columns_data[el]["one_hot"]["categories"])

                if extra:
                    raise ValueError(f"Unexpected categories: {extra}")

                for col in self._columns_data[el]["one_hot"]["categories"]:
                    if col not in cat_df_cols_set:
                        new_df[f"{el}_{col}"] = [False] * df.shape[0]
                    else:
                        new_df[f"{el}_{col}"] = cat_df[col]

        return torch.tensor(
            self._final_scaler.transform(
                pd.DataFrame(new_df).to_numpy()
            ),
            dtype=torch.float32,
        )

    def fit_transform(self, df: pd.DataFrame) -> torch.Tensor:
        self.fit(df)
        return self.transform(df)

    def calculate_mse_vae(
            self,
            reconstructed: torch.Tensor,
            true: torch.Tensor
    ) -> torch.Tensor:

        loss = torch.nn.functional.mse_loss(
            reconstructed[:, self._ordinal_idx_list],
            true[:, self._ordinal_idx_list],
            reduction="sum"
        )
        for el in self._softmax_idx:
            loss += torch.nn.functional.mse_loss(
                torch.nn.functional.softmax(reconstructed[:, el[0]: el[1] + 1], dim=1),
                true[:, el[0]: el[1] + 1],
                reduction="sum"
            )

        for el in self._masked_idx:
            loss += (
                    torch.nn.functional.mse_loss(
                        reconstructed[:, el[0]],
                        true[:, el[0]],
                        reduction="none"
                    ) * (
                        1 - true[:, el[1]]
                    )
            ).sum()

            loss += torch.nn.functional.mse_loss(
                reconstructed[:, el[1]],
                true[:, el[1]],
                reduction="sum"
            )

        return loss

    def inverse_transform(self, tensor: torch.Tensor) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(
                "fit method must be called before inverse_transform"
            )

        tensor_unscaled = self._final_scaler.inverse_transform(tensor)
        new_df = {value: tensor_unscaled[:, key] for key, value in self._ordinal_idx.items()}

        for key, value in self._softmax_idx.items():
            new_df[value] = [
                self._columns_data[value]["one_hot"]["categories"][i]
                if self._columns_data[value]["one_hot"]["categories"][i] != MISSING_VALUE
                else np.nan

                for i in np.argmax(tensor_unscaled[:, key[0]: key[1] + 1], axis=1)
            ]

        for key, value in self._masked_idx.items():
            ser_tens = pd.Series(tensor_unscaled[:, key[0]])
            mask = tensor_unscaled[:, key[1]] >= 0.5
            ser_tens[mask] = np.nan
            new_df[value] = ser_tens

        df = pd.DataFrame(new_df)
        for el in self._columns_data:

            if self._columns_data[el]["apply_round"]:
                df[el] = df[el].round()

            df[el] = df[el].astype(self._columns_data[el]["dtype"])

        return df.reindex(columns=self._initial_columns_order)

    def get_params(self) -> dict:
        if not self._fitted:
            raise RuntimeError(
                "fit method must be called before get_params"
            )

        data = {
            "columns_data": dict(self._columns_data),
            "initial_columns_order": list(self._initial_columns_order),
            "final_size": self._final_size,
            "final_scaler": {
                "scale_": self._final_scaler.scale_.tolist(),
                "min_": self._final_scaler.min_.tolist(),
                "data_min_": self._final_scaler.data_min_.tolist(),
                "data_max_": self._final_scaler.data_max_.tolist(),
                "feature_range": self._final_scaler.feature_range,
            },
        }
        return data

    @classmethod
    def save_params(cls, params: dict) -> "DataTransformer":
        self = cls.__new__(cls)

        self._columns_data = params["columns_data"]
        self._initial_columns_order = params["initial_columns_order"]
        self._final_size = params["final_size"]

        fs = params["final_scaler"]
        scaler = MinMaxScaler(feature_range=tuple(fs["feature_range"]))
        scaler.scale_ = np.array(fs["scale_"])
        scaler.min_ = np.array(fs["min_"])
        scaler.data_min_ = np.array(fs["data_min_"])
        scaler.data_max_ = np.array(fs["data_max_"])
        scaler.data_range_ = scaler.data_max_ - scaler.data_min_
        self._final_scaler = scaler

        self._fitted = True
        self._prepare_indices()

        return self

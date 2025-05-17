import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from collections import OrderedDict


class DataTransformer:
    def __init__(self):
        self.final_scaler = MinMaxScaler(feature_range=(-1, 1))

        self.columns_data = {}
        self.initial_columns_order = []
        self.fitted = False

    @staticmethod
    def _check_df(df: pd.DataFrame) -> None:
        if df.empty:
            raise RuntimeError("DataFrame is empty")

        if df.select_dtypes(include="number").map(np.isinf).any(axis=None):
            raise RuntimeError("DataFrame is infinite")

    def _prepare_indices(self) -> None:
        self._softmax_idx = {
            tuple(self.columns_data[el]["one_hot"]["one_hot_idx"]): el for el in self.columns_data
            if len(self.columns_data[el]["one_hot"]["one_hot_idx"]) != 0
        }

        self._masked_idx = {
            (
                self.columns_data[el]["col_idx"],
                self.columns_data[el]["mask_idx"]
            ): el for el in self.columns_data
            if self.columns_data[el]["mask_idx"] is not None
        }

        self._ordinal_idx = {
            self.columns_data[el]["col_idx"]: el
            for el in self.columns_data
            if self.columns_data[el]["mask_idx"] is None
            and self.columns_data[el]["col_idx"] is not None
        }
        self._ordinal_idx_list = list(self._ordinal_idx.keys())

    def fit(self, df: pd.DataFrame) -> None:
        self._check_df(df)
        self.initial_columns_order = df.columns.tolist()

        new_df = OrderedDict()
        i = 0

        for el in df.columns:
            self.columns_data[el] = {
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
                    self.columns_data[el]["col_idx"] = i
                    self.columns_data[el]["mask_idx"] = i + 1
                    i += 2
                else:
                    new_df[el] = df[el]
                    self.columns_data[el]["col_idx"] = i
                    i += 1

                if (df[el][~mask].round() == df[el][~mask]).fillna(False).all():
                    self.columns_data[el]["apply_round"] = True

            else:
                cat_df = pd.get_dummies(df[el], dummy_na=True)
                self.columns_data[el]["one_hot"]["categories"] = cat_df.columns.to_list()
                self.columns_data[el]["one_hot"]["one_hot_idx"] = [i, i + cat_df.shape[1] - 1]
                i += cat_df.shape[1]

                for col in cat_df.columns:
                    new_df[f"{el}_{col}"] = cat_df[col]

        self.final_scaler.fit(pd.DataFrame(new_df).to_numpy())

        self.final_size = len(new_df)
        self._prepare_indices()

        self.fitted = True

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        self._check_df(df)

        new_df = OrderedDict()

        for el in self.initial_columns_order:
            if self.columns_data[el]["col_idx"] is not None:
                new_df[el] = df[el]

                if self.columns_data[el]["mask_idx"] is not None:
                    new_df[el] = df[el].fillna(0, inplace=False)
                    new_df[f"{el}_mask"] = df[el].isna()
                else:
                    new_df[el] = df[el]

            else:
                cat_df = pd.get_dummies(df[el], dummy_na=True)
                cat_df_cols_set = set(cat_df.columns)

                extra = cat_df_cols_set - set(self.columns_data[el]["one_hot"]["categories"])

                if extra:
                    raise ValueError(f"Unexpected categories: {extra}")

                for col in self.columns_data[el]["one_hot"]["categories"]:
                    if col not in cat_df_cols_set:
                        new_df[f"{el}_{col}"] = [False] * df.shape[0]
                    else:
                        new_df[f"{el}_{col}"] = cat_df[col]

        return torch.tensor(
            self.final_scaler.transform(
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
        if not self.fitted:
            raise RuntimeError(
                "fit method must be called before inverse_transform"
            )

        tensor_unscaled = self.final_scaler.inverse_transform(tensor)
        new_df = {value: tensor_unscaled[:, key] for key, value in self._ordinal_idx.items()}

        for key, value in self._softmax_idx.items():
            new_df[value] = [
                self.columns_data["one_hot"]["categories"][i]
                for i in np.argmax(tensor_unscaled[:, key[0]: key[1] + 1].numpy(), axis=1)
            ]

        for key, value in self._masked_idx.items():
            ser_tens = pd.Series(tensor_unscaled[:, key[0]].numpy())
            mask = tensor_unscaled[:, key[1]].numpy() >= 0.5
            ser_tens[mask] = np.nan
            new_df[value] = ser_tens

        df = pd.DataFrame(new_df)
        for el in self.columns_data:

            if self.columns_data[el]["apply_round"]:
                df[el] = df[el].round()

            df[el] = df[el].astype(self.columns_data[el]["dtype"])


        return df


    def get_params(self) -> dict:
        pass

    @classmethod
    def save_params(cls, params: dict) -> "DataTransformer":
        pass

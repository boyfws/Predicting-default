import pandas as pd
import numpy as np
import torch

from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class DataTransformer:
    def fit_transform(self, df: pd.DataFrame) -> torch.Tensor:
        df = df.copy()

        self.final_scaler = StandardScaler()

        self.encoders = {}
        self.min_max = {}

        self.columns = []

        for el in df.columns:

            if is_numeric_dtype(df[el]):
                mask = df[el].isna()
                if mask.any():
                    df[f"{el}_mask"] = mask

                df[el] = df[el].fillna(0)
            else:
                self.encoders[el] = OrdinalEncoder(
                    categories=[df[el].unique()],
                    encoded_missing_value=-1
                )

                transformed = self.encoders[el].fit_transform(df[el].to_numpy().reshape(-1, 1))

                self.min_max[el] = [transformed.min(), transformed.max()]

                df[el] = transformed

        self.columns = df.columns

        scaled = self.final_scaler.fit_transform(df.to_numpy())

        return torch.Tensor(scaled.astype(np.float32))

    def inverse_transform(self, tensor: torch.Tensor) -> pd.DataFrame:
        tensor_unscaled = self.final_scaler.inverse_transform(tensor)

        df = pd.DataFrame(tensor_unscaled, columns=self.columns)
        for el in df.columns:

            if el in self.encoders:
                df[el] = self.encoders[el].inverse_transform(
                    df[el]
                    .clip(
                        lower=self.min_max[el][0],
                        upper=self.min_max[el][1]
                    )
                    .round()
                    .astype(int)
                    .to_numpy()
                    .reshape(-1, 1)
                ).reshape(-1)

            if el.endswith("_mask"):
                col = el[:-5]

                mask = df[el] >= 0.5

                df.loc[mask, col] = np.nan

                df = df.drop(el, axis=1)

        return df
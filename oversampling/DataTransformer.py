import pandas as pd
import numpy as np
import torch

from pandas.api.types import is_numeric_dtype, is_bool_dtype
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


class DataTransformer:
    def __init__(self):
        self.final_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.encoders = {}
        self.min_max = {}
        self.types = {}
        self.apply_round = {}

        self.columns = []

        self.fitted = False

    def fit_transform(
            self,
            df: pd.DataFrame
    ) -> torch.Tensor:
        if df.empty:
            raise RuntimeError('DataFrame is empty')

        if df.select_dtypes(include="number").map(np.isinf).any(axis=None):
            raise RuntimeError('DataFrame is infinite')

        df = df.copy()

        for el in df.columns:
            self.types[el] = df[el].dtype
            self.apply_round[el] = False

            if is_numeric_dtype(df[el]) or is_bool_dtype(df[el]):

                mask = df[el].isna()
                if mask.any():
                    df[f"{el}_mask"] = mask

                if (
                        df[el][~mask].round() == df[el][~mask]
                ).fillna(False).all():
                    self.apply_round[el] = True

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

        self.fitted = True

        return torch.Tensor(scaled.astype(np.float32))

    def inverse_transform(self, tensor: torch.Tensor) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError('fit_transform method must be called before inverse_transform')

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

        for el in self.types:

            if self.apply_round[el]:
                df[el] = df[el].round()

            df[el] = df[el].astype(self.types[el])

        return df

    def get_params(self) -> dict:
        if not self.fitted:
            raise RuntimeError('fit_transform method must be called before get_params')
        encoders = {}
        for col, enc in self.encoders.items():
            mi = None
            if hasattr(enc, '_missing_indices'):
                mi = enc._missing_indices

            else:
                cats = enc.categories_[0]
                mask = pd.isna(cats)
                idx = int(np.where(mask)[0][0]) if mask.any() else None
                mi = [idx]

            encoders[col] = {
                'categories': enc.categories_[0].tolist(),
                'encoded_missing_value': enc.encoded_missing_value,
                'missing_indices': mi
            }

        return {
            'final_scaler': {
                'scale_': self.final_scaler.scale_.tolist(),
                'min_': self.final_scaler.min_.tolist(),
                'data_min_': self.final_scaler.data_min_.tolist(),
                'data_max_': self.final_scaler.data_max_.tolist(),
                'feature_range': self.final_scaler.feature_range
            },
            'encoders': encoders,
            'min_max': self.min_max,
            'types': {col: str(dtype) for col, dtype in self.types.items()},
            'columns': self.columns.tolist(),
            "apply_round": self.apply_round,
        }

    @classmethod
    def save_params(cls, params: dict) -> 'DataTransformer':
        self = cls.__new__(cls)

        fs = params['final_scaler']
        scaler = MinMaxScaler(feature_range=tuple(fs['feature_range']))
        scaler.scale_ = np.array(fs['scale_'])
        scaler.min_ = np.array(fs['min_'])
        scaler.data_min_ = np.array(fs['data_min_'])
        scaler.data_max_ = np.array(fs['data_max_'])
        scaler.data_range_ = scaler.data_max_ - scaler.data_min_
        self.final_scaler = scaler

        # encoders
        self.encoders = {}
        for col, st in params['encoders'].items():
            enc = OrdinalEncoder(
                categories=[st['categories']],
                encoded_missing_value=st['encoded_missing_value']
            )
            enc.categories_ = [np.array(st['categories'])]
            enc._missing_indices = st['missing_indices']
            self.encoders[col] = enc

        # other
        self.min_max = params['min_max']
        self.types = {col: np.dtype(dt) for col, dt in params['types'].items()}
        self.columns = pd.Index(params['columns'])

        self.apply_round = params['apply_round']

        self.fitted = True

        return self

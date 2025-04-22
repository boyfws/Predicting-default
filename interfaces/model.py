from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Model(ABC):
    """Interface for logistic regression classes"""

    @abstractmethod
    def fit(self,
            X: pd.DataFrame,
            y: pd.Series
            ) -> None:
        """
        Training the model on the data

        Parameters:
        X - feature matrix (n_samples, n_features)
        y - vector of target values (n_samples,)
        """
        pass

    @abstractmethod
    def predict_proba(
            self,
            X: pd.DataFrame
    ) -> np.ndarray:
        """
        Predicting class probabilities

        Parameters:
        X is the feature matrix (n_samples, n_features)

        Returns:
        Probabilities for class 1 (n_samples, )
        """
        pass

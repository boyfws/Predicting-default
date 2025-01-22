
from prepare_data_for_emb import DataPreparator
from create_embeddings import EmbedCreator

import faiss
from typing import Iterable, Any
import numpy as np
from numpy.typing import NDArray


class KNNClassifier(DataPreparator):
    def _create_faiss_index(self) -> None:
        if self._gpu:
            res = faiss.StandardGpuResources()
            self._index = faiss.GpuIndexFlatIP(res, self._NDIM)
        else:
            self._index = faiss.IndexFlatIP(self._NDIM)

    def __init__(self, gpu: bool = False, NDIM: int = 512) -> None:
        self._gpu = gpu
        self._NDIM = NDIM

        self._create_faiss_index()
        self._emb_creator = EmbedCreator()

        super().__init__()

    def _convert_data_to_emb(self, data: Iterable[str]) -> NDArray:
        prepared_data = self.prepare(data)

        mode = "CPU"
        if self._gpu:
            mode = "GPU"

        return self._emb_creator.create_embed(prepared_data, mode).numpy()

    def _determine_label_index(self, dist: NDArray, labels: NDArray, limit: float) -> int:

        dist = np.clip(dist, a_min=0, a_max=None)
        weights = 1 / (dist + 1e-10)

        unique_labels = np.unique(labels)

        class_weights = np.array([
            np.sum(weights[labels == label]) for label in unique_labels
        ])

        best_label = unique_labels[np.argmax(class_weights)]

        return best_label

    def fit(self, data: Iterable[str], labels: Iterable[str]) -> None:
        self._labels = np.unique(list(labels))
        self._labels.sort()

        self._data_mask = np.searchsorted(self._labels, np.array(labels))

        self._index.add(self._convert_data_to_emb(data))

    def predict(self, X: Iterable[str], k: int, limit: float = 1):
        X_emb = self._convert_data_to_emb(X)
        D, L = self._index.search(X_emb, k)

        ret_labels_index = np.array(
            [self._determine_label_index(1 - d, self._data_mask[l], limit) for d, l in zip(D, L)]
        )
        return self._labels[ret_labels_index]


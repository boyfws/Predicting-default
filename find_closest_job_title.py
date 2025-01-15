from create_embeddings import EmbedCreator

from typing import Literal
import faiss
from numpy.typing import NDArray
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')



DEFAULT_LABELS = [
    'Accountant',
    'Manual laborer',
    "Consultant",
    "Constructor",
    'Creative Arts',
    'Driver',
    'Customer Service',
    'Engineer',
    'Entertainment',
    'Finance',
    'IT',
    'Legal',
    'Manager',
    'Medical worker',
    'Postal worker',
    'Policeman',
    'Soldier',
    'Sales and Marketing',
    'Scientist',
    'Social worker',
    'Teacher or Professor',
    'Transportation worker'
]


class FindClosestJobTitle:
    """
    Выполняет следующую задачу:
    К нам приходит двумерный массив из нормализованных embedding-ов
    с описанием рода деятельности человека, нам нужно отнести его к одной из
    заданных групп i.e решается задача классификации

    Мы используем faiss для быстрого поиска наиболее близкого вектора
    """
    _NDIM: int = 512
    _lemmatizer = WordNetLemmatizer()

    @staticmethod
    def _lemmatize_phrase(phrase: str) -> str:
        words = word_tokenize(phrase)
        lemmatized_words = [FindClosestJobTitle._lemmatizer.lemmatize(word.strip(), pos='v') for word in words]
        return ' '.join(lemmatized_words)

    @staticmethod
    def _prep_one_label(s: str) -> str:
        s = s.lower()
        s = re.sub(r'[^\w\s]', '', s)
        s = re.sub(r'\s+', ' ', s)
        return FindClosestJobTitle._lemmatize_phrase(s)

    @staticmethod
    def _prepare_labels(
            labels: list[str]
    ) -> list[str]:
        """
        Подготавливает переданные данные для их конвертации в эмбеддинги
        """
        labels = [FindClosestJobTitle._prep_one_label(el) for el in labels]

        return labels

    @staticmethod
    def _get_mode(gpu: bool) -> Literal["CPU", "GPU"]:
        """
        Возвращает mode для EmbedCreator
        """
        mode = "CPU"
        if gpu:
            mode = "GPU"

        return mode

    def _convert_labels_to_emb(self,
                               labels: list[str],
                               gpu: bool) -> NDArray:
        """
        Конвертирует переданные данные в эмбеддинги
        """
        good_labels = self._prepare_labels(labels)

        mode = self._get_mode(gpu)

        embeddings = self._emb_creator.create_embed(good_labels, mode)

        return embeddings.numpy()

    def _create_faiss_index(self,
                            embeddings: NDArray,
                            gpu: bool) -> None:
        if gpu:
            res = faiss.StandardGpuResources()
            self._index = faiss.GpuIndexFlatIP(res, self._NDIM)
        else:
            self._index = faiss.IndexFlatIP(self._NDIM)

        self._index.add(embeddings)

    def __init__(self,
                 gpu: bool = False,
                 labels: list[str] = DEFAULT_LABELS
                 ) -> None:

        self._emb_creator = EmbedCreator()

        embeddings = self._convert_labels_to_emb(
            labels=labels,
            gpu=gpu
        )

        self._create_faiss_index(
            embeddings,
            gpu
        )

        self._labels = labels

    def predict(self,
                X: NDArray,
                noise_element: str = "NOISE",
                limit: float = 0.5,
                ) -> tuple[NDArray, NDArray]:
        D, L = self._index.search(X, 1)

        labels = np.array(
            [noise_element if (1 - d[0]) > limit or l[0] == -1 else self._labels[l[0]] for d, l in zip(D, L)]
        )

        dist = np.array(
            [1 - d[0] if labels[i] != noise_element else np.NaN for i, d in enumerate(D)], dtype=np.float32
        )

        return dist, labels

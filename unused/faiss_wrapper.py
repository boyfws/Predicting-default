import faiss
from numpy.typing import NDArray

# В итоге не использовалось
# Есть только в старых коммитах на Kaggle

class FaissKMeans:
    def __init__(self,
                 n_clusters: int,
                 max_iter: int = 25,
                 n_init: int = 1,
                 verbose: bool = False,
                 random_state: int = 1234,
                 max_points_per_centroid: int = 256,
                 min_points_per_centroid: int = 39,
                 update_index: bool = False,
                 int_centroids: bool = False
                 ) -> None:
        """
        Инициализируем кластеризатор

        :param n_clusters: The number of clusters
        :param max_iter: The maximum number of iterations
        :param n_init: Run the clustering this number of times, and keep the best centroids (selected according to clustering objective)
        :param verbose:
        :param random_state: Seed for the random number generator
        :param max_points_per_centroid:
        :param min_points_per_centroid:
        :param update_index: Re-train index after each iteration?
        :param int_centroids: Round centroids coordinates to integer
        """

        self.n_c = n_clusters
        self.niter = max_iter
        self.nredo = n_init
        self.verbose = verbose
        self.seed = random_state
        self.max_points_per_centroid = max_points_per_centroid
        self.min_points_per_centroid = min_points_per_centroid
        self.update_index = update_index
        self.int_centroids = int_centroids

        self.kmeans = None
        self.labels_ = None

    def fit(self,
            X: NDArray,
            gpu: bool = False,
            num_threads: int = 10
            ) -> None:
        """
        Метод для обучения кластеризатора

        :param X: Данные в формате NDArray
        :param gpu: Флаг для обучения на GPU
        :param num_threads: Кол-во ядер для обучения, игнорируется если GPU=True
        """

        if not gpu:
            faiss.omp_set_num_threads(num_threads)

        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_c,
                                   niter=self.niter,
                                   nredo=self.nredo,
                                   verbose=self.verbose,
                                   seed=self.seed,
                                   max_points_per_centroid=self.max_points_per_centroid,
                                   min_points_per_centroid=self.min_points_per_centroid,
                                   update_index=self.update_index,
                                   int_centroids=self.int_centroids,
                                   gpu=gpu)

        self.kmeans.train(X)
        labels = self.kmeans.index.search(X, 1)[1]
        self.labels_ = labels.flatten()

    def labels_(self) -> NDArray:
        """
        Возвращает лейблы для данных, на которых училась модель
        :return: NDArray
        """
        return self.labels_

    def centroids(self) -> NDArray:
        """
        Возвращает центроиды модели
        :return: NDArray
        """
        return self.kmeans.centroids

    def predict(self, X: NDArray) -> NDArray:
        return self.kmeans.index.search(X, 1)[1].flatten()
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Literal, Protocol
import tensorflow as tf
from numpy.typing import NDArray

class EmbedCreator:
    """
    Создаёт эмбеддинги при помощи BERT (PyTorch-версия) из библиотеки transformers.
    """
    MODEL_NAME: str = "bert-base-uncased"

    def __init__(self) -> None:
        # Загружаем токенизатор
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        # Загружаем базовую модель BERT (без "головы" на классификацию)
        self._model = AutoModel.from_pretrained(self.MODEL_NAME)
        # Переводим модель в режим "оценки", чтобы отключить dropout и т.п.
        self._model.eval()

    def create_embed(self, data: NDArray, mode: Literal["CPU", "GPU"] = "CPU") -> tf.Tensor:
        """
        Создаёт эмбеддинги для входных данных с использованием BERT (PyTorch).

        :param data: Входные данные (одна строка или список строк).
        :param mode: Режим выполнения ("CPU" или "GPU").
        :return: Нормализованные эмбеддинги (PyTorch-тензор).
        """
        device = torch.device("cuda" if (mode == "GPU" and torch.cuda.is_available()) else "cpu")
        self._model.to(device)

        inputs = self._tokenizer(
            data.tolist(),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        for k, v in inputs.items():
            inputs[k] = v.to(device)

        with torch.no_grad():

            outputs = self._model(**inputs)
            # В случае BERT "сырое" представление включает:
            #  - last_hidden_state: [batch_size, seq_len, hidden_dim=768]
            #  - pooler_output: [batch_size, hidden_dim=768], если в конфигурации есть пулер
            # Для базовых моделей BERT pooler_output есть (если не отключён в конфиге).
            pooled_output = outputs.pooler_output

            # Нормализация эмбеддингов
            norms = torch.norm(pooled_output, dim=-1, keepdim=True)
            normalized_embeddings = pooled_output / norms


        return tf.convert_to_tensor(
            normalized_embeddings.cpu().detach().numpy()
        )


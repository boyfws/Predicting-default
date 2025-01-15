import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from typing import Literal, Protocol


# Вторая версия для BERT

class HasGetItem(Protocol):
    def __getitem__(self, key):
        pass


class EmbedCreator:
    _BERT_PREPROCESS_URL: str = "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3"
    _BERT_ENCODER_URL: str = "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-l-12-h-768-a-12/4"

    def __init__(self) -> None:
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        preprocessor = hub.KerasLayer(self._BERT_PREPROCESS_URL)
        encoder_inputs = preprocessor(text_input)
        encoder = hub.KerasLayer(self._BERT_ENCODER_URL, trainable=False)
        outputs = encoder(encoder_inputs)
        pooled_output = outputs["pooled_output"]  # [batch_size, 768].
        self._embedding_model = tf.keras.Model(text_input, pooled_output)

    def create_embed(self, data: HasGetItem, mode: Literal["CPU", "GPU"] = "CPU") -> tf.Tensor:
        """
        Создает эмбеддинги для входных данных с использованием BERT.

        :param data: Входные данные (текст или список текстов).
        :param mode: Режим выполнения ("CPU" или "GPU").
        :return: Нормализованные эмбеддинги.
        """
        with tf.device(f'/{mode}:0'):

            embeddings = self._embedding_model(data)

            norms = tf.norm(embeddings, axis=-1, keepdims=True)
            normalized_embeddings = embeddings / norms

        return normalized_embeddings


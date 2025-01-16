from typing import Literal, Protocol
import tensorflow_hub as hub
import tensorflow as tf


class HasGetItem(Protocol):
    def __getitem__(self, key):
        pass


class EmbedCreator:
    _MODEL_URL: str = "https://tfhub.dev/google/universal-sentence-encoder/4"

    def __init__(self) -> None:
        self.model = hub.load(self._MODEL_URL)

    def create_embed(self, data: HasGetItem, mode: Literal["CPU", "GPU"] = "CPU") -> tf.Tensor:
        with tf.device(f'/device:{mode}:0'):
            embeddings = self.model(data)

            norms = tf.norm(embeddings, axis=-1, keepdims=True)
            normalized_embeddings = embeddings / norms
        return normalized_embeddings


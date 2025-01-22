
from typing import Iterable
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


class DataPreparator:
    _lemmatizer = WordNetLemmatizer()

    def __init__(self):
        pass

    @staticmethod
    def _lemmatize_phrase(phrase: str) -> str:
        words = word_tokenize(phrase)
        lemmatized_words = [DataPreparator._lemmatizer.lemmatize(word.strip(), pos='v') for word in words]
        return ' '.join(lemmatized_words)

    @staticmethod
    def _prep_one_label(s: str) -> str:
        s = s.lower()
        s = re.sub(r'[^\w\s]', '', s)
        s = re.sub(r'\s+', ' ', s)
        return DataPreparator._lemmatize_phrase(s)

    @staticmethod
    def _prepare_labels(
            labels: Iterable[str]
    ) -> list[str]:
        """
        Подготавливает переданные данные для их конвертации в эмбеддинги
        """
        labels = [DataPreparator._prep_one_label(el) for el in labels]

        return labels

    @staticmethod
    def prepare(data: Iterable[str]) -> list[str]:
        return DataPreparator._prepare_labels(data)
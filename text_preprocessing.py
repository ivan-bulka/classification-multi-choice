import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import spacy
import spacy.cli
spacy.cli.download("en_core_web_lg")

class TextProcessor:
    """
    Methods to text processing
    Can be used independently
    """
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.ps = PorterStemmer()

    @staticmethod
    def lowercasing(text):
        """
        Lowercasing a text string
        """
        return text.lower()

    @staticmethod
    def remove_punctuation(text):
        """
        Removing punctuation from a string
        """
        punctuation_pattern = r'[^\w\s]'
        return re.sub(punctuation_pattern, '', text)

    @staticmethod
    def remove_stop_words(text, language="english") -> str:
        """
        Removing stopwords from the string
        """
        stop_words = set(stopwords.words(language))
        word_tokens = text.split()
        return " ".join([word for word in word_tokens if word not in stop_words])

    def lemmatization(self, text: str) -> str:
        """
        Text lemmatization
        """
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def stemming(self, text: str) -> str:
        """
        Text stemming
        """
        words = word_tokenize(text)
        return " ".join([self.ps.stem(word) for word in words])

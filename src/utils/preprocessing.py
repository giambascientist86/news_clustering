import re
import json
import spacy
import nltk
import logging
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser
from typing import List, Optional
from pathlib import Path
from contractions import fix

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('punkt_tab')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

class TextPreprocessor:
    """
    A class for preprocessing text data including cleaning, tokenization, stopword removal,
    lemmatization, named entity handling, and phrase detection.
    """

    def __init__(self, stopword_removal: bool = True, remove_named_entities: bool = False):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.remove_named_entities = remove_named_entities
        self.phrases = None  # Placeholder for bigram model

    def clean_text(self, text: str) -> str:
        """Cleans text by lowercasing, removing punctuation, numbers, and fixing contractions."""
        text = text.lower()
        text = fix(text)  # Expand contractions
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters & numbers
        return text

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenizes and lemmatizes text while optionally removing stopwords."""
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        if self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
        return tokens

    def remove_named_entities(self, text: str) -> str:
        """Removes named entities (e.g., names, locations) using spaCy."""
        doc = nlp(text)
        return " ".join([token.text for token in doc if not token.ent_type_])

    def detect_phrases(self, sentences: List[List[str]]) -> List[List[str]]:
        """Detects multi-word expressions using Gensim's Phrases model."""
        if not self.phrases:
            self.phrases = Phraser(Phrases(sentences, min_count=5, threshold=10))
        return [self.phrases[sentence] for sentence in sentences]

    def preprocess(self, text: str) -> str:
        """Applies the full preprocessing pipeline to a single text input."""
        text = self.clean_text(text)
        if self.remove_named_entities:
            text = self.remove_named_entities(text)
        tokens = self.tokenize_and_lemmatize(text)
        return " ".join(tokens)

    def process_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """Applies preprocessing to specified text columns in a DataFrame."""
        for col in text_columns:
            df[col] = df[col].astype(str).apply(self.preprocess)
        return df

if __name__ == "__main__":
    file_path = Path("C:/Users/batti/topic-modeling/data/Huff_news/news.jsonl")
    df = pd.read_json(file_path, lines=True)
    processor = TextPreprocessor()
    cleaned_df = processor.process_dataframe(df, text_columns=["headline", "short_description"])
    cleaned_df.to_csv("C:/Users/batti/topic-modeling/data/cleaned_news.csv", index=False)
    logging.info("Preprocessing complete. Cleaned dataset saved.")

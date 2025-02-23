import logging
import multiprocessing
import numpy as np
import scipy.sparse as sp
import pandas as pd
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sentence_transformers import SentenceTransformer
import torch
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TextVectorizer:
    def __init__(self, method: str = "tfidf", model_name: str = "all-MiniLM-L6-v2", vector_size: int = 50):
        self.method = method.lower()
        self.model_name = model_name
        self.vector_size = vector_size
        self.vectorizer = None
        self.model = None
        self.embeddings = None

        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        elif self.method == "bert":
            self.model = SentenceTransformer(self.model_name)
            if torch.cuda.is_available():
                self.model.to('cuda')
        elif self.method not in ["word2vec", "doc2vec"]:
            raise ValueError("Invalid vectorization method. Choose from 'tfidf', 'word2vec', 'bert', or 'doc2vec'.")

    def fit_tfidf(self, texts: List[str]) -> np.ndarray:
        logging.info("Fitting TF-IDF...")
        self.embeddings = self.vectorizer.fit_transform(texts)
        return self.embeddings.toarray()

    def fit_word2vec(self, texts: List[List[str]]) -> np.ndarray:
        logging.info("Training Word2Vec...")
        model = Word2Vec(sentences=texts, vector_size=self.vector_size, window=5, min_count=1, workers=multiprocessing.cpu_count())
        self.embeddings = np.array([np.mean([model.wv[word] for word in doc if word in model.wv] or [np.zeros(self.vector_size)], axis=0) for doc in texts])
        return self.embeddings

    def fit_bert(self, texts: List[str]) -> np.ndarray:
        logging.info("Generating BERT embeddings...")
        self.embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        return self.embeddings

    def fit_doc2vec(self, texts: List[List[str]]) -> np.ndarray:
        logging.info("Training Doc2Vec...")
        tagged_data = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(texts)]
        model = Doc2Vec(tagged_data, vector_size=self.vector_size, window=5, min_count=1, workers=multiprocessing.cpu_count(), epochs=20)
        self.embeddings = np.array([model.dv[i] for i in range(len(texts))])
        return self.embeddings

    def transform(self, texts: List[Union[str, List[str]]]) -> np.ndarray:
        if self.method in ["tfidf", "bert"]:
            return self.fit_tfidf(texts) if self.method == "tfidf" else self.fit_bert(texts)
        elif self.method in ["word2vec", "doc2vec"]:
            if isinstance(texts[0], str):
                texts = [text.split() for text in texts]
            return self.fit_word2vec(texts) if self.method == "word2vec" else self.fit_doc2vec(texts)
        else:
            raise ValueError("Unsupported vectorization method.")

    def save_embeddings(self, filepath: str):
        if self.embeddings is None:
            raise ValueError("No embeddings found. Run 'transform' first.")
        if isinstance(self.embeddings, sp.spmatrix):
            self.embeddings = self.embeddings.toarray()
        np.save(filepath, self.embeddings)
        logging.info(f"Embeddings saved to {filepath}")

    def load_embeddings(self, filepath: str) -> np.ndarray:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")
        self.embeddings = np.load(filepath, allow_pickle=True)
        logging.info(f"Embeddings loaded from {filepath}")
        return self.embeddings


def main():
    df = pd.read_csv("../../data/cleaned_news.csv")
    texts = (df["headline"] + " " + df["short_description"]).fillna("").tolist()

    vectorizers = {
        "tfidf": TextVectorizer(method="tfidf"),
        "bert": TextVectorizer(method="bert"),
        "word2vec": TextVectorizer(method="word2vec"),
        "doc2vec": TextVectorizer(method="doc2vec"),
    }
    
    for method, vectorizer in vectorizers.items():
        embeddings = vectorizer.transform(texts if method in ["tfidf", "bert"] else [text.split() for text in texts])
        vectorizer.save_embeddings(f"../../data/{method}_embeddings.npy")

if __name__ == "__main__":
    main()
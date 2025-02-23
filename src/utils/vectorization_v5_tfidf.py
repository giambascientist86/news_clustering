import logging
import multiprocessing
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
import torch
from typing import List, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sentence_transformers import SentenceTransformer

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
        self.tfidf_matrix = None

        # Initialize vectorization methods
        if self.method in ["tfidf", "both"]:
            self.vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        if self.method == "bert" or self.method == "both":
            self.model = SentenceTransformer(self.model_name)
            if torch.cuda.is_available():
                self.model.to('cuda')
        if self.method not in ["tfidf", "word2vec", "bert", "doc2vec", "both"]:
            raise ValueError("Invalid vectorization method. Choose from 'tfidf', 'word2vec', 'bert', 'doc2vec', or 'both'.")

    def fit_tfidf(self, texts: List[str]) -> sp.csr_matrix:
        """Computes TF-IDF matrix"""
        logging.info("Fitting TF-IDF...")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        return self.tfidf_matrix

    def fit_word2vec(self, texts: List[List[str]]) -> np.ndarray:
        """Trains Word2Vec model and generates embeddings"""
        logging.info("Training Word2Vec...")
        model = Word2Vec(sentences=texts, vector_size=self.vector_size, window=5, min_count=1, workers=multiprocessing.cpu_count())
        self.embeddings = np.array([
            np.mean([model.wv[word] for word in doc if word in model.wv] or [np.zeros(self.vector_size)], axis=0)
            for doc in texts
        ])
        return self.embeddings

    def fit_bert(self, texts: List[str]) -> np.ndarray:
        """Generates BERT embeddings"""
        logging.info("Generating BERT embeddings...")
        self.embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        return self.embeddings

    def fit_doc2vec(self, texts: List[List[str]]) -> np.ndarray:
        """Trains Doc2Vec model and generates embeddings"""
        logging.info("Training Doc2Vec...")
        tagged_data = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(texts)]
        model = Doc2Vec(tagged_data, vector_size=self.vector_size, window=5, min_count=1, workers=multiprocessing.cpu_count(), epochs=20)
        self.embeddings = np.array([model.dv[i] for i in range(len(texts))])
        return self.embeddings

    def transform(self, texts: List[Union[str, List[str]]]) -> Union[np.ndarray, sp.csr_matrix]:
        """Applies selected vectorization method"""
        if self.method in ["tfidf", "bert", "both"]:
            return self.fit_tfidf(texts) if self.method == "tfidf" else self.fit_bert(texts)
        else:
            tokenized_texts = [text.split() for text in texts]
            return self.fit_word2vec(tokenized_texts) if self.method == "word2vec" else self.fit_doc2vec(tokenized_texts)

    def get_dimensions(self) -> Tuple[int, int]:
        """Computes dimensions of the generated embeddings"""
        if self.method == "tfidf" and self.tfidf_matrix is not None:
            return self.tfidf_matrix.shape
        elif self.embeddings is not None:
            return self.embeddings.shape
        else:
            raise ValueError("No embeddings or TF-IDF matrix found. Run 'transform' first.")

    def save_embeddings(self, filepath: str):
        """Saves embeddings and logs their dimensions"""
        if self.embeddings is None:
            raise ValueError("No embeddings found. Run 'transform' first.")
        np.save(filepath, self.embeddings)
        logging.info(f"Embeddings saved to {filepath}, shape: {self.embeddings.shape}")

    def save_tfidf(self, filepath: str):
        """Saves TF-IDF matrix and logs its dimensions"""
        if self.tfidf_matrix is None:
            raise ValueError("No TF-IDF matrix found. Run 'transform' first.")
        sp.save_npz(filepath, self.tfidf_matrix)
        logging.info(f"TF-IDF matrix saved to {filepath}, shape: {self.tfidf_matrix.shape}")

    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Loads saved embeddings"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")
        self.embeddings = np.load(filepath, allow_pickle=True)
        logging.info(f"Embeddings loaded from {filepath}, shape: {self.embeddings.shape}")
        return self.embeddings

    def load_tfidf(self, filepath: str) -> sp.csr_matrix:
        """Loads saved TF-IDF matrix"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")
        self.tfidf_matrix = sp.load_npz(filepath)
        logging.info(f"TF-IDF matrix loaded from {filepath}, shape: {self.tfidf_matrix.shape}")
        return self.tfidf_matrix


def main():
    """Main function to process text data and compute embeddings"""
    df = pd.read_csv("../../data/cleaned_news.csv")
    texts = df["short_description"].fillna("").tolist()

    vectorizers = {
        "tfidf": TextVectorizer(method="tfidf"),
        "bert": TextVectorizer(method="bert"),
        "word2vec": TextVectorizer(method="word2vec"),
        "doc2vec": TextVectorizer(method="doc2vec"),
        #"both": TextVectorizer(method="both"),
    }

    for method, vectorizer in vectorizers.items():
        try:
            result = vectorizer.transform(texts)
            dimensions = vectorizer.get_dimensions()
            logging.info(f"Vectorization method: {method}, Shape: {dimensions}")

            save_path = f"../../data/{method}_"
            if method == "both":
                tfidf, embeddings = result
                vectorizer.save_tfidf(save_path + "tfidf.npz")
                vectorizer.save_embeddings(save_path + "embeddings.npy")
            elif method == "tfidf":
                vectorizer.save_tfidf(save_path + "tfidf.npz")
            else:
                vectorizer.save_embeddings(save_path + "embeddings.npy")

        except Exception as e:
            logging.error(f"Error processing {method}: {e}")


if __name__ == "__main__":
    main()

import os
import logging
import pickle
import multiprocessing
from typing import List, Union
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TextVectorizer:
    """
    A class for text vectorization using various embedding techniques.
    Supports TF-IDF, Word2Vec, BERT, and Doc2Vec.
    """
    def __init__(self, method: str = "tfidf", model_name: str = "all-MiniLM-L6-v2", vector_size: int = 50):
        """
        Initialize the vectorizer.
        
        :param method: Vectorization method: 'tfidf', 'word2vec', 'bert', or 'doc2vec'
        :param model_name: Pretrained BERT model name (for SentenceTransformers)
        :param vector_size: Embedding size for Word2Vec/Doc2Vec
        """
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
                self.model.to("cuda")
        elif self.method not in ["word2vec", "doc2vec"]:
            raise ValueError("Invalid method. Choose from 'tfidf', 'word2vec', 'bert', or 'doc2vec'.")

    def fit_tfidf(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts using TF-IDF."""
        logging.info("Fitting TF-IDF...")
        return self.vectorizer.fit_transform(texts).toarray()

    def fit_word2vec(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """Train Word2Vec and generate document embeddings."""
        logging.info("Training Word2Vec...")
        model = Word2Vec(sentences=tokenized_texts, vector_size=self.vector_size, window=5, min_count=1, workers=multiprocessing.cpu_count())
        return np.array([np.mean([model.wv[word] for word in doc if word in model.wv] or [np.zeros(self.vector_size)], axis=0) for doc in tokenized_texts])

    def fit_bert(self, texts: List[str]) -> np.ndarray:
        """Generate sentence embeddings using a BERT model."""
        logging.info("Generating BERT embeddings...")
        return self.model.encode(texts, convert_to_numpy=True, batch_size=32)

    def fit_doc2vec(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """Train a Doc2Vec model and return document embeddings."""
        logging.info("Training Doc2Vec...")
        tagged_data = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(tokenized_texts)]
        model = Doc2Vec(tagged_data, vector_size=self.vector_size, window=5, min_count=1, workers=multiprocessing.cpu_count(), epochs=20)
        return np.array([model.dv[i] for i in range(len(tokenized_texts))])

    def transform(self, texts: List[Union[str, List[str]]]) -> np.ndarray:
        """Transforms input text into vector representations."""
        if not texts:
            raise ValueError("Input text list is empty. Provide valid text data.")

        if self.method == "tfidf":
            self.embeddings = self.fit_tfidf(texts)
        elif self.method == "word2vec":
            tokenized_texts = [text.split() for text in texts]
            self.embeddings = self.fit_word2vec(tokenized_texts)
        elif self.method == "bert":
            self.embeddings = self.fit_bert(texts)
        elif self.method == "doc2vec":
            tokenized_texts = [text.split() for text in texts]
            self.embeddings = self.fit_doc2vec(tokenized_texts)
        else:
            raise ValueError("Unsupported vectorization method.")

        return self.embeddings

    def save_embeddings(self, file_path: str):
        """Saves the computed embeddings to a file."""
        if self.embeddings is None:
            raise ValueError("No embeddings available. Run `transform()` first.")
        with open(file_path, "wb") as f:
            pickle.dump(self.embeddings, f)
        logging.info(f"Embeddings saved to {file_path}")

    def load_embeddings(self, file_path: str) -> np.ndarray:
        """Loads embeddings from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        with open(file_path, "rb") as f:
            self.embeddings = pickle.load(f)
        logging.info(f"Embeddings loaded from {file_path}")
        return self.embeddings


def main():
    """Example usage of TextVectorizer"""
    try:
        # Load dataset
        df = pd.read_csv("../../data/cleaned_news.csv")
        texts = (df["headline"] + " " + df["short_description"]).fillna("").tolist()

        # TF-IDF Vectorization
        tfidf_vectorizer = TextVectorizer(method="tfidf")
        tfidf_vectors = tfidf_vectorizer.transform(texts)
        tfidf_vectorizer.save_embeddings("../../data/tfidf_embeddings.pkl")

        # BERT Vectorization
        bert_vectorizer = TextVectorizer(method="bert")
        bert_vectors = bert_vectorizer.transform(texts)
        bert_vectorizer.save_embeddings("../../data/bert_embeddings.pkl")

        # Word2Vec Vectorization
        w2v_vectorizer = TextVectorizer(method="word2vec")
        tokenized_texts = [text.split() for text in texts]
        w2v_vectors = w2v_vectorizer.transform(tokenized_texts)
        w2v_vectorizer.save_embeddings("../../data/word2vec_embeddings.pkl")

        # Doc2Vec Vectorization
        doc2vec_vectorizer = TextVectorizer(method="doc2vec")
        doc2vec_vectors = doc2vec_vectorizer.transform(tokenized_texts)
        doc2vec_vectorizer.save_embeddings("../../data/doc2vec_embeddings.pkl")

        # Load saved embeddings (example)
        loaded_bert_vectors = bert_vectorizer.load_embeddings("../../data/bert_embeddings.pkl")
        logging.info(f"Loaded BERT embeddings shape: {loaded_bert_vectors.shape}")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

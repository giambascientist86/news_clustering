import logging
import multiprocessing
import pickle
from typing import List, Union
import pandas as pd
import numpy as np
import scipy
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TextVectorizer:
    """
    A class that implements various text vectorization techniques.
    Supports TF-IDF, Word2Vec, BERT, and Doc2Vec.
    """
    def __init__(self, method: str = "tfidf", model_name: str = "all-MiniLM-L6-v2", vector_size: int = 50):
        """
        Initializes the vectorizer with the specified method.
        :param method: The vectorization method ("tfidf", "word2vec", "bert", "doc2vec")
        :param model_name: Pretrained model name for BERT embeddings
        :param vector_size: Dimensionality for Word2Vec and Doc2Vec
        """
        self.method = method.lower()
        self.model_name = model_name
        self.vector_size = vector_size
        self.vectorizer = None
        self.model = None
        self.embeddings = None  # Store embeddings after transformation

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
        self.embeddings = self.vectorizer.fit_transform(texts).toarray()
        return self.embeddings

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
            # Ensure input is tokenized
            if isinstance(texts[0], str):
                texts = [text.split() for text in texts]
            return self.fit_word2vec(texts) if self.method == "word2vec" else self.fit_doc2vec(texts)
        else:
            raise ValueError("Unsupported vectorization method.")
    def save_embeddings(self, filepath: str):
        """Save computed embeddings to a file."""
        if self.embeddings is None:
            raise ValueError("No embeddings found. Run 'transform' first.")
        
        # Controllo e conversione
        if isinstance(self.embeddings, list):
            self.embeddings = np.array(self.embeddings)
        elif isinstance(self.embeddings, sparse.spmatrix):  # Per le matrici sparse
            self.embeddings = self.embeddings.toarray()

        with open(filepath, "wb") as f:
            pickle.dump(self.embeddings, f)
        logging.info(f"Embeddings saved to {filepath}") 

    def load_embeddings(self, filepath: str):
        """Load embeddings from a file."""
        with open(filepath, "rb") as f:
            self.embeddings = pickle.load(f)

        if sparse.issparse(self.embeddings):
            logging.info("Embedding sparse rilevati. Convertiamo a numpy...")
            self.embeddings = self.embeddings.toarray()

        logging.info(f"Embeddings loaded from {filepath}")
        return self.embeddings
    
"""
    def save_embeddings(self, filepath: str):
        
        if self.embeddings is None:
            raise ValueError("No embeddings found. Run 'transform' first.")
        with open(filepath, "wb") as f:
            pickle.dump(self.embeddings, f)
        logging.info(f"Embeddings saved to {filepath}")

    def load_embeddings(self, filepath: str):
        
        with open(filepath, "rb") as f:
            self.embeddings = pickle.load(f)
        logging.info(f"Embeddings loaded from {filepath}")
        return self.embeddings

"""
def main():
    """Example usage of the TextVectorizer class."""
    df = pd.read_csv("../../data/cleaned_news.csv")
    texts = (df["headline"] + " " + df["short_description"]).fillna("").tolist()

    # TF-IDF
    tfidf_vectorizer = TextVectorizer(method="tfidf")
    tfidf_vectors = tfidf_vectorizer.transform(texts)
    tfidf_vectorizer.save_embeddings("../../data/tfidf_embeddings.pkl")

    # BERT
    bert_vectorizer = TextVectorizer(method="bert")
    bert_vectors = bert_vectorizer.transform(texts)
    bert_vectorizer.save_embeddings("../../data/bert_embeddings.pkl")

    # Word2Vec
    tokenized_texts = [text.split() for text in texts]
    w2v_vectorizer = TextVectorizer(method="word2vec")
    w2v_vectors = w2v_vectorizer.transform(tokenized_texts)
    w2v_vectorizer.save_embeddings("../../data/w2v_embeddings.pkl")

    # Doc2Vec
    doc2vec_vectorizer = TextVectorizer(method="doc2vec")
    doc2vec_vectors = doc2vec_vectorizer.transform(tokenized_texts)
    doc2vec_vectorizer.save_embeddings("../../data/doc2vec_embeddings.pkl")

if __name__ == "__main__":
    main()

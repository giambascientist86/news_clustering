import logging
import multiprocessing
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
import torch
from typing import List, Union, Tuple, Dict
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
        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        elif self.method == "bert":
            self.model = SentenceTransformer(self.model_name)
            if torch.cuda.is_available():
                self.model.to('cuda')
        elif self.method == "word2vec":
            pass  # Word2Vec model initialized in method
        elif self.method == "doc2vec":
            pass  # Doc2Vec model initialized in method
        elif self.method == "glove":
            self.glove_embeddings = self.load_glove_embeddings()
        else:
            raise ValueError("Invalid method. Choose from 'tfidf', 'word2vec', 'bert', 'doc2vec', 'glove'.")

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

    def fit_glove(self, texts: List[List[str]]) -> np.ndarray:
        """Generates GloVe embeddings by averaging word vectors"""
        logging.info("Generating GloVe embeddings...")
        self.embeddings = np.array([
            np.mean([self.glove_embeddings.get(word, np.zeros(self.vector_size)) for word in doc], axis=0)
            if doc else np.zeros(self.vector_size) for doc in texts
        ])
        return self.embeddings

    def load_glove_embeddings(self, glove_path: str = "../../data/glove.6B.50d.txt") -> Dict[str, np.ndarray]:
        """Loads pretrained GloVe embeddings"""
        logging.info("Loading GloVe embeddings...")
        glove_dict = {}
        if not os.path.exists(glove_path):
            raise FileNotFoundError(f"GloVe file not found: {glove_path}")
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                glove_dict[word] = vector
        return glove_dict

    def transform(self, texts: List[Union[str, List[str]]]) -> Union[np.ndarray, sp.csr_matrix]:
        """Applies selected vectorization method"""
        if self.method == "tfidf":
            return self.fit_tfidf(texts)
        elif self.method == "bert":
            return self.fit_bert(texts)
        else:
            tokenized_texts = [text.split() for text in texts]
            if self.method == "word2vec":
                return self.fit_word2vec(tokenized_texts)
            elif self.method == "doc2vec":
                return self.fit_doc2vec(tokenized_texts)
            elif self.method == "glove":
                return self.fit_glove(tokenized_texts)

    def save_embeddings(self, filepath: str):
        """Saves embeddings and logs their dimensions"""
        if self.embeddings is None:
            raise ValueError("No embeddings found. Run 'transform' first.")
        np.save(filepath, self.embeddings)
        logging.info(f"Embeddings saved to {filepath}, shape: {self.embeddings.shape}")

    def save_tfidf(self, filepath: str, feature_path: str):
        """Saves TF-IDF matrix and feature names."""
        if self.tfidf_matrix is None:
            raise ValueError("No TF-IDF matrix found. Run 'transform' first.")
        
        # Save TF-IDF matrix
        sp.save_npz(filepath, self.tfidf_matrix)
        logging.info(f"TF-IDF matrix saved to {filepath}, shape: {self.tfidf_matrix.shape}")
        
        # Save feature names
        if self.vectorizer is not None:
            feature_names = self.vectorizer.get_feature_names_out()
            np.save(feature_path, feature_names)
            logging.info(f"TF-IDF feature names saved to {feature_path}")


def main():
    """Main function to process text data and compute embeddings"""
    df = pd.read_csv(r"C:\Users\batti\topic-modeling\data\cleaned_news.csv")
    texts = df["short_description"].fillna("").tolist()

    save_tfidf_path = "../../data/tfidf.npz"
    save_features_path = "../../data/tfidf_features.npy"
    
    # Always compute and save TF-IDF if not already saved
    if not os.path.exists(save_tfidf_path) or not os.path.exists(save_features_path):
        tfidf_vectorizer = TextVectorizer(method="tfidf")
        tfidf_matrix = tfidf_vectorizer.transform(texts)
        tfidf_vectorizer.save_tfidf(save_tfidf_path, save_features_path)
    
    # Define vectorizers
    vectorizers = {
        "bert": TextVectorizer(method="bert"),
        "word2vec": TextVectorizer(method="word2vec"),
        "doc2vec": TextVectorizer(method="doc2vec"),
        "glove": TextVectorizer(method="glove"),
    }

    # Compute embeddings
    for method, vectorizer in vectorizers.items():
        try:
            result = vectorizer.transform(texts)
            save_path = f"../../data/{method}_embeddings.npy"
            vectorizer.save_embeddings(save_path)
        except Exception as e:
            logging.error(f"Error processing {method}: {e}")
    

if __name__ == "__main__":
    main()
import logging
from typing import List, Union
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TextVectorizer:
    """
    A class that implements various text vectorization techniques.
    Supports TF-IDF, Word2Vec, BERT, and Doc2Vec.
    """
    def __init__(self, method: str = "tfidf", model_name: str = "paraphrase-MiniLM-L6-v2"):
        """
        Initializes the vectorizer with the specified method.
        :param method: The vectorization method ("tfidf", "word2vec", "bert", "doc2vec")
        :param model_name: Pretrained model name for BERT embeddings
        """
        self.method = method.lower()
        self.model_name = model_name
        self.vectorizer = None
        self.model = None
        
        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer()
        elif self.method == "bert":
            self.model = SentenceTransformer(self.model_name)
        elif self.method not in ["word2vec", "doc2vec"]:
            raise ValueError("Invalid vectorization method. Choose from 'tfidf', 'word2vec', 'bert', or 'doc2vec'.")
    
    def fit_tfidf(self, texts: List[str]) -> np.ndarray:
        """Fits and transforms text using TF-IDF."""
        logging.info("Fitting TF-IDF...")
        return self.vectorizer.fit_transform(texts).toarray()
    
    def fit_word2vec(self, texts: List[List[str]], vector_size: int = 100) -> np.ndarray:
        """Trains a Word2Vec model and returns document embeddings."""
        logging.info("Training Word2Vec...")
        model = Word2Vec(sentences=texts, vector_size=vector_size, window=5, min_count=1, workers=4)
        return np.array([np.mean([model.wv[word] for word in doc if word in model.wv] or [np.zeros(vector_size)], axis=0) for doc in texts])
    
    def fit_bert(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings using a BERT-based model."""
        logging.info("Generating BERT embeddings...")
        return self.model.encode(texts, convert_to_numpy=True)
    
    def fit_doc2vec(self, texts: List[List[str]], vector_size: int = 100) -> np.ndarray:
        """Trains a Doc2Vec model and returns document embeddings."""
        logging.info("Training Doc2Vec...")
        tagged_data = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(texts)]
        model = Doc2Vec(tagged_data, vector_size=vector_size, window=5, min_count=1, workers=4, epochs=20)
        return np.array([model.dv[i] for i in range(len(texts))])
    
    def transform(self, texts: List[Union[str, List[str]]]) -> np.ndarray:
        """Transforms input text based on the selected method."""
        if self.method == "tfidf":
            return self.fit_tfidf(texts)
        elif self.method == "word2vec":
            return self.fit_word2vec(texts)
        elif self.method == "bert":
            return self.fit_bert(texts)
        elif self.method == "doc2vec":
            return self.fit_doc2vec(texts)
        else:
            raise ValueError("Unsupported vectorization method.")
    

def main():
    """Example usage of the TextVectorizer class."""
    # Load dataset
    df = pd.read_csv("../../data/cleaned_news.csv")
    texts = df["short_description"].tolist()
    
    # TF-IDF
    vectorizer = TextVectorizer(method="tfidf")
    tfidf_vectors = vectorizer.transform(texts)
    logging.info(f"TF-IDF vector shape: {tfidf_vectors.shape}")
    
    # BERT
    vectorizer = TextVectorizer(method="bert")
    bert_vectors = vectorizer.transform(texts)
    logging.info(f"BERT vector shape: {bert_vectors.shape}")
    
    # Word2Vec
    tokenized_texts = [text.split() for text in texts]
    vectorizer = TextVectorizer(method="word2vec")
    w2v_vectors = vectorizer.transform(tokenized_texts)
    logging.info(f"Word2Vec vector shape: {w2v_vectors.shape}")
    
    # Doc2Vec
    vectorizer = TextVectorizer(method="doc2vec")
    doc2vec_vectors = vectorizer.transform(tokenized_texts)
    logging.info(f"Doc2Vec vector shape: {doc2vec_vectors.shape}")

if __name__ == "__main__":
    main()

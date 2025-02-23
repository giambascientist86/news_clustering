import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
DATA_PATH = Path("C:/Users/batti/topic-modeling/data/Huff_news/news.jsonl")

# --- data_loader.py ---
class DataLoader:
    """Loads and preprocesses the dataset."""
    def __init__(self, file_path: Path):
        self.file_path = file_path
    
    def load_data(self) -> pd.DataFrame:
        """Loads JSONL dataset into a Pandas DataFrame."""
        try:
            data = [json.loads(line) for line in open(self.file_path, 'r', encoding='utf-8')]
            df = pd.DataFrame(data)
            logging.info("Dataset loaded successfully.")
            return df
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise

if __name__ == "__main__":
    # Load Data
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
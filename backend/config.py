import os

DATASET_PATH = os.getenv("DATASET_PATH", r"C:\Users\batti\topic-modeling\data\cleaned_news.csv")
MODEL_PATH = os.getenv("MODEL_PATH", r"C:\Users\batti\topic-modeling\src\models\bertopic_model")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

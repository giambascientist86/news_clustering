import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from config import DATASET_PATH, MODEL_PATH, EMBEDDING_MODEL

class TopicModelingService:
    def __init__(self):
        self.texts = self._load_data()
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.umap_model = umap.UMAP(n_neighbors=10, n_components=2, metric='cosine', random_state=42)
        self.hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=100, metric='euclidean', prediction_data=True)
        self.vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.9, min_df=5)
        
        try:
            self.model = BERTopic.load(MODEL_PATH)
        except:
            print("Modello non trovato, inizializzo un nuovo BERTopic.")
            self.model = BERTopic(embedding_model=self.embedding_model, 
                                  umap_model=self.umap_model,
                                  hdbscan_model=self.hdbscan_model, 
                                  vectorizer_model=self.vectorizer_model)
    def load_model(self):
        self.model = BERTopic.load(MODEL_PATH)
        
    def _load_data(self) -> list:
        df = pd.read_csv(DATASET_PATH)
        return df["short_description"].dropna().astype(str).tolist()

    def train_model(self):
        topics, _ = self.model.fit_transform(self.texts)
        self.model.save(MODEL_PATH)
        return "Model has been trained successfully!"

    def get_topics(self):
        if not hasattr(self.model, "topics_") or not self.model.topics_:
            try:
                self.load_model()
            except:
                raise ValueError("Il modello BERTopic non è stato addestrato o non può essere caricato.")

        if not hasattr(self.model, "topics_") or not self.model.topics_:
            raise ValueError("Il modello è stato caricato, ma non ha ancora identificato nessun topic.")

        return self.model.get_topic_info().head(5).to_dict(orient="records")
 


    

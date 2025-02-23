import logging
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EmbeddingEvaluator:
    """
    A class to evaluate the quality of embeddings efficiently.
    """
    def __init__(self, embeddings_path: str, n_clusters: int = 5, reduce_dim: int = 50):
        """
        Initialize the evaluator with reduced embeddings.
        """
        self.embeddings_path = embeddings_path
        self.embeddings = self.load_embeddings()
        self.n_clusters = n_clusters
        self.reduced_embeddings = self.reduce_dimensionality(reduce_dim)

    def load_embeddings(self) -> np.ndarray:
        """Loads embeddings from a .npy file."""
        try:
            embeddings = np.load(self.embeddings_path)
            if not isinstance(embeddings, np.ndarray):
                raise ValueError("Loaded embeddings are not a NumPy array.")
            logging.info(f"✅ Embeddings loaded. Shape: {embeddings.shape}")
            return embeddings.astype(np.float32)  # Ensure float32 for FAISS
        except Exception as e:
            logging.error(f"❌ Error loading embeddings: {e}")
            raise

    def reduce_dimensionality(self, n_components: int = 50) -> np.ndarray:
        """Reduce dimensionality using PCA to speed up clustering."""
        logging.info(f"Reducing embeddings to {n_components} dimensions...")
        pca = PCA(n_components=n_components)
        return pca.fit_transform(self.embeddings)

    def cosine_similarity_faiss(self, top_k: int = 10) -> np.ndarray:
        """Compute cosine similarity using Faiss."""
        logging.info("Computing cosine similarity with Faiss...")
        d = self.reduced_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(self.reduced_embeddings)  # Normalize before adding
        index.add(self.reduced_embeddings)
        _, indices = index.search(self.reduced_embeddings, top_k)
        return indices

    def nearest_neighbors_consistency(self, n_neighbors: int = 5) -> float:
        """Efficient k-NN consistency using Faiss."""
        logging.info("Computing nearest neighbor consistency...")
        index = faiss.IndexFlatL2(self.reduced_embeddings.shape[1])
        index.add(self.reduced_embeddings)
        _, distances = index.search(self.reduced_embeddings, n_neighbors + 1)
        return np.mean(distances[:, 1:])  # Exclude self-distance

    def evaluate_clustering_quality(self) -> dict:
        """Cluster with MiniBatchKMeans and compute quality metrics."""
        logging.info("Evaluating clustering quality...")
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=1000, random_state=42)
        labels = kmeans.fit_predict(self.reduced_embeddings)
        return {
            "silhouette_score": silhouette_score(self.reduced_embeddings, labels),
            "davies_bouldin_index": davies_bouldin_score(self.reduced_embeddings, labels),
            "calinski_harabasz_score": calinski_harabasz_score(self.reduced_embeddings, labels),
        }

# Example usage
if __name__ == "__main__":
    evaluator = EmbeddingEvaluator(r"C:\Users\batti\topic-modeling\data\tfidf_embeddings.npy")
    logging.info(f"Nearest Neighbor Consistency: {evaluator.nearest_neighbors_consistency():.4f}")
    logging.info(f"Clustering Quality: {evaluator.evaluate_clustering_quality()}")
